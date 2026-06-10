#!/usr/bin/env python3
import argparse
import datetime as dt
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import graph_lib
import noise_lib
import utils
from data import ProtoSequenceDataset, resolve_special_tokens
from load_model import _resolve_checkpoint_path, _resolve_run_root
from model import SEDD
from model.ema import ExponentialMovingAverage

SAMPLING_EPS_DEFAULT = 1e-5
DEFAULT_FIRST_STEP_STEPS = 10000


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dataloader_from_proto(
    dataset_path: str,
    block_size: int,
    batch_size: int,
    pad_token: int,
    eos_token: int,
    num_workers: int,
) -> DataLoader:
    dataset = ProtoSequenceDataset(
        Path(dataset_path),
        block_size=block_size,
        pad_token=pad_token,
        eos_token=eos_token,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=True,
    )


def _make_output_dir(base_run_root: Path, out_dir: Optional[str], tag: str) -> Path:
    if out_dir:
        path = Path(out_dir).expanduser().resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (base_run_root / "step0_grpo_runs" / f"{stamp}_{tag}").resolve()
    path.mkdir(parents=True, exist_ok=True)

    hydra_src = base_run_root / ".hydra"
    hydra_dst = path / ".hydra"
    if hydra_src.exists() and not hydra_dst.exists():
        shutil.copytree(hydra_src, hydra_dst)
    return path


def _resolve_transition_path(cfg, override_path: Optional[str]) -> Optional[Path]:
    candidates = []
    if override_path:
        candidates.append(Path(override_path).expanduser())
    cfg_path = getattr(cfg.graph, "transition_path", None)
    if cfg_path:
        candidates.append(Path(str(cfg_path)).expanduser())
        candidates.append(Path.cwd() / str(cfg_path))
    proto_cfg = getattr(cfg.data, "proto", None)
    if proto_cfg is not None and cfg_path:
        base_dir = getattr(proto_cfg, "base_dir", None)
        if base_dir:
            candidates.append(Path(str(base_dir)).expanduser() / str(cfg_path))
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _load_transition_matrix(
    cfg,
    vocab_size: int,
    override_path: Optional[str],
) -> Optional[torch.Tensor]:
    path = _resolve_transition_path(cfg, override_path)
    if path is None:
        return None
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        matrix = obj.get("transition_prob", obj.get("transition"))
        if matrix is None:
            raise KeyError(f"Unsupported transition file format: {path}")
    else:
        matrix = obj
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Transition matrix must be square, got {tuple(matrix.shape)}")
    if matrix.shape[0] < vocab_size:
        raise ValueError(
            f"Transition matrix has vocab {matrix.shape[0]} but needs {vocab_size}"
        )
    matrix = matrix[:vocab_size, :vocab_size].clamp_min(1e-12)
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _first_step_x0_distribution(model, graph, noise, x, t, eps: float = 1e-12):
    if not graph.absorb:
        raise ValueError("Step-0 GRPO requires absorb graph.")
    sigma, _ = noise(t)
    logits = model(x, sigma.reshape(-1))
    logits = logits[..., : graph.dim - 1].float()
    return F.softmax(logits, dim=-1).clamp_min(eps)


def _masked_match_reward(sampled_tokens, true_tokens, masked_positions):
    masked_counts = masked_positions.sum(dim=-1).float()
    matches = ((sampled_tokens == true_tokens) & masked_positions).float().sum(dim=-1)
    return torch.where(
        masked_counts > 0,
        matches / masked_counts.clamp_min(1.0),
        torch.zeros_like(matches),
    )


def _build_block_mask(batch_size, seq_len, mask_ratio, device, valid_mask):
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    ratio = min(max(float(mask_ratio), 0.0), 1.0)
    if ratio <= 0.0:
        return mask
    for idx in range(batch_size):
        candidates = (
            torch.nonzero(valid_mask[idx], as_tuple=False).squeeze(1)
            if valid_mask is not None
            else torch.arange(seq_len, device=device)
        )
        valid_len = int(candidates.numel())
        if valid_len <= 0:
            continue
        block_len = max(1, min(int(round(valid_len * ratio)), valid_len))
        start = 0
        if block_len < valid_len:
            start = int(
                torch.randint(0, valid_len - block_len + 1, (1,), device=device)
            )
        mask[idx, candidates[start : start + block_len]] = True
    return mask


def _build_suffix_mask(batch_size, seq_len, mask_ratio, device, valid_mask):
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    ratio = min(max(float(mask_ratio), 0.0), 1.0)
    if ratio <= 0.0:
        return mask
    for idx in range(batch_size):
        candidates = (
            torch.nonzero(valid_mask[idx], as_tuple=False).squeeze(1)
            if valid_mask is not None
            else torch.arange(seq_len, device=device)
        )
        valid_len = int(candidates.numel())
        if valid_len <= 0:
            continue
        suffix_len = max(1, min(int(round(valid_len * ratio)), valid_len))
        mask[idx, candidates[-suffix_len:]] = True
    return mask


def _sample_training_mask(
    batch_size, seq_len, mask_ratio, mask_mode, device, valid_mask
):
    random_mask = torch.rand(batch_size, seq_len, device=device) < float(mask_ratio)
    if valid_mask is not None:
        random_mask = random_mask & valid_mask
    if mask_mode == "random":
        return random_mask

    block_mask = _build_block_mask(batch_size, seq_len, mask_ratio, device, valid_mask)
    suffix_mask = _build_suffix_mask(
        batch_size, seq_len, mask_ratio, device, valid_mask
    )
    if mask_mode == "block":
        return block_mask
    if mask_mode == "suffix":
        return suffix_mask
    if mask_mode == "hybrid":
        mode_ids = torch.randint(0, 3, (batch_size,), device=device)
        out = random_mask.clone()
        out = torch.where((mode_ids == 1).unsqueeze(1), block_mask, out)
        out = torch.where((mode_ids == 2).unsqueeze(1), suffix_mask, out)
        return out
    raise ValueError(f"Unsupported mask_mode: {mask_mode}")


def _transition_log_reward(
    sampled_tokens,
    masked_positions,
    transition_prob,
    valid_mask,
    masked_edges_only=True,
):
    if transition_prob is None or sampled_tokens.size(1) < 2:
        return sampled_tokens.new_zeros(sampled_tokens.size(0), dtype=torch.float32)
    left = sampled_tokens[:, :-1]
    right = sampled_tokens[:, 1:]
    edge_mask = torch.ones_like(left, dtype=torch.bool)
    if valid_mask is not None:
        edge_mask = edge_mask & valid_mask[:, :-1] & valid_mask[:, 1:]
    if masked_edges_only:
        edge_mask = edge_mask & (masked_positions[:, :-1] | masked_positions[:, 1:])
    edge_scores = transition_prob[left, right].clamp_min(1e-12).log()
    edge_scores = edge_scores.masked_fill(~edge_mask, 0.0)
    edge_counts = edge_mask.sum(dim=-1).float()
    edge_sum = edge_scores.sum(dim=-1)
    return torch.where(
        edge_counts > 0,
        edge_sum / edge_counts.clamp_min(1.0),
        torch.zeros_like(edge_sum),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step-0 mask-GRPO fine-tuning with transition reward."
    )
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--tag", type=str, default="grpo_mask")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="random",
        choices=["random", "block", "suffix", "hybrid"],
    )
    parser.add_argument("--n_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1.5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=3)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--reward_match_weight", type=float, default=0.45)
    parser.add_argument("--reward_transition_weight", type=float, default=0.3)
    parser.add_argument("--reward_transition_path", type=str, default=None)
    parser.add_argument(
        "--reward_transition_masked_edges_only",
        action="store_true",
        default=True,
        help="Kept for compatibility. Masked-edge transition reward is enabled by default.",
    )
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.mask_ratio <= 1.0):
        raise ValueError("mask_ratio must be in (0, 1].")
    if args.group_size <= 1:
        raise ValueError("group_size must be > 1.")
    if args.ppo_epochs <= 0:
        raise ValueError("ppo_epochs must be > 0.")

    _set_seed(int(args.seed))
    device = torch.device(args.device)
    base_path = Path(args.base_model_path).expanduser().resolve()
    run_root = _resolve_run_root(base_path)
    cfg = utils.load_hydra_config_from_run(run_root)
    out_dir = _make_output_dir(run_root, args.out_dir, args.tag)
    logger = utils.get_logger(str(out_dir / "train_step0_grpo_mask.log"))

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    if not graph.absorb:
        raise ValueError("This script requires absorb graph.")
    transition_prob = _load_transition_matrix(
        cfg, graph.dim - 1, args.reward_transition_path
    )
    if transition_prob is not None:
        transition_prob = transition_prob.to(device)

    ckpt_path = _resolve_checkpoint_path(base_path, run_root)
    state = torch.load(ckpt_path, map_location=device)
    model = SEDD(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    ema = ExponentialMovingAverage(model.parameters(), decay=float(cfg.training.ema))
    if "ema" in state:
        ema.load_state_dict(state["ema"])
        ema.copy_to(model.parameters())
    model.train()

    model_ref = SEDD(cfg).to(device)
    model_ref.load_state_dict(model.state_dict(), strict=False)
    for param in model_ref.parameters():
        param.requires_grad_(False)
    model_ref.eval()

    seq_len = int(cfg.model.length)
    pad_token, eos_token = resolve_special_tokens(cfg)
    loader = _dataloader_from_proto(
        args.dataset_path,
        block_size=seq_len,
        batch_size=int(args.batch_size),
        pad_token=pad_token,
        eos_token=eos_token,
        num_workers=int(args.num_workers),
    )
    loader_iter = iter(loader)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    dt_value = (1.0 - SAMPLING_EPS_DEFAULT) / float(DEFAULT_FIRST_STEP_STEPS)
    t_min = SAMPLING_EPS_DEFAULT
    t_val = max(t_min, 1.0 - dt_value)
    t_val = min(max(t_val, t_min), 1.0 - 1e-6)
    start_step = 0

    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser().resolve()
        try:
            resume_run_root = _resolve_run_root(resume_path)
            resume_ckpt = _resolve_checkpoint_path(resume_path, resume_run_root)
        except FileNotFoundError:
            resume_ckpt = (
                resume_path
                if resume_path.is_file()
                else resume_path / "checkpoints-meta" / "checkpoint.pth"
            )
        resume_state = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(resume_state["model"], strict=False)
        if "ema" in resume_state:
            ema.load_state_dict(resume_state["ema"])
            ema.copy_to(model.parameters())
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        start_step = int(resume_state.get("step", 0))
        model.train()

    checkpoints_dir = out_dir / "checkpoints"
    ckpt_meta_dir = out_dir / "checkpoints-meta"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_meta_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting step-0 GRPO fine-tuning.")

    def next_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter)

    for step in range(start_step + 1, int(args.n_iters) + 1):
        batch = next_batch()
        tokens = batch["input_ids"].to(device)
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)

        bs = tokens.shape[0]
        mask = _sample_training_mask(
            batch_size=bs,
            seq_len=seq_len,
            mask_ratio=float(args.mask_ratio),
            mask_mode=args.mask_mode,
            device=device,
            valid_mask=valid_mask,
        )
        total_masked = int(mask.sum().item())
        if total_masked == 0:
            continue
        keep_mask = ~mask

        x_t = graph.sample_limit(bs, seq_len).to(device)
        x_t = torch.where(keep_mask, tokens, x_t)
        t0 = torch.full((bs, 1), t_val, device=device)

        with torch.no_grad():
            probs_old = _first_step_x0_distribution(model, graph, noise, x_t, t0)
            probs_ref = _first_step_x0_distribution(model_ref, graph, noise, x_t, t0)

        flat_probs = probs_old[mask]
        flat_probs_ref = probs_ref[mask]
        if flat_probs.numel() == 0:
            continue

        seq_ids = torch.arange(bs, device=device).unsqueeze(1).expand(bs, seq_len)[mask]
        ones = torch.ones_like(seq_ids, dtype=torch.float32)
        masked_counts = torch.zeros(bs, device=device).scatter_add_(0, seq_ids, ones)
        masked_counts = masked_counts.clamp_min(1.0)

        actions_matrix = torch.multinomial(
            flat_probs,
            num_samples=int(args.group_size),
            replacement=True,
        )
        logp_matrix = torch.log(flat_probs.gather(1, actions_matrix).clamp_min(1e-12))
        logp_ref_matrix = torch.log(
            flat_probs_ref.gather(1, actions_matrix).clamp_min(1e-12)
        )

        logp_old = torch.zeros(int(args.group_size), bs, device=device)
        logp_ref = torch.zeros(int(args.group_size), bs, device=device)
        for group_idx in range(int(args.group_size)):
            logp_old[group_idx].scatter_add_(0, seq_ids, logp_matrix[:, group_idx])
            logp_ref[group_idx].scatter_add_(0, seq_ids, logp_ref_matrix[:, group_idx])
        logp_old = logp_old / masked_counts
        logp_ref = logp_ref / masked_counts
        logp_old_detached = logp_old.detach()

        actions_g = actions_matrix.t().contiguous()
        x0_hat = x_t.repeat(int(args.group_size), 1)
        mask_g = mask.repeat(int(args.group_size), 1)
        x0_hat[mask_g] = actions_g.reshape(-1)
        x0_hat = torch.where(
            keep_mask.repeat(int(args.group_size), 1),
            tokens.repeat(int(args.group_size), 1),
            x0_hat,
        )

        tokens_cat = tokens.repeat(int(args.group_size), 1)
        valid_mask_cat = (
            valid_mask.repeat(int(args.group_size), 1)
            if valid_mask is not None
            else None
        )
        reward_match = _masked_match_reward(x0_hat, tokens_cat, mask_g).view(
            int(args.group_size), bs
        )
        reward_transition = _transition_log_reward(
            sampled_tokens=x0_hat,
            masked_positions=mask_g,
            transition_prob=transition_prob,
            valid_mask=valid_mask_cat,
            masked_edges_only=bool(args.reward_transition_masked_edges_only),
        ).view(int(args.group_size), bs)
        rewards = (
            float(args.reward_match_weight) * reward_match
            + float(args.reward_transition_weight) * reward_transition
        )
        reward_mean = rewards.mean(dim=0, keepdim=True)
        reward_std = rewards.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        advantages = ((rewards - reward_mean) / reward_std).detach()

        for _ in range(int(args.ppo_epochs)):
            probs_new = _first_step_x0_distribution(model, graph, noise, x_t, t0)
            flat_probs_new = probs_new[mask]
            logp_new_matrix = torch.log(
                flat_probs_new.gather(1, actions_matrix).clamp_min(1e-12)
            )
            logp_new = torch.zeros(int(args.group_size), bs, device=device)
            for group_idx in range(int(args.group_size)):
                logp_new[group_idx].scatter_add_(
                    0, seq_ids, logp_new_matrix[:, group_idx]
                )
            logp_new = logp_new / masked_counts

            ratio = torch.exp(logp_new - logp_old_detached)
            ratio_clipped = ratio.clamp(
                1.0 - float(args.clip_eps), 1.0 + float(args.clip_eps)
            )
            loss_pg = -torch.min(ratio * advantages, ratio_clipped * advantages).mean()

            kl = (
                flat_probs_new
                * (
                    torch.log(flat_probs_new.clamp_min(1e-12))
                    - torch.log(flat_probs_ref.clamp_min(1e-12))
                )
            ).sum(dim=1)
            entropy = -(
                flat_probs_new * torch.log(flat_probs_new.clamp_min(1e-12))
            ).sum(dim=1)
            kl_sum = torch.zeros(bs, device=device).scatter_add_(0, seq_ids, kl)
            entropy_sum = torch.zeros(bs, device=device).scatter_add_(
                0, seq_ids, entropy
            )
            kl_mean = kl_sum / masked_counts
            entropy_mean = entropy_sum / masked_counts
            loss = (
                loss_pg
                + float(args.kl_coef) * kl_mean.mean()
                - float(args.entropy_coef) * entropy_mean.mean()
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(args.grad_clip)
                )
            optimizer.step()
            ema.update(model.parameters())

        if step % int(args.log_every) == 0 or step == 1:
            logger.info(
                "step=%d loss=%.6f reward=%.6f match=%.6f transition=%.6f "
                "kl=%.6f entropy=%.6f masked=%d",
                step,
                float(loss.item()),
                float(rewards.mean().item()),
                float(reward_match.mean().item()),
                float(reward_transition.mean().item()),
                float(kl_mean.mean().item()),
                float(entropy_mean.mean().item()),
                total_masked,
            )

        if step % int(args.save_every) == 0 or step == int(args.n_iters):
            payload = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "base_checkpoint": str(ckpt_path),
                "step": step,
                "grpo": {
                    "group_size": int(args.group_size),
                    "ppo_epochs": int(args.ppo_epochs),
                    "clip_eps": float(args.clip_eps),
                    "kl_coef": float(args.kl_coef),
                    "entropy_coef": float(args.entropy_coef),
                    "mask_ratio": float(args.mask_ratio),
                    "mask_mode": str(args.mask_mode),
                    "reward_match_weight": float(args.reward_match_weight),
                    "reward_transition_weight": float(args.reward_transition_weight),
                    "reward_transition_masked_edges_only": bool(
                        args.reward_transition_masked_edges_only
                    ),
                    "t_value": float(t_val),
                },
            }
            save_path = checkpoints_dir / f"step0_grpo_step{step}.pth"
            meta_path = ckpt_meta_dir / "checkpoint.pth"
            torch.save(payload, save_path)
            torch.save(payload, meta_path)
            logger.info("Saved checkpoint: %s", save_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
