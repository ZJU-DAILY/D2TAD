#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    ProtoSequenceDataset,
    resolve_missing_placeholder_token,
    resolve_special_tokens,
)
from load_model import load_model_local

SAMPLING_EPS_DEFAULT = 1e-5
DEFAULT_FIRST_STEP_STEPS = 10000
DEFAULT_DETECTION_T_VALUE = 1.0 - (
    (1.0 - SAMPLING_EPS_DEFAULT) / float(DEFAULT_FIRST_STEP_STEPS)
)


def _load_outlier_ids(path: Optional[str], total: int) -> Optional[np.ndarray]:
    if not path:
        return None
    raw = np.load(Path(path).expanduser(), allow_pickle=True)
    labels = np.zeros(total, dtype=np.int64)
    ids = np.asarray(raw, dtype=np.int64).reshape(-1)
    ids = ids[(ids >= 0) & (ids < total)]
    labels[ids] = 1
    return labels


def _topk_aggregate(
    scores: torch.Tensor, mask: torch.Tensor, topk_ratio: float
) -> torch.Tensor:
    ratio = min(max(float(topk_ratio), 0.0), 1.0)
    if ratio <= 0.0:
        raise ValueError("topk_ratio must be in (0, 1].")
    out = []
    for row, row_mask in zip(scores, mask):
        valid_scores = row[row_mask]
        if valid_scores.numel() == 0:
            out.append(scores.new_tensor(0.0))
            continue
        k = int(
            torch.ceil(valid_scores.new_tensor(valid_scores.numel() * ratio)).item()
        )
        k = max(1, min(k, valid_scores.numel()))
        out.append(
            torch.topk(valid_scores, k=k, largest=True, sorted=False).values.mean()
        )
    return torch.stack(out)


def _logrank_of_true_token(
    logits: torch.Tensor, true_tokens: torch.Tensor
) -> torch.Tensor:
    true_logits = logits.gather(-1, true_tokens[..., None]).squeeze(-1)
    rank = (logits > true_logits[..., None]).sum(dim=-1) + 1
    return rank.float().log()


def _resolve_mask_runs(mask_ratio: float, mask_runs: Optional[int]) -> int:
    ratio = min(max(float(mask_ratio), 0.0), 1.0)
    if ratio <= 0.0:
        raise ValueError("mask_ratio must be in (0, 1].")
    if mask_runs is not None:
        runs = int(mask_runs)
        if runs <= 0:
            raise ValueError("mask_runs must be positive.")
        return runs
    return max(1, int(round(1.0 / ratio)))


def _build_covering_score_masks(
    score_mask: torch.Tensor,
    mask_ratio: float,
    mask_runs: Optional[int],
    generator: Optional[torch.Generator] = None,
) -> list[torch.Tensor]:
    runs = _resolve_mask_runs(mask_ratio, mask_runs)
    run_masks = [torch.zeros_like(score_mask, dtype=torch.bool) for _ in range(runs)]
    for row_idx in range(score_mask.size(0)):
        candidates = torch.nonzero(score_mask[row_idx], as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            continue
        if runs == 1:
            run_masks[0][row_idx, candidates] = True
            continue
        perm = torch.randperm(
            candidates.numel(),
            device=score_mask.device,
            generator=generator,
        )
        chunks = torch.chunk(candidates[perm], runs)
        for run_idx, chunk in enumerate(chunks):
            if chunk.numel() > 0:
                run_masks[run_idx][row_idx, chunk] = True
    return run_masks


def _score_parallel_mask_runs(
    model,
    graph,
    tokens: torch.Tensor,
    run_masks: list[torch.Tensor],
    sigma_batch: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_masks = [mask for mask in run_masks if bool(mask.any())]
    point_scores = torch.zeros(tokens.shape, dtype=torch.float32, device=tokens.device)
    scored_mask = torch.zeros(tokens.shape, dtype=torch.bool, device=tokens.device)
    if not active_masks:
        return point_scores, scored_mask

    mask_stack = torch.stack(active_masks, dim=0).to(device=tokens.device)
    groups = mask_stack.size(0)
    flat_mask = mask_stack.reshape(groups * tokens.size(0), tokens.size(1))
    repeated_tokens = tokens.repeat(groups, 1)
    repeated_sigma = sigma_batch.repeat(groups)
    masked_tokens = torch.where(
        flat_mask,
        graph.sample_limit(*repeated_tokens.shape).to(tokens.device),
        repeated_tokens,
    )
    logits = model(masked_tokens, repeated_sigma)[..., : int(vocab_size)].float()
    flat_scores = _logrank_of_true_token(
        logits,
        repeated_tokens.clamp_max(int(vocab_size) - 1),
    )
    grouped_scores = flat_scores.reshape(groups, tokens.size(0), tokens.size(1))
    point_scores = torch.where(
        mask_stack,
        grouped_scores,
        torch.zeros_like(grouped_scores),
    ).sum(dim=0)
    scored_mask = mask_stack.any(dim=0)
    return point_scores, scored_mask


def _write_scores(path: Path, scores: np.ndarray, labels: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["index", "score"]
        if labels is not None:
            header.append("label")
        writer.writerow(header)
        for idx, score in enumerate(scores):
            row = [idx, float(score)]
            if labels is not None:
                row.append(int(labels[idx]))
            writer.writerow(row)


def _compute_metrics(
    scores: np.ndarray, labels: Optional[np.ndarray]
) -> dict[str, float]:
    if labels is None:
        return {}
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return {}

    return {"pr_auc": float(average_precision_score(labels, scores))}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trajectory anomaly detection with logrank scoring."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--outlier_ids_path", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default="detection_scores.csv")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.33,
        help="Artificial mask ratio. By default this creates round(1 / ratio) covering mask runs.",
    )
    parser.add_argument(
        "--mask_runs",
        type=int,
        default=None,
        help="Optional explicit number of artificial covering mask runs.",
    )
    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=0.1,
        help="Aggregate each trajectory by averaging the top ratio of point logrank scores.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 < float(args.topk_ratio) <= 1.0):
        raise ValueError("topk_ratio must be in (0, 1].")
    if not (0.0 < float(args.mask_ratio) <= 1.0):
        raise ValueError("mask_ratio must be in (0, 1].")
    if args.mask_runs is not None and int(args.mask_runs) <= 0:
        raise ValueError("mask_runs must be positive.")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    mask_generator = torch.Generator(device=device).manual_seed(int(args.seed))
    model, graph, _noise = load_model_local(args.model_path, device)
    model.eval()
    cfg = model.config
    pad_token, eos_token = resolve_special_tokens(cfg)
    missing_token = resolve_missing_placeholder_token(cfg)
    dataset = ProtoSequenceDataset(
        Path(args.dataset_path),
        block_size=int(cfg.model.length),
        pad_token=pad_token,
        eos_token=eos_token,
        missing_placeholder_token=missing_token,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
    )

    all_scores = []
    sigma = torch.full(
        (int(args.batch_size),), DEFAULT_DETECTION_T_VALUE, device=device
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc="scoring"):
            tokens = batch["input_ids"].to(device)
            valid_mask = batch.get("valid_mask")
            if valid_mask is None:
                valid_mask = torch.ones_like(tokens, dtype=torch.bool)
            else:
                valid_mask = valid_mask.to(device)
            score_mask = valid_mask & (tokens >= 0) & (tokens < int(cfg.tokens))
            if missing_token is not None:
                score_mask = score_mask & (tokens != int(missing_token))
            run_masks = _build_covering_score_masks(
                score_mask,
                mask_ratio=float(args.mask_ratio),
                mask_runs=args.mask_runs,
                generator=mask_generator,
            )
            sigma_batch = sigma[: tokens.size(0)]
            if sigma_batch.size(0) != tokens.size(0):
                sigma_batch = torch.full(
                    (tokens.size(0),), DEFAULT_DETECTION_T_VALUE, device=device
                )
            point_scores, scored_mask = _score_parallel_mask_runs(
                model=model,
                graph=graph,
                tokens=tokens,
                run_masks=run_masks,
                sigma_batch=sigma_batch,
                vocab_size=int(cfg.tokens),
            )
            all_scores.append(
                _topk_aggregate(point_scores, scored_mask, args.topk_ratio)
            )

    scores = torch.cat(all_scores).detach().cpu().numpy()
    labels = _load_outlier_ids(args.outlier_ids_path, total=scores.shape[0])
    output_csv = Path(args.output_csv).expanduser().resolve()
    _write_scores(output_csv, scores, labels)
    metrics = _compute_metrics(scores, labels)
    if metrics:
        metrics_path = output_csv.with_suffix(".metrics.csv")
        with metrics_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
        print(f"saved_metrics={metrics_path}")
    print(f"saved_scores={output_csv}")


if __name__ == "__main__":
    main()
