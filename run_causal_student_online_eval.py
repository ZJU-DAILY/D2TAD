#!/usr/bin/env python3
from __future__ import annotations
import argparse
import bisect
import math
import os
import pathlib
import time
from collections import deque
from typing import Optional

import noise_lib
import numpy as np
import torch
import utils
from graph_lib import get_graph
from torch.utils.data import DataLoader
from train_autoregressive_distill import CausalSEDDStudent

from data import (
    ProtoSequenceDataset,
    resolve_missing_placeholder_token,
    resolve_special_tokens,
)

SAMPLING_EPS_DEFAULT = 1e-5
DEFAULT_FIRST_STEP_STEPS = 10000

try:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve
except Exception:
    average_precision_score = None
    precision_recall_curve = None
    auc = None


def dataloader_from_proto(
    path: str,
    block_size: int,
    batch_size: int,
    pad_token: int,
    eos_token: int,
    num_workers: int,
    pin_memory: bool,
    missing_placeholder_token: int | None,
) -> DataLoader:
    dataset = ProtoSequenceDataset(
        pathlib.Path(path),
        block_size=block_size,
        pad_token=pad_token,
        eos_token=eos_token,
        missing_placeholder_token=missing_placeholder_token,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers > 0),
    )


def load_student_checkpoint(path: str, device: torch.device):
    ckpt_path = pathlib.Path(path).expanduser().resolve()
    state = torch.load(ckpt_path, map_location=device)
    cfg = state["config"]
    graph = get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    student = CausalSEDDStudent(cfg=cfg).to(device)
    student.load_state_dict(state["student"], strict=True)
    student.eval()
    return student, graph, noise, cfg, ckpt_path


def load_teacher_checkpoint(path: str, device: torch.device):
    from load_model import load_model_local
    teacher, graph, noise = load_model_local(path, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, graph, noise, teacher.config, pathlib.Path(path)


def load_outlier_mask(path: Optional[str], n: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr).reshape(-1)
    mask = np.zeros(n, dtype=bool)
    idx = arr.astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    mask[idx] = True
    return mask


def parse_ratio_list(spec: str) -> list[float]:
    vals = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("report_progress_ratios is empty.")
    return sorted(set(vals))


def batched_student_logits(student, graph, noise, x, t_scalar: float, chunk_size: int):
    if x.size(0) == 0:
        return x.new_zeros((0, x.size(1), graph.dim - 1), dtype=torch.float32)
    outputs = []
    chunk = max(1, int(chunk_size))
    for start in range(0, x.size(0), chunk):
        end = min(start + chunk, x.size(0))
        x_chunk = x[start:end]
        t = torch.full((x_chunk.size(0),), float(t_scalar), device=x.device)
        logits = student(x_chunk, t).float()
        outputs.append(logits[..., : graph.dim - 1])
    return torch.cat(outputs, dim=0)


def batched_teacher_logits(teacher, graph, noise, x, t_scalar: float, chunk_size: int):
    if x.size(0) == 0:
        return x.new_zeros((0, x.size(1), graph.dim - 1), dtype=torch.float32)
    outputs = []
    chunk = max(1, int(chunk_size))
    sigma = float(t_scalar)
    for start in range(0, x.size(0), chunk):
        end = min(start + chunk, x.size(0))
        x_chunk = x[start:end]
        t_batch = torch.full((x_chunk.size(0),), sigma, device=x.device)
        with torch.inference_mode():
            logits = teacher(x_chunk, t_batch).float()
        outputs.append(logits[..., : graph.dim - 1])
    return torch.cat(outputs, dim=0)


def query_logranks(
    logits: torch.Tensor, target_positions: torch.Tensor, true_tokens: torch.Tensor
):
    if logits.size(0) == 0:
        empty = logits.new_zeros((0,), dtype=torch.float32)
        empty_i = logits.new_zeros((0,), dtype=torch.long)
        return empty, empty_i
    row_idx = torch.arange(logits.size(0), device=logits.device)
    logits_target = logits[row_idx, target_positions]
    true_logits = logits_target.gather(1, true_tokens.unsqueeze(1)).squeeze(1)
    rank = (logits_target > true_logits.unsqueeze(1)).sum(dim=1).long() + 1
    logrank = rank.float().clamp_min(1).log()
    return logrank, rank


def topk_mean_score(values: torch.Tensor, ratio: float) -> float:
    n = int(values.numel())
    if n <= 0:
        return float("nan")
    k = max(1, int(math.ceil(n * float(ratio))))
    return float(
        torch.topk(values.float(), k=k, largest=True, sorted=False).values.mean().item()
    )


def topk_mean_score_np(values: np.ndarray, ratio: float) -> float:
    n = int(values.size)
    if n <= 0:
        return float("nan")
    k = max(1, int(math.ceil(n * float(ratio))))
    if k >= n:
        return float(values.mean())
    idx = np.argpartition(values, n - k)[n - k :]
    return float(values[idx].mean())


def score_window_np(
    values: np.ndarray,
    alpha: float,
    intra_topk_ratio: float,
) -> float:
    n = int(values.size)
    if n <= 0:
        return float("nan")
    values = np.asarray(values, dtype=np.float32)
    alpha = float(alpha)
    intra_topk_ratio = float(intra_topk_ratio)
    r = max(1, int(math.ceil(n * intra_topk_ratio)))
    if r >= n:
        top_mean = float(values.mean())
    else:
        idx = np.argpartition(values, n - r)[n - r :]
        top_mean = float(values[idx].mean())
    return float(alpha * float(values.max()) + (1.0 - alpha) * top_mean)


def chunk_scores_np(
    values: np.ndarray,
    chunk_size: int,
    alpha: float,
    intra_topk_ratio: float,
) -> np.ndarray:
    n = int(values.size)
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    chunk_size = max(1, int(chunk_size))
    alpha = float(alpha)
    intra_topk_ratio = float(intra_topk_ratio)
    parts = []
    for start in range(0, n, chunk_size):
        part = values[start : start + chunk_size].astype(np.float32, copy=False)
        if int(part.size) <= 0:
            continue
        parts.append(score_window_np(part, alpha=alpha, intra_topk_ratio=intra_topk_ratio))
    if not parts:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(parts, dtype=np.float32)


def chunk_score_from_list(
    values: list[float],
    alpha: float,
    intra_topk_ratio: float,
) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float32)
    return score_window_np(
        arr, alpha=float(alpha), intra_topk_ratio=float(intra_topk_ratio)
    )


def sliding_window_scores_np(
    values: np.ndarray,
    window_size: int,
    alpha: float,
    intra_topk_ratio: float,
) -> np.ndarray:
    n = int(values.size)
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    window_size = max(1, int(window_size))
    scores = np.empty((n,), dtype=np.float32)
    for end in range(n):
        start = max(0, end + 1 - window_size)
        scores[end] = score_window_np(
            values[start : end + 1],
            alpha=float(alpha),
            intra_topk_ratio=float(intra_topk_ratio),
        )
    return scores


def blended_sliding_window_scores_np(
    values: np.ndarray,
    window_size: int,
    alpha: float,
    intra_topk_ratio: float,
    beta: float,
) -> np.ndarray:
    n = int(values.size)
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    window_scores = sliding_window_scores_np(
        values,
        window_size=window_size,
        alpha=alpha,
        intra_topk_ratio=intra_topk_ratio,
    )
    beta = float(beta)
    return beta * values + (1.0 - beta) * window_scores


class ExactSlidingWindowScore:
    def __init__(
        self,
        window_size: int,
        alpha: float,
        intra_topk_ratio: float,
    ):
        self.window_size = max(1, int(window_size))
        self.alpha = np.float32(alpha)
        self.one_minus_alpha = np.float32(1.0 - float(alpha))
        self.window_fifo = deque()
        self.window_sorted: list[float] = []
        self.window_sum = np.float32(0.0)
        self.topk_counts = [0] * (self.window_size + 1)
        for n in range(1, self.window_size + 1):
            self.topk_counts[n] = max(1, int(math.ceil(n * float(intra_topk_ratio))))

    def add(self, value: float) -> float:
        value32 = float(np.float32(value))
        if len(self.window_fifo) >= self.window_size:
            old_value = self.window_fifo.popleft()
            old_idx = bisect.bisect_left(self.window_sorted, old_value)
            self.window_sorted.pop(old_idx)
            self.window_sum = np.float32(self.window_sum - np.float32(old_value))

        self.window_fifo.append(value32)
        bisect.insort(self.window_sorted, value32)
        self.window_sum = np.float32(self.window_sum + np.float32(value32))

        n = len(self.window_sorted)
        r = self.topk_counts[n]
        if r >= n:
            top_mean = float(np.float32(self.window_sum / np.float32(n)))
        else:
            top_sum = np.float32(0.0)
            start = n - r
            for idx in range(start, n):
                top_sum = np.float32(top_sum + np.float32(self.window_sorted[idx]))
            top_mean = float(np.float32(top_sum / np.float32(r)))

        max_value = self.window_sorted[-1]
        return float(
            np.float32(
                self.alpha * np.float32(max_value)
                + self.one_minus_alpha * np.float32(top_mean)
            )
        )


class OnlineTrajectoryAccumulator:
    def __init__(
        self,
        obs_len: int,
        valid_len: int,
        topk_ratio: float,
        report_ratios: list[float],
        approx_block: int,
        chunk_score_size: int,
        chunk_score_alpha: float,
        chunk_score_topk_ratio: float,
        online_score_mode: str = "point",
        online_score_beta: float = 1.0,
    ):
        self.obs_len = int(obs_len)
        self.valid_len = int(valid_len)
        self.topk_ratio = float(topk_ratio)
        self.approx_block = int(approx_block)
        self.chunk_score_size = int(chunk_score_size)
        self.chunk_score_alpha = float(chunk_score_alpha)
        self.chunk_score_topk_ratio = float(chunk_score_topk_ratio)
        self.online_score_mode = str(online_score_mode).strip().lower()
        self.online_score_beta = float(online_score_beta)
        self.point_values: list[float] = []
        self.completed_chunk_scores: list[float] = []
        self.partial_chunk_values: list[float] = []
        self.report_targets = {}
        self.report_targets_by_count = {}
        self.report_point_scores = {}
        self.report_chunk_scores = {}
        self.window_score_state = None
        self.online_score_beta32 = np.float32(self.online_score_beta)
        self.one_minus_online_score_beta32 = np.float32(1.0 - self.online_score_beta)

        if self.online_score_mode == "sliding_window":
            self.window_score_state = ExactSlidingWindowScore(
                window_size=max(1, int(self.chunk_score_size)),
                alpha=self.chunk_score_alpha,
                intra_topk_ratio=self.chunk_score_topk_ratio,
            )

        total_scored = max(0, self.valid_len - self.obs_len)
        for ratio in report_ratios:
            threshold = min(
                self.valid_len, max(1, int(math.ceil(self.valid_len * float(ratio))))
            )
            scored_count = max(0, min(total_scored, threshold - self.obs_len))
            key = f"{float(ratio):.2f}"
            self.report_targets[key] = int(scored_count)
            self.report_targets_by_count.setdefault(int(scored_count), []).append(key)
            self.report_point_scores[key] = float("nan")
            self.report_chunk_scores[key] = float("nan")

    def _current_chunk_scores(self) -> np.ndarray:
        parts = list(self.completed_chunk_scores)
        if self.chunk_score_size > 0 and self.partial_chunk_values:
            parts.append(
                chunk_score_from_list(
                    self.partial_chunk_values,
                    alpha=self.chunk_score_alpha,
                    intra_topk_ratio=self.chunk_score_topk_ratio,
                )
            )
        if not parts:
            return np.empty((0,), dtype=np.float32)
        return np.asarray(parts, dtype=np.float32)

    def _maybe_snapshot(self):
        count = int(len(self.point_values))
        ratio_keys = self.report_targets_by_count.get(count)
        if not ratio_keys:
            return
        point_arr = None
        chunk_arr = None
        if count <= 0:
            for ratio_key in ratio_keys:
                self.report_point_scores[ratio_key] = float("nan")
                self.report_chunk_scores[ratio_key] = float("nan")
            return
        if point_arr is None:
            point_arr = np.asarray(self.point_values, dtype=np.float32)
        point_score = topk_mean_score_np(point_arr, self.topk_ratio)
        chunk_score = float("nan")
        if self.chunk_score_size > 0:
            if self.online_score_mode == "sliding_window":
                chunk_score = point_score
            else:
                if chunk_arr is None:
                    chunk_arr = self._current_chunk_scores()
                chunk_score = (
                    topk_mean_score_np(chunk_arr, self.topk_ratio)
                    if int(chunk_arr.size) > 0
                    else float("nan")
                )
        for ratio_key in ratio_keys:
            self.report_point_scores[ratio_key] = point_score
            if self.chunk_score_size > 0:
                self.report_chunk_scores[ratio_key] = chunk_score

    def add(self, score: float):
        raw_value = float(np.float32(score))
        if self.online_score_mode == "sliding_window":
            window_score = self.window_score_state.add(raw_value)
            value = float(
                np.float32(
                    self.online_score_beta32 * np.float32(raw_value)
                    + self.one_minus_online_score_beta32 * np.float32(window_score)
                )
            )
        else:
            value = raw_value
        self.point_values.append(value)
        if self.chunk_score_size > 0 and self.online_score_mode != "sliding_window":
            self.partial_chunk_values.append(value)
            if len(self.partial_chunk_values) >= self.chunk_score_size:
                self.completed_chunk_scores.append(
                    chunk_score_from_list(
                        self.partial_chunk_values,
                        alpha=self.chunk_score_alpha,
                        intra_topk_ratio=self.chunk_score_topk_ratio,
                    )
                )
                self.partial_chunk_values.clear()
        self._maybe_snapshot()

    def build_row(self):
        values = np.asarray(self.point_values, dtype=np.float32)
        num_scored = int(values.size)
        initial_values = values[: max(0, self.obs_len - 1)]
        initial_score = (
            topk_mean_score_np(initial_values, self.topk_ratio)
            if initial_values.size > 0
            else float("nan")
        )
        final_score = (
            topk_mean_score_np(values, self.topk_ratio)
            if values.size > 0
            else float("nan")
        )
        max_score = float(values.max()) if values.size > 0 else float("nan")
        row = {
            "num_scored_points": num_scored,
            "initial_online_score": initial_score,
            "final_online_score": final_score,
            "max_online_score": max_score,
            "num_refresh": (
                self.valid_len
                if self.approx_block <= 1
                else int(math.ceil(self.valid_len / float(self.approx_block)))
            ),
        }
        for ratio_key, value in self.report_point_scores.items():
            row[f"score_at_ratio_{ratio_key}"] = float(value)

        if self.chunk_score_size > 0:
            if self.online_score_mode == "sliding_window":
                row["chunk_final_online_score"] = float(final_score)
                row["chunk_max_online_score"] = float(max_score)
                row["num_chunks_scored"] = int(num_scored)
                for ratio_key, value in self.report_point_scores.items():
                    row[f"chunk_score_at_ratio_{ratio_key}"] = float(value)
            else:
                chunk_vals = self._current_chunk_scores()
                row["chunk_final_online_score"] = (
                    topk_mean_score_np(chunk_vals, self.topk_ratio)
                    if int(chunk_vals.size) > 0
                    else float("nan")
                )
                row["chunk_max_online_score"] = (
                    float(chunk_vals.max()) if int(chunk_vals.size) > 0 else float("nan")
                )
                row["num_chunks_scored"] = int(chunk_vals.size)
                for ratio_key, value in self.report_chunk_scores.items():
                    row[f"chunk_score_at_ratio_{ratio_key}"] = float(value)
        else:
            row["chunk_final_online_score"] = float("nan")
            row["chunk_max_online_score"] = float("nan")
            row["num_chunks_scored"] = 0
            for ratio_key in self.report_targets:
                row[f"chunk_score_at_ratio_{ratio_key}"] = float("nan")
        return row


def summarize_trajectory_scores(
    values,
    obs_len: int,
    valid_len: int,
    topk_ratio: float,
    report_ratios: list[float],
    approx_block: int,
    chunk_score_size: int,
    chunk_score_alpha: float,
    chunk_score_topk_ratio: float,
    online_score_mode: str = "point",
    online_score_beta: float = 1.0,
):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    raw_values = np.asarray(values, dtype=np.float32)
    raw_values = raw_values[~np.isnan(raw_values)]
    online_score_mode = str(online_score_mode).strip().lower()
    if online_score_mode == "sliding_window":
        if int(chunk_score_size) <= 0:
            raise ValueError(
                "chunk_score_size must be > 0 when online_score_mode=sliding_window."
            )
        values = blended_sliding_window_scores_np(
            raw_values,
            window_size=int(chunk_score_size),
            alpha=float(chunk_score_alpha),
            intra_topk_ratio=float(chunk_score_topk_ratio),
            beta=float(online_score_beta),
        )
    else:
        values = raw_values
    num_scored = int(values.size)
    initial_values = values[: max(0, obs_len - 1)]
    initial_score = (
        topk_mean_score_np(initial_values, topk_ratio)
        if initial_values.size > 0
        else float("nan")
    )
    final_score = (
        topk_mean_score_np(values, topk_ratio) if values.size > 0 else float("nan")
    )
    max_score = float(values.max()) if values.size > 0 else float("nan")
    row = {
        "num_scored_points": num_scored,
        "initial_online_score": initial_score,
        "final_online_score": final_score,
        "max_online_score": max_score,
        "num_refresh": (
            valid_len
            if approx_block <= 1
            else int(math.ceil(valid_len / float(approx_block)))
        ),
    }
    for ratio in report_ratios:
        threshold = min(valid_len, max(1, int(math.ceil(valid_len * ratio))))
        scored_count = max(0, min(num_scored, threshold - obs_len))
        row[f"score_at_ratio_{ratio:.2f}"] = (
            topk_mean_score_np(values[:scored_count], topk_ratio)
            if scored_count > 0
            else float("nan")
        )

    if int(chunk_score_size) > 0:
        if online_score_mode == "sliding_window":
            row["chunk_final_online_score"] = float(final_score)
            row["chunk_max_online_score"] = float(max_score)
            row["num_chunks_scored"] = int(num_scored)
            for ratio in report_ratios:
                row[f"chunk_score_at_ratio_{ratio:.2f}"] = float(
                    row[f"score_at_ratio_{ratio:.2f}"]
                )
        else:
            chunk_vals = chunk_scores_np(
                raw_values,
                chunk_size=int(chunk_score_size),
                alpha=float(chunk_score_alpha),
                intra_topk_ratio=float(chunk_score_topk_ratio),
            )
            row["chunk_final_online_score"] = (
                topk_mean_score_np(chunk_vals, topk_ratio)
                if int(chunk_vals.size) > 0
                else float("nan")
            )
            row["chunk_max_online_score"] = (
                float(chunk_vals.max())
                if int(chunk_vals.size) > 0
                else float("nan")
            )
            row["num_chunks_scored"] = int(chunk_vals.size)
            for ratio in report_ratios:
                threshold = min(valid_len, max(1, int(math.ceil(valid_len * ratio))))
                scored_count = max(0, min(num_scored, threshold - obs_len))
                partial_chunk_vals = chunk_scores_np(
                    raw_values[:scored_count],
                    chunk_size=int(chunk_score_size),
                    alpha=float(chunk_score_alpha),
                    intra_topk_ratio=float(chunk_score_topk_ratio),
                )
                row[f"chunk_score_at_ratio_{ratio:.2f}"] = (
                    topk_mean_score_np(partial_chunk_vals, topk_ratio)
                    if int(partial_chunk_vals.size) > 0
                    else float("nan")
                )
    else:
        row["chunk_final_online_score"] = float("nan")
        row["chunk_max_online_score"] = float("nan")
        row["num_chunks_scored"] = 0
        for ratio in report_ratios:
            row[f"chunk_score_at_ratio_{ratio:.2f}"] = float("nan")
    return row


def trim_kv_cache(kv_cache, batch_size: int):
    if kv_cache is None:
        return None
    keep = int(batch_size)
    trimmed = []
    for entry in kv_cache:
        if len(entry) == 3:
            k, v, cache_len = entry
            if k.size(0) == keep:
                trimmed.append(entry)
                continue
            trimmed.append((k[:keep], v[:keep], int(cache_len)))
        else:
            k, v, num_heads, head_dim, dtype, device = entry
            if k.size(0) == keep:
                trimmed.append(entry)
                continue
            trimmed.append((k[:keep], v[:keep]))
    return trimmed


def lazy_trim_kv_cache(kv_cache, new_size: int, old_size: int):
    if kv_cache is None or new_size == old_size:
        return kv_cache
    return trim_kv_cache(kv_cache, new_size)


def build_score_row_index_lists(
    sort_order_cpu: np.ndarray,
    obs_lens_sorted_cpu: list[int],
    valid_lens_sorted_cpu: list[int],
    max_valid_len: int,
) -> list[np.ndarray]:
    score_rows_by_pos = []
    total = int(len(sort_order_cpu))
    for local_idx in range(int(max_valid_len)):
        rows = [
            int(sort_order_cpu[i])
            for i in range(total)
            if obs_lens_sorted_cpu[i] <= local_idx < valid_lens_sorted_cpu[i]
        ]
        score_rows_by_pos.append(np.asarray(rows, dtype=np.int64))
    return score_rows_by_pos


def update_online_accumulators(
    accumulators: Optional[list[OnlineTrajectoryAccumulator]],
    score_rows_cpu: np.ndarray,
    logranks_cpu: np.ndarray,
):
    if accumulators is None:
        return
    for row_idx, logrank in zip(score_rows_cpu, logranks_cpu):
        accumulators[int(row_idx)].add(float(logrank))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Exact causal online evaluation for distilled autoregressive student."
    )
    parser.add_argument("--student_ckpt", type=str, required=True)
    parser.add_argument("--use_teacher", action="store_true", help="Use original SEDD teacher model instead of distilled student.")
    parser.add_argument("--teacher_model_path", type=str, default=None, help="Path to SEDD teacher checkpoint (if different from --student_ckpt).")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--outlier_idx_path", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--query_batch_size", type=int, default=1024)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--obs_ratio", type=float, default=0.0)
    parser.add_argument("--topk_ratio", type=float, default=0.1)
    parser.add_argument(
        "--chunk_score_size",
        type=int,
        default=2,
        help="If >0, also compute chunk-level anomaly scores over point logrank values.",
    )
    parser.add_argument(
        "--chunk_score_alpha",
        type=float,
        default=0.0,
        help="Chunk score = alpha * max + (1-alpha) * intra-chunk top-r mean.",
    )
    parser.add_argument(
        "--chunk_score_topk_ratio",
        type=float,
        default=1.0,
        help="Intra-chunk top-r ratio used by chunk-level scoring.",
    )
    parser.add_argument(
        "--online_score_mode",
        type=str,
        default="sliding_window",
        choices=["point", "sliding_window"],
        help="Base online score sequence: raw per-point logrank or causal sliding-window aggregation.",
    )
    parser.add_argument(
        "--online_score_beta",
        type=float,
        default=0.5,
        help="When online_score_mode=sliding_window, final point score = beta * logrank + (1-beta) * window_agg.",
    )
    parser.add_argument("--disable_kv_cache", action="store_true")
    parser.add_argument(
        "--kv_cache_block_size",
        type=int,
        default=1,
        help="Exact block size used for update-only KV prefill. Per-step scoring remains exact.",
    )
    parser.add_argument(
        "--approx_score_block_size",
        type=int,
        default=1,
        help="Approximate online scoring block size. >1 scores a whole block with one masked block forward.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument(
        "--report_progress_ratios", type=str, default="0.25,0.50,0.75,1.00"
    )
    args = parser.parse_args(argv)
    args.use_kv_cache = not args.disable_kv_cache

    if not (0.0 <= args.obs_ratio <= 1.0):
        raise ValueError("obs_ratio must be in [0, 1].")
    if not (0.0 < args.topk_ratio <= 1.0):
        raise ValueError("topk_ratio must be in (0, 1].")
    if int(args.kv_cache_block_size) <= 0:
        raise ValueError("kv_cache_block_size must be >= 1.")
    if int(args.approx_score_block_size) <= 0:
        raise ValueError("approx_score_block_size must be >= 1.")
    if int(args.chunk_score_size) < 0:
        raise ValueError("chunk_score_size must be >= 0.")
    if not (0.0 <= float(args.chunk_score_alpha) <= 1.0):
        raise ValueError("chunk_score_alpha must be in [0, 1].")
    if not (0.0 < float(args.chunk_score_topk_ratio) <= 1.0):
        raise ValueError("chunk_score_topk_ratio must be in (0, 1].")
    if (
        str(args.online_score_mode).strip().lower() == "sliding_window"
        and int(args.chunk_score_size) <= 0
    ):
        raise ValueError(
            "chunk_score_size must be > 0 when online_score_mode=sliding_window."
        )
    if not (0.0 <= float(args.online_score_beta) <= 1.0):
        raise ValueError("online_score_beta must be in [0, 1].")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    use_teacher = bool(args.use_teacher)
    if use_teacher:
        args.use_kv_cache = False
        teacher_path = args.teacher_model_path or args.student_ckpt
        teacher, graph, noise, cfg, ckpt_path = load_teacher_checkpoint(teacher_path, device)
        model_for_logits = teacher
    else:
        teacher = None
        student, graph, noise, cfg, ckpt_path = load_student_checkpoint(args.student_ckpt, device)
        model_for_logits = student

    proto_cfg = getattr(cfg.data, "proto", None)
    if proto_cfg is None:
        raise ValueError("config.data.proto is required.")

    pad_token, eos_token = resolve_special_tokens(cfg)
    missing_placeholder_token = resolve_missing_placeholder_token(cfg)
    seq_len = int(cfg.model.length)
    mask_token = graph.dim - 1
    if args.dataset_path is None:
        dataset_path = pathlib.Path(proto_cfg.base_dir).expanduser().resolve() / str(
            proto_cfg.valid_file
        )
    else:
        dataset_path = pathlib.Path(args.dataset_path).expanduser().resolve()

    pin_memory = (device.type == "cuda") and (not args.disable_pin_memory)
    loader = dataloader_from_proto(
        str(dataset_path),
        seq_len,
        int(args.batch_size),
        pad_token,
        eos_token,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        missing_placeholder_token=missing_placeholder_token,
    )
    outlier_mask = load_outlier_mask(args.outlier_idx_path, len(loader.dataset))
    report_ratios = parse_ratio_list(args.report_progress_ratios)

    t_scalar = 1.0 - (
        (1.0 - SAMPLING_EPS_DEFAULT) / float(DEFAULT_FIRST_STEP_STEPS)
    )

    print(f"student_ckpt={ckpt_path}", flush=True)
    print(f"use_teacher={use_teacher}", flush=True)
    print(f"dataset_path={dataset_path}", flush=True)
    print(
        f"seq_len={seq_len} valid_token_vocab={cfg.tokens} absorb_mask_id={mask_token}",
        flush=True,
    )
    print(
        f"obs_ratio={args.obs_ratio} topk_ratio={args.topk_ratio} fixed_first_step_time={t_scalar:.8f}",
        flush=True,
    )
    print(f"kv_cache_block_size={int(args.kv_cache_block_size)}", flush=True)
    print(f"approx_score_block_size={int(args.approx_score_block_size)}", flush=True)
    print(
        f"chunk_score_size={int(args.chunk_score_size)} "
        f"chunk_score_alpha={float(args.chunk_score_alpha):.4f} "
        f"chunk_score_topk_ratio={float(args.chunk_score_topk_ratio):.4f} "
        f"online_score_mode={args.online_score_mode} "
        f"online_score_beta={float(args.online_score_beta):.4f}",
        flush=True,
    )
    print(f"outlier_idx_path={args.outlier_idx_path}", flush=True)

    traj_counter = 0
    trajectory_rows = []
    detection_start_time = time.perf_counter()
    mask_row_template = torch.full(
        (seq_len,), int(mask_token), dtype=torch.long, device=device
    )
    online_score_mode = str(args.online_score_mode).strip().lower()
    use_online_accumulators = online_score_mode == "sliding_window"

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            tokens_batch = batch["input_ids"].to(device, non_blocking=True)
            valid_mask_batch = batch.get("valid_mask")
            if valid_mask_batch is not None:
                valid_mask_batch = valid_mask_batch.to(device, non_blocking=True)

            remaining = (
                None
                if args.max_sequences is None
                else max(0, int(args.max_sequences) - traj_counter)
            )
            effective_batch = (
                tokens_batch.size(0)
                if remaining is None
                else min(tokens_batch.size(0), remaining)
            )
            if effective_batch <= 0:
                break

            if args.use_kv_cache:
                batch_tokens = tokens_batch[:effective_batch]
                if valid_mask_batch is not None:
                    valid_lens = valid_mask_batch[:effective_batch].sum(dim=1).long()
                else:
                    valid_lens = torch.full(
                        (effective_batch,), seq_len, device=device, dtype=torch.long
                    )
                obs_lens = torch.round(
                    valid_lens.float() * float(args.obs_ratio)
                ).long()
                obs_lens = torch.clamp(obs_lens, min=1)
                obs_lens = torch.minimum(obs_lens, valid_lens - 1)

                sigma = torch.full((effective_batch,), float(t_scalar), device=device)
                sort_order = torch.argsort(valid_lens, descending=True)
                batch_tokens_sorted = batch_tokens.index_select(0, sort_order)
                valid_lens_sorted = valid_lens.index_select(0, sort_order)
                obs_lens_sorted = obs_lens.index_select(0, sort_order)
                sigma_sorted = sigma.index_select(0, sort_order)
                approx_block = max(1, int(args.approx_score_block_size))
                score_matrix = None
                if not use_online_accumulators:
                    score_matrix = torch.empty(
                        (effective_batch, max(1, seq_len)),
                        dtype=torch.float32,
                        device=device,
                    )
                    score_matrix.fill_(float("nan"))
                accumulators = None
                if use_online_accumulators:
                    accumulators = [
                        OnlineTrajectoryAccumulator(
                            obs_len=int(obs_lens[i].item()),
                            valid_len=int(valid_lens[i].item()),
                            topk_ratio=float(args.topk_ratio),
                            report_ratios=report_ratios,
                            approx_block=int(approx_block),
                            chunk_score_size=int(args.chunk_score_size),
                            chunk_score_alpha=float(args.chunk_score_alpha),
                            chunk_score_topk_ratio=float(args.chunk_score_topk_ratio),
                            online_score_mode=online_score_mode,
                            online_score_beta=float(args.online_score_beta),
                        )
                        for i in range(effective_batch)
                    ]
                kv_cache = None
                mask_buf = torch.empty(effective_batch, dtype=torch.long, device=device)
                max_valid_len = (
                    int(valid_lens.max().item()) if effective_batch > 0 else 0
                )
                active_count = effective_batch
                sort_order_cpu = sort_order.detach().cpu().numpy()
                valid_lens_sorted_cpu = valid_lens_sorted.detach().cpu().tolist()
                obs_lens_sorted_cpu = obs_lens_sorted.detach().cpu().tolist()
                score_rows_by_pos = None
                if use_online_accumulators:
                    score_rows_by_pos = build_score_row_index_lists(
                        sort_order_cpu,
                        obs_lens_sorted_cpu,
                        valid_lens_sorted_cpu,
                        max_valid_len,
                    )

                min_obs_len = (
                    int(obs_lens_sorted.min().item()) if effective_batch > 0 else 0
                )
                start_idx = 0

                if min_obs_len > 0:
                    prefill_block = max(1, int(args.kv_cache_block_size))
                    for prefill_start in range(0, min_obs_len, prefill_block):
                        prefill_end = min(min_obs_len, prefill_start + prefill_block)
                        prefill_inputs = batch_tokens_sorted[
                            :, prefill_start:prefill_end
                        ]
                        _, kv_cache = student.forward_incremental_block(
                            prefill_inputs,
                            sigma_sorted,
                            start_position=prefill_start,
                            kv_cache=kv_cache,
                            update_cache=True,
                            compute_logits=False,
                        )
                    start_idx = min_obs_len

                if approx_block > 1:
                    prev_active = active_count
                    for block_start in range(start_idx, max_valid_len, approx_block):
                        while (
                            active_count > 0
                            and valid_lens_sorted_cpu[active_count - 1] <= block_start
                        ):
                            active_count -= 1
                        if active_count <= 0:
                            break
                        if active_count != prev_active:
                            kv_cache = lazy_trim_kv_cache(kv_cache, active_count, prev_active)
                            prev_active = active_count

                        block_end = min(max_valid_len, block_start + approx_block)
                        active_tokens = batch_tokens_sorted[:active_count]
                        active_sigma = sigma_sorted[:active_count]
                        active_valid_lens = valid_lens_sorted[:active_count]
                        active_obs_lens = obs_lens_sorted[:active_count]
                        active_rows = (
                            sort_order[:active_count] if score_matrix is not None else None
                        )
                        block_len = block_end - block_start

                        mask_block = torch.full(
                            (active_count, block_len),
                            int(mask_token),
                            dtype=torch.long,
                            device=device,
                        )
                        update_block = active_tokens[:, block_start:block_end].clone()
                        block_positions = torch.arange(
                            block_start, block_end, device=device
                        )[None, :]
                        invalid_mask = block_positions >= active_valid_lens[:, None]
                        update_block[invalid_mask] = int(pad_token)

                        block_logits, _ = student.forward_incremental_block(
                            mask_block,
                            active_sigma,
                            start_position=block_start,
                            kv_cache=kv_cache,
                            update_cache=False,
                            compute_logits=True,
                        )
                        block_logits = block_logits[..., : graph.dim - 1]
                        _, kv_cache = student.forward_incremental_block(
                            update_block,
                            active_sigma,
                            start_position=block_start,
                            kv_cache=kv_cache,
                            update_cache=True,
                            compute_logits=False,
                        )

                        for offset, local_idx in enumerate(
                            range(block_start, block_end)
                        ):
                            score_mask = (active_valid_lens > local_idx) & (
                                active_obs_lens <= local_idx
                            )
                            if not bool(score_mask.any()):
                                continue
                            active_logits = block_logits[score_mask, offset]
                            active_true = active_tokens[score_mask, local_idx]
                            true_logits = active_logits.gather(
                                1, active_true.unsqueeze(1)
                            ).squeeze(1)
                            ranks = (active_logits > true_logits.unsqueeze(1)).sum(
                                dim=1
                            ).long() + 1
                            logranks = ranks.float().clamp_min(1).log()
                            score_rows = None
                            if score_matrix is not None:
                                score_rows = active_rows[score_mask]
                                score_matrix[score_rows, local_idx] = logranks
                            if use_online_accumulators:
                                update_online_accumulators(
                                    accumulators,
                                    score_rows_by_pos[local_idx],
                                    logranks.detach().cpu().numpy(),
                                )

                else:
                    prev_active = active_count
                    for local_idx in range(start_idx, max_valid_len):
                        while (
                            active_count > 0
                            and valid_lens_sorted_cpu[active_count - 1] <= local_idx
                        ):
                            active_count -= 1
                        if active_count <= 0:
                            break
                        if active_count != prev_active:
                            kv_cache = lazy_trim_kv_cache(kv_cache, active_count, prev_active)
                            prev_active = active_count

                        active_tokens = batch_tokens_sorted[:active_count]
                        active_sigma = sigma_sorted[:active_count]
                        active_obs_lens = obs_lens_sorted[:active_count]
                        active_rows = (
                            sort_order[:active_count] if score_matrix is not None else None
                        )

                        score_mask = active_obs_lens <= local_idx
                        has_score = bool(score_mask.any())

                        update_inputs = active_tokens[:, local_idx]

                        if has_score:
                            mask_inputs = mask_buf[:active_count].fill_(int(mask_token))
                            logits, _ = student.forward_incremental(
                                mask_inputs,
                                active_sigma,
                                position=local_idx,
                                kv_cache=kv_cache,
                                update_cache=False,
                                compute_logits=True,
                            )
                            logits = logits[..., : graph.dim - 1].squeeze(1)
                            active_logits = logits[score_mask]
                            active_true = active_tokens[score_mask, local_idx]
                            true_logits = active_logits.gather(
                                1, active_true.unsqueeze(1)
                            ).squeeze(1)
                            ranks = (active_logits > true_logits.unsqueeze(1)).sum(
                                dim=1
                            ).long() + 1
                            logranks = ranks.float().clamp_min(1).log()
                            score_rows = None
                            if score_matrix is not None:
                                score_rows = active_rows[score_mask]
                                score_matrix[score_rows, local_idx] = logranks
                            if use_online_accumulators:
                                update_online_accumulators(
                                    accumulators,
                                    score_rows_by_pos[local_idx],
                                    logranks.detach().cpu().numpy(),
                                )

                        _, kv_cache = student.forward_incremental(
                            update_inputs,
                            active_sigma,
                            position=local_idx,
                            kv_cache=kv_cache,
                            update_cache=True,
                            compute_logits=False,
                        )

                valid_lens_cpu = valid_lens[:effective_batch].detach().cpu().tolist()
                obs_lens_cpu = obs_lens[:effective_batch].detach().cpu().tolist()
                score_matrix_cpu = None if use_online_accumulators else score_matrix.cpu()
                for b in range(effective_batch):
                    traj_id = traj_counter + b
                    label = (
                        int(outlier_mask[traj_id]) if outlier_mask is not None else -1
                    )
                    valid_len = int(valid_lens_cpu[b])
                    obs_len_b = int(obs_lens_cpu[b])
                    if valid_len < 2:
                        trajectory_rows.append(
                            {
                                "traj_id": traj_id,
                                "label": label,
                                "valid_len": valid_len,
                                "num_scored_points": 0,
                                "initial_online_score": float("nan"),
                                "final_online_score": float("nan"),
                                "max_online_score": float("nan"),
                                "chunk_final_online_score": float("nan"),
                                "chunk_max_online_score": float("nan"),
                                "num_chunks_scored": 0,
                                "num_refresh": 0,
                                **{
                                    f"score_at_ratio_{ratio:.2f}": float("nan")
                                    for ratio in report_ratios
                                },
                                **{
                                    f"chunk_score_at_ratio_{ratio:.2f}": float("nan")
                                    for ratio in report_ratios
                                },
                            }
                        )
                        continue

                    if use_online_accumulators:
                        row = accumulators[b].build_row()
                    else:
                        scored_values = score_matrix_cpu[b, obs_len_b:valid_len].numpy()
                        scored_values = scored_values[~np.isnan(scored_values)]
                        row = summarize_trajectory_scores(
                            scored_values,
                            obs_len=obs_len_b,
                            valid_len=valid_len,
                            topk_ratio=float(args.topk_ratio),
                            report_ratios=report_ratios,
                            approx_block=int(approx_block),
                            chunk_score_size=int(args.chunk_score_size),
                            chunk_score_alpha=float(args.chunk_score_alpha),
                            chunk_score_topk_ratio=float(args.chunk_score_topk_ratio),
                            online_score_mode=online_score_mode,
                            online_score_beta=float(args.online_score_beta),
                        )
                    row.update(
                        {
                            "traj_id": traj_id,
                            "label": label,
                            "valid_len": valid_len,
                        }
                    )
                    trajectory_rows.append(row)
            else:
                batch_tokens = tokens_batch[:effective_batch]
                if valid_mask_batch is not None:
                    valid_lens = valid_mask_batch[:effective_batch].sum(dim=1).long()
                else:
                    valid_lens = torch.full(
                        (effective_batch,), seq_len, device=device, dtype=torch.long
                    )
                obs_lens = torch.round(
                    valid_lens.float() * float(args.obs_ratio)
                ).long()
                obs_lens = torch.clamp(obs_lens, min=1)
                obs_lens = torch.minimum(obs_lens, valid_lens - 1)

                sort_order = torch.argsort(valid_lens, descending=True)
                batch_tokens_sorted = batch_tokens.index_select(0, sort_order)
                valid_lens_sorted = valid_lens.index_select(0, sort_order)
                obs_lens_sorted = obs_lens.index_select(0, sort_order)
                score_matrix = None
                if not use_online_accumulators:
                    score_matrix = torch.empty(
                        (effective_batch, max(1, seq_len)),
                        dtype=torch.float32,
                        device=device,
                    )
                    score_matrix.fill_(float("nan"))
                accumulators = None
                if use_online_accumulators:
                    accumulators = [
                        OnlineTrajectoryAccumulator(
                            obs_len=int(obs_lens[i].item()),
                            valid_len=int(valid_lens[i].item()),
                            topk_ratio=float(args.topk_ratio),
                            report_ratios=report_ratios,
                            approx_block=1,
                            chunk_score_size=int(args.chunk_score_size),
                            chunk_score_alpha=float(args.chunk_score_alpha),
                            chunk_score_topk_ratio=float(args.chunk_score_topk_ratio),
                            online_score_mode=online_score_mode,
                            online_score_beta=float(args.online_score_beta),
                        )
                        for i in range(effective_batch)
                    ]
                max_valid_len = int(valid_lens.max().item()) if effective_batch > 0 else 0
                active_count = effective_batch
                sort_order_cpu = sort_order.detach().cpu().numpy()
                valid_lens_sorted_cpu = valid_lens_sorted.detach().cpu().tolist()
                obs_lens_sorted_cpu = obs_lens_sorted.detach().cpu().tolist()
                score_rows_by_pos = None
                if use_online_accumulators:
                    score_rows_by_pos = build_score_row_index_lists(
                        sort_order_cpu,
                        obs_lens_sorted_cpu,
                        valid_lens_sorted_cpu,
                        max_valid_len,
                    )

                for local_idx in range(0, max_valid_len):
                    while (
                        active_count > 0
                        and valid_lens_sorted_cpu[active_count - 1] <= local_idx
                    ):
                        active_count -= 1
                    if active_count <= 0:
                        break

                    active_tokens = batch_tokens_sorted[:active_count]
                    active_obs_lens = obs_lens_sorted[:active_count]
                    active_rows = (
                        sort_order[:active_count] if score_matrix is not None else None
                    )
                    score_mask = active_obs_lens <= local_idx
                    if not bool(score_mask.any()):
                        continue

                    score_tokens = active_tokens[score_mask]
                    score_rows = active_rows[score_mask] if score_matrix is not None else None
                    x_step = mask_row_template.repeat(score_tokens.size(0), 1)
                    if local_idx > 0:
                        x_step[:, :local_idx] = score_tokens[:, :local_idx]
                    logits = batched_teacher_logits(
                        model_for_logits,
                        graph,
                        noise,
                        x_step,
                        float(t_scalar),
                        int(args.query_batch_size),
                    ) if use_teacher else batched_student_logits(
                        model_for_logits,
                        graph,
                        noise,
                        x_step,
                        float(t_scalar),
                        int(args.query_batch_size),
                    )
                    active_logits = logits[:, local_idx]
                    active_true = score_tokens[:, local_idx]
                    true_logits = active_logits.gather(
                        1, active_true.unsqueeze(1)
                    ).squeeze(1)
                    ranks = (active_logits > true_logits.unsqueeze(1)).sum(dim=1).long() + 1
                    logranks = ranks.float().clamp_min(1).log()
                    if score_matrix is not None:
                        score_matrix[score_rows, local_idx] = logranks
                    if use_online_accumulators:
                        update_online_accumulators(
                            accumulators,
                            score_rows_by_pos[local_idx],
                            logranks.detach().cpu().numpy(),
                        )

                valid_lens_cpu = valid_lens[:effective_batch].detach().cpu().tolist()
                obs_lens_cpu = obs_lens[:effective_batch].detach().cpu().tolist()
                score_matrix_cpu = None if use_online_accumulators else score_matrix.cpu()
                for b in range(effective_batch):
                    traj_id = traj_counter + b
                    label = (
                        int(outlier_mask[traj_id]) if outlier_mask is not None else -1
                    )
                    valid_len = int(valid_lens_cpu[b])
                    obs_len_b = int(obs_lens_cpu[b])
                    if valid_len < 2:
                        trajectory_rows.append(
                            {
                                "traj_id": traj_id,
                                "label": label,
                                "valid_len": valid_len,
                                "num_scored_points": 0,
                                "initial_online_score": float("nan"),
                                "final_online_score": float("nan"),
                                "max_online_score": float("nan"),
                                "chunk_final_online_score": float("nan"),
                                "chunk_max_online_score": float("nan"),
                                "num_chunks_scored": 0,
                                "num_refresh": 0,
                                **{
                                    f"score_at_ratio_{ratio:.2f}": float("nan")
                                    for ratio in report_ratios
                                },
                                **{
                                    f"chunk_score_at_ratio_{ratio:.2f}": float("nan")
                                    for ratio in report_ratios
                                },
                            }
                        )
                        continue

                    if use_online_accumulators:
                        row = accumulators[b].build_row()
                    else:
                        scored_values = score_matrix_cpu[b, obs_len_b:valid_len].numpy()
                        scored_values = scored_values[~np.isnan(scored_values)]
                        row = summarize_trajectory_scores(
                            scored_values,
                            obs_len=obs_len_b,
                            valid_len=valid_len,
                            topk_ratio=float(args.topk_ratio),
                            report_ratios=report_ratios,
                            approx_block=1,
                            chunk_score_size=int(args.chunk_score_size),
                            chunk_score_alpha=float(args.chunk_score_alpha),
                            chunk_score_topk_ratio=float(args.chunk_score_topk_ratio),
                            online_score_mode=online_score_mode,
                            online_score_beta=float(args.online_score_beta),
                        )
                    row.update(
                        {
                            "traj_id": traj_id,
                            "label": label,
                            "valid_len": valid_len,
                            "num_refresh": int(row["num_scored_points"]),
                        }
                    )
                    trajectory_rows.append(row)

            traj_counter += effective_batch
            if traj_counter % 200 == 0:
                print(f"processed_trajectories={traj_counter}", flush=True)
            if args.max_sequences is not None and traj_counter >= int(
                args.max_sequences
            ):
                break

    print(f"processed_trajectories_total={traj_counter}", flush=True)
    valid_rows = [
        row for row in trajectory_rows if not math.isnan(row["final_online_score"])
    ]
    print(f"valid_final_scores={len(valid_rows)}", flush=True)
    elapsed = time.perf_counter() - detection_start_time
    print(f"detection_elapsed_seconds={elapsed:.6f}", flush=True)
    if traj_counter > 0:
        print(
            f"detection_avg_seconds_per_trajectory={elapsed / traj_counter:.6f}",
            flush=True,
        )
    valid_lens_list = []
    for row in trajectory_rows:
        vlen = row.get("valid_len")
        if vlen is not None and not math.isnan(row.get("final_online_score", float("nan"))):
            valid_lens_list.append(int(vlen))
    if valid_lens_list:
        vlens_arr = np.array(valid_lens_list, dtype=np.float64)
        avg_full_len = float(np.mean(vlens_arr))
        print(f"per_ratio_elapsed_estimate_from_total_{elapsed:.3f}s:", flush=True)
        for ratio in report_ratios:
            est_steps = np.maximum(1.0, np.ceil(vlens_arr * float(ratio)))
            avg_steps = float(np.mean(est_steps))
            per_ratio_elapsed = elapsed * avg_steps / max(avg_full_len, 1.0)
            print(
                f"estimated_elapsed@observed_ratio={ratio:.2f}: {per_ratio_elapsed:.3f}s",
                flush=True,
            )

    labels = None
    final_scores = None
    if outlier_mask is not None and valid_rows:
        labels = np.asarray([row["label"] for row in valid_rows], dtype=np.int64)
        final_scores = np.asarray(
            [row["final_online_score"] for row in valid_rows], dtype=np.float64
        )

    if (
        labels is not None
        and average_precision_score is not None
        and precision_recall_curve is not None
        and auc is not None
        and len(np.unique(labels)) > 1
    ):
        ap = float(average_precision_score(labels, final_scores))
        precision, recall, _ = precision_recall_curve(labels, final_scores)
        pr_auc = float(auc(recall, precision))
        print(f"trajectory_final_score_avg_precision={ap:.6f}", flush=True)
        print(f"trajectory_final_score_pr_auc={pr_auc:.6f}", flush=True)
        if int(args.chunk_score_size) > 0 and str(args.online_score_mode) != "sliding_window":
            chunk_scores_np = np.asarray(
                [row["chunk_final_online_score"] for row in valid_rows],
                dtype=np.float64,
            )
            keep = ~np.isnan(chunk_scores_np)
            if keep.sum() > 0:
                chunk_ap = float(average_precision_score(labels[keep], chunk_scores_np[keep]))
                chunk_precision, chunk_recall, _ = precision_recall_curve(
                    labels[keep], chunk_scores_np[keep]
                )
                chunk_pr_auc = float(auc(chunk_recall, chunk_precision))
                print(f"trajectory_chunk_final_score_avg_precision={chunk_ap:.6f}", flush=True)
                print(f"trajectory_chunk_final_score_pr_auc={chunk_pr_auc:.6f}", flush=True)
        for ratio in report_ratios:
            ratio_scores = np.asarray(
                [row[f"score_at_ratio_{ratio:.2f}"] for row in valid_rows],
                dtype=np.float64,
            )
            keep = ~np.isnan(ratio_scores)
            if keep.sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(
                labels[keep], ratio_scores[keep]
            )
            ratio_pr_auc = float(auc(recall, precision))
            print(f"PR_AUC@observed_ratio={ratio:.2f}: {ratio_pr_auc:.6f}", flush=True)
            if int(args.chunk_score_size) > 0 and str(args.online_score_mode) != "sliding_window":
                chunk_ratio_scores = np.asarray(
                    [row[f"chunk_score_at_ratio_{ratio:.2f}"] for row in valid_rows],
                    dtype=np.float64,
                )
                keep_chunk = ~np.isnan(chunk_ratio_scores)
                if keep_chunk.sum() > 0:
                    chunk_precision, chunk_recall, _ = precision_recall_curve(
                        labels[keep_chunk], chunk_ratio_scores[keep_chunk]
                    )
                    chunk_ratio_pr_auc = float(auc(chunk_recall, chunk_precision))
                    print(
                        f"Chunk_PR_AUC@observed_ratio={ratio:.2f}: {chunk_ratio_pr_auc:.6f}",
                        flush=True,
                    )
            if abs(ratio - 1.0) < 1e-8:
                print(f"Full teacher-forcing PR_AUC: {ratio_pr_auc:.6f}", flush=True)
    else:
        print(
            "Skipped anomaly metrics: labels are missing or single-class.", flush=True
        )


if __name__ == "__main__":
    main()

# python run_causal_student_online_eval.py \
#     --student_ckpt /data3/fr/ImputeAD/exp_local/proto/2026.01.29/003254/autoregressive_distill_runs/20260408_200step/checkpoints/student_step_200.pth \
#     --device cuda:1 \
#     --batch_size 64 \
#     --query_batch_size 512 \
#     --max_sequences 128 \
#     --obs_ratio 0 \
#     --topk_ratio 0.05 \
#     --dataset_path /data3/fr/Datasets/dataset_porto/outliers_data_unified_a0.1_d3_notime_miss_block_0.00.npy \
#     --outlier_idx_path /data3/fr/Datasets/dataset_porto/outliers_idx_unified_a0.1_d3_notime_miss_block_0.00.npy
