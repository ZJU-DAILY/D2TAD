#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def parse_ignore_tokens(text: str) -> set[int]:
    if not text:
        return set()
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def iter_token_sequences(array: np.ndarray) -> Iterable[list[int]]:
    for seq in array.tolist():
        tokens: list[int] = []
        for item in seq:
            if isinstance(item, (list, tuple)):
                token = item[0]
            else:
                token = item
            if token is None:
                continue
            tokens.append(int(token))
        if len(tokens) >= 2:
            yield tokens


def build_transition_prob(
    dataset_path: Path,
    num_tokens: int,
    ignore_tokens: set[int],
    smoothing: float,
) -> torch.Tensor:
    array = np.load(dataset_path, allow_pickle=True)
    counts = torch.zeros((num_tokens, num_tokens), dtype=torch.float32)

    for tokens in iter_token_sequences(array):
        src = torch.tensor(tokens[:-1], dtype=torch.long)
        dst = torch.tensor(tokens[1:], dtype=torch.long)
        valid = (src >= 0) & (src < num_tokens) & (dst >= 0) & (dst < num_tokens)
        if ignore_tokens:
            for tok in ignore_tokens:
                valid &= (src != tok) & (dst != tok)
        if not torch.any(valid):
            continue
        src = src[valid]
        dst = dst[valid]
        ones = torch.ones_like(src, dtype=torch.float32)
        counts.index_put_((src, dst), ones, accumulate=True)

    if smoothing > 0:
        counts.add_(float(smoothing))
    row_sum = counts.sum(dim=-1, keepdim=True)
    zero_rows = row_sum.squeeze(-1) <= 0
    row_sum.clamp_min_(1e-12)
    prob = counts / row_sum
    if zero_rows.any():
        idx = torch.arange(num_tokens)
        prob[zero_rows] = 0.0
        prob[zero_rows, idx[zero_rows]] = 1.0
    return prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense empirical transition probabilities from trajectories.")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--num_tokens", type=int, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ignore_tokens", type=str, default="")
    parser.add_argument("--smoothing", type=float, default=1e-6)
    args = parser.parse_args()

    prob = build_transition_prob(
        dataset_path=args.dataset_path,
        num_tokens=int(args.num_tokens),
        ignore_tokens=parse_ignore_tokens(args.ignore_tokens),
        smoothing=float(args.smoothing),
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "transition_prob": prob,
            "dataset_path": str(args.dataset_path),
            "num_tokens": int(args.num_tokens),
            "ignore_tokens": sorted(parse_ignore_tokens(args.ignore_tokens)),
            "smoothing": float(args.smoothing),
        },
        args.output_path,
    )
    print(f"saved_transition_prob={args.output_path}")


if __name__ == "__main__":
    main()
