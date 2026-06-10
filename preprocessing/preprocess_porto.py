#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .preprocess_utils import (
        PORTO_BOUNDARY,
        grid_mapping,
        in_boundary,
        point_to_token,
        porto_time_vector,
        save_object_array,
    )
except ImportError:
    from preprocess_utils import (
        PORTO_BOUNDARY,
        grid_mapping,
        in_boundary,
        point_to_token,
        porto_time_vector,
        save_object_array,
    )


def parse_datetime(text: str):
    if not text:
        return None
    return dt.datetime.fromisoformat(str(text).replace(" ", "T"))


def load_porto(
    raw_csv: Path,
    boundary: dict,
    lat_size: float,
    lng_size: float,
    lng_grid_num: int,
    shortest: int,
    sampling_interval: int,
    start_dt=None,
    end_dt=None,
) -> list[dict]:
    df = pd.read_csv(raw_csv, usecols=["POLYLINE", "TIMESTAMP"])
    trajectories = []

    for row in df.itertuples():
        polyline = json.loads(row.POLYLINE)
        if len(polyline) < shortest:
            continue

        start_ts = int(row.TIMESTAMP)
        start_time = dt.datetime.fromtimestamp(start_ts)
        if start_dt is not None and start_time < start_dt:
            continue
        if end_dt is not None and start_time >= end_dt:
            continue

        grid_seq = []
        coords = []
        valid = True
        for lng, lat in polyline:
            lat = float(lat)
            lng = float(lng)
            if not in_boundary(lat, lng, boundary):
                valid = False
                break
            token = point_to_token(lat, lng, boundary, lat_size, lng_size, lng_grid_num)
            grid_seq.append(token)
            coords.append((lat, lng))

        if not valid:
            continue

        times = [start_ts + idx * int(sampling_interval) for idx in range(len(grid_seq))]
        trajectories.append(
            {
                "start_ts": start_ts,
                "grid_seq": grid_seq,
                "coords": coords,
                "times": times,
            }
        )

    return trajectories


def apply_random_drop_and_slice(
    trajectories: list[dict],
    shortest: int,
    longest: int,
    drop_ratio: float,
    seed: int,
) -> list[dict]:
    rng = np.random.RandomState(seed)
    processed = []
    for traj in trajectories:
        packed = list(zip(traj["grid_seq"], traj["coords"], traj["times"]))
        if len(packed) < shortest:
            continue

        drop_num = int(len(packed) * float(drop_ratio))
        if drop_num > 0:
            drop_idx = set(rng.choice(len(packed), drop_num, replace=False))
            packed = [item for idx, item in enumerate(packed) if idx not in drop_idx]

        if len(packed) < shortest:
            continue

        if len(packed) > longest:
            length = int(rng.randint(shortest, longest))
            segments = [packed[:length]]
        else:
            segments = [packed]

        for segment in segments:
            grid_seq = [cell for cell, _, _ in segment]
            coords = [coord for _, coord, _ in segment]
            times = [timestamp for _, _, timestamp in segment]
            processed.append(
                {
                    "start_ts": times[0],
                    "grid_seq": grid_seq,
                    "coords": coords,
                    "times": times,
                }
            )

    return processed


def random_train_test_split(data: list, test_size: float = 0.1, seed: int = 1234):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    split = int(len(data) * (1.0 - float(test_size)))
    train_idx, test_idx = indices[:split], indices[split:]
    return [data[idx] for idx in train_idx], [data[idx] for idx in test_idx]


def sd_pair_split(
    trajectories: list[dict],
    min_sd_traj_num: int = 25,
    train_ratio: float = 0.9,
    val_ratio: float = 0.0,
    seed: int = 1234,
):
    rng = np.random.RandomState(seed)
    sd_groups = defaultdict(list)
    for traj in trajectories:
        if not traj["grid_seq"]:
            continue
        sd = (traj["grid_seq"][0], traj["grid_seq"][-1])
        sd_groups[sd].append(traj)

    train_trajs, val_trajs, test_trajs = [], [], []
    kept_sd = 0
    for _, group in sd_groups.items():
        if len(group) < min_sd_traj_num:
            continue
        kept_sd += 1
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        train_trajs.extend(group[:n_train])
        val_trajs.extend(group[n_train : n_train + n_val])
        test_trajs.extend(group[n_train + n_val : n_train + n_val + n_test])

    return train_trajs, val_trajs, test_trajs, kept_sd


def to_proto_sequences(trajectories: list[dict]) -> list[list]:
    converted = []
    for traj in trajectories:
        seq = []
        for cell_id, timestamp in zip(traj["grid_seq"], traj["times"]):
            seq.append([int(cell_id), porto_time_vector(int(timestamp))])
        converted.append(seq)
    return converted


def preprocess_porto(
    input_csv: Path,
    output_dir: Path,
    grid_size_km: float,
    sampling_interval: int,
    shortest: int,
    longest: int,
    drop_ratio: float,
    start_time: str,
    end_time: str,
    min_sd_traj_num: int,
    test_ratio: float,
    seed: int,
) -> dict:
    lat_size, lng_size, lat_rows, lng_cols = grid_mapping(PORTO_BOUNDARY, grid_size_km)
    raw_trajs = load_porto(
        raw_csv=input_csv,
        boundary=PORTO_BOUNDARY,
        lat_size=lat_size,
        lng_size=lng_size,
        lng_grid_num=lng_cols,
        shortest=1,
        sampling_interval=sampling_interval,
        start_dt=parse_datetime(start_time),
        end_dt=parse_datetime(end_time),
    )
    trajectories = apply_random_drop_and_slice(
        raw_trajs,
        shortest=shortest,
        longest=longest,
        drop_ratio=drop_ratio,
        seed=seed,
    )

    if min_sd_traj_num > 0:
        train_data, _, test_data, kept_sd = sd_pair_split(
            trajectories,
            min_sd_traj_num=min_sd_traj_num,
            train_ratio=1.0 - float(test_ratio),
            val_ratio=0.0,
            seed=seed,
        )
        split_strategy = "sd_pair"
    else:
        train_data, test_data = random_train_test_split(
            trajectories,
            test_size=test_ratio,
            seed=seed,
        )
        kept_sd = 0
        split_strategy = "random"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_object_array(output_dir / "train_data_init.npy", to_proto_sequences(train_data))
    save_object_array(output_dir / "test_data_init.npy", to_proto_sequences(test_data))
    metadata = {
        "dataset": "porto",
        "input_csv": str(input_csv),
        "grid_size_km": float(grid_size_km),
        "grid_shape": [int(lat_rows), int(lng_cols)],
        "num_tokens_without_specials": int(lat_rows * lng_cols),
        "sampling_interval": int(sampling_interval),
        "shortest": int(shortest),
        "longest": int(longest),
        "drop_ratio": float(drop_ratio),
        "start_time": str(start_time),
        "end_time": str(end_time),
        "num_raw_trajectories": int(len(raw_trajs)),
        "num_trajectories": int(len(trajectories)),
        "num_points": int(sum(len(traj["grid_seq"]) for traj in trajectories)),
        "split_strategy": split_strategy,
        "test_ratio": float(test_ratio),
        "min_sd_traj_num": int(min_sd_traj_num),
        "kept_sd": int(kept_sd),
        "train_size": int(len(train_data)),
        "test_size": int(len(test_data)),
        "seed": int(seed),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Porto trajectories.")
    parser.add_argument("--input_csv", "--raw_csv", dest="input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", "--out_dir", dest="output_dir", type=Path, default=Path("data/porto"))
    parser.add_argument("--grid_size_km", type=float, default=0.1)
    parser.add_argument("--sampling_interval", type=int, default=15)
    parser.add_argument("--shortest", type=int, default=20)
    parser.add_argument("--longest", type=int, default=50)
    parser.add_argument("--drop_ratio", type=float, default=0.3)
    parser.add_argument("--start_time", "--start_date", dest="start_time", type=str, default="")
    parser.add_argument("--end_time", "--end_date", dest="end_time", type=str, default="")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--min_sd_traj_num", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = preprocess_porto(
        input_csv=args.input_csv.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        grid_size_km=float(args.grid_size_km),
        sampling_interval=int(args.sampling_interval),
        shortest=int(args.shortest),
        longest=int(args.longest),
        drop_ratio=float(args.drop_ratio),
        start_time=str(args.start_time),
        end_time=str(args.end_time),
        min_sd_traj_num=int(args.min_sd_traj_num),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
