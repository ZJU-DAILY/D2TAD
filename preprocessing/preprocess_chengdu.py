#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .preprocess_utils import (
        CHENGDU_BOUNDARY,
        grid_mapping,
        in_boundary,
        point_to_token,
        save_object_array,
    )
except ImportError:
    from preprocess_utils import (
        CHENGDU_BOUNDARY,
        grid_mapping,
        in_boundary,
        point_to_token,
        save_object_array,
    )


def parse_timestamp(timestamp: str, timestamp_format: str):
    try:
        return dt.datetime.strptime(str(timestamp), timestamp_format)
    except (TypeError, ValueError):
        return None


def time_vector(value: dt.datetime) -> list[int]:
    return [
        value.hour,
        value.minute,
        value.second,
        value.year,
        value.month,
        value.day,
    ]


def cutting_trajs_with_meta(
    grid_seq: list,
    coords: list,
    times: list,
    time_vecs: list,
    longest: int,
    shortest: int,
    rng,
) -> list[dict]:
    cutted = []
    while len(grid_seq) > longest:
        length = int(rng.randint(shortest, longest))
        cutted.append(
            {
                "grid_seq": grid_seq[:length],
                "coords": coords[:length],
                "times": times[:length],
                "time_vecs": time_vecs[:length],
                "start_ts": times[0],
            }
        )
        grid_seq = grid_seq[length:]
        coords = coords[length:]
        times = times[length:]
        time_vecs = time_vecs[length:]
    return cutted


def append_or_cut(
    trajectories: list,
    grid_seq: list,
    coords: list,
    times: list,
    time_vecs: list,
    shortest: int,
    longest: int,
    rng,
) -> None:
    if shortest <= len(grid_seq) <= longest:
        trajectories.append(
            {
                "grid_seq": grid_seq,
                "coords": coords,
                "times": times,
                "time_vecs": time_vecs,
                "start_ts": times[0],
            }
        )
    elif len(grid_seq) > longest:
        trajectories.extend(
            cutting_trajs_with_meta(
                grid_seq,
                coords,
                times,
                time_vecs,
                longest,
                shortest,
                rng,
            )
        )


def preprocess_file(
    file_path: Path,
    boundary: dict,
    lat_size: float,
    lng_size: float,
    lng_grid_num: int,
    shortest: int,
    longest: int,
    max_gap: float,
    timestamp_format: str,
    state_value: int,
    rng,
    has_header: bool = False,
) -> list[dict]:
    df = pd.read_csv(file_path, header=0 if has_header else None)
    if df.empty:
        return []
    df = df.iloc[:, :5]
    df.columns = ["id", "lat", "lon", "state", "timestamp"]
    df = df.sort_values(by=["id", "timestamp"])
    df = df[df["state"] == state_value]
    if df.empty:
        return []

    trajectories = []
    grid_seq = []
    coords = []
    times = []
    time_vecs = []
    valid = True
    pre_id = None
    pre_dt = None

    def reset():
        return [], [], [], []

    for row in df.itertuples(index=False):
        try:
            lat = float(row.lat)
            lng = float(row.lon)
        except (TypeError, ValueError):
            lat = None
            lng = None

        cur_dt = parse_timestamp(row.timestamp, timestamp_format)
        if cur_dt is None:
            if valid and grid_seq:
                append_or_cut(
                    trajectories,
                    grid_seq,
                    coords,
                    times,
                    time_vecs,
                    shortest,
                    longest,
                    rng,
                )
            grid_seq, coords, times, time_vecs = reset()
            valid = True
            pre_id = None
            pre_dt = None
            continue

        if pre_id is None:
            if lat is not None and lng is not None and in_boundary(lat, lng, boundary):
                token = point_to_token(
                    lat,
                    lng,
                    boundary,
                    lat_size,
                    lng_size,
                    lng_grid_num,
                )
                grid_seq = [token]
                coords = [(lat, lng)]
                times = [int(cur_dt.timestamp())]
                time_vecs = [time_vector(cur_dt)]
                valid = True
            else:
                grid_seq, coords, times, time_vecs = reset()
                valid = False
            pre_id = row.id
            pre_dt = cur_dt
            continue

        gap = (cur_dt - pre_dt).total_seconds()
        same_traj = row.id == pre_id and gap <= max_gap and gap >= 0

        if not same_traj:
            if valid and grid_seq:
                append_or_cut(
                    trajectories,
                    grid_seq,
                    coords,
                    times,
                    time_vecs,
                    shortest,
                    longest,
                    rng,
                )

            grid_seq, coords, times, time_vecs = reset()
            if lat is not None and lng is not None and in_boundary(lat, lng, boundary):
                token = point_to_token(
                    lat,
                    lng,
                    boundary,
                    lat_size,
                    lng_size,
                    lng_grid_num,
                )
                grid_seq.append(token)
                coords.append((lat, lng))
                times.append(int(cur_dt.timestamp()))
                time_vecs.append(time_vector(cur_dt))
                valid = True
            else:
                valid = False
        else:
            if valid:
                if lat is not None and lng is not None and in_boundary(lat, lng, boundary):
                    token = point_to_token(
                        lat,
                        lng,
                        boundary,
                        lat_size,
                        lng_size,
                        lng_grid_num,
                    )
                    grid_seq.append(token)
                    coords.append((lat, lng))
                    times.append(int(cur_dt.timestamp()))
                    time_vecs.append(time_vector(cur_dt))
                else:
                    valid = False

        pre_id = row.id
        pre_dt = cur_dt

    if pre_id is not None and valid and grid_seq:
        append_or_cut(
            trajectories,
            grid_seq,
            coords,
            times,
            time_vecs,
            shortest,
            longest,
            rng,
        )

    return trajectories


def random_train_test_split(data: list, test_size: float = 0.1, seed: int = 42):
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
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    sd_groups = defaultdict(list)
    for traj in trajectories:
        if not traj or not traj.get("grid_seq"):
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
    return [
        [[int(cell_id), list(time_vec)] for cell_id, time_vec in zip(traj["grid_seq"], traj["time_vecs"])]
        for traj in trajectories
    ]


def preprocess_chengdu(
    input_dir: Path,
    output_dir: Path,
    grid_size_km: float,
    shortest: int,
    longest: int,
    max_gap: float,
    test_ratio: float,
    seed: int,
    min_sd_traj_num: int,
    state_value: int,
    max_files: int,
    merge_first: int,
    split_seed: int,
    has_header: bool,
    timestamp_format: str,
    lat_rows: Optional[int] = None,
    lng_cols: Optional[int] = None,
) -> dict:
    if (lat_rows is None) != (lng_cols is None):
        raise ValueError("lat_rows and lng_cols must be provided together")

    if lat_rows is None:
        lat_size, lng_size, lat_grid_num, lng_grid_num = grid_mapping(
            CHENGDU_BOUNDARY,
            grid_size_km,
        )
    else:
        lat_grid_num = int(lat_rows)
        lng_grid_num = int(lng_cols)
        lat_size = (CHENGDU_BOUNDARY["max_lat"] - CHENGDU_BOUNDARY["min_lat"]) / lat_grid_num
        lng_size = (CHENGDU_BOUNDARY["max_lng"] - CHENGDU_BOUNDARY["min_lng"]) / lng_grid_num

    files = sorted(path for path in input_dir.iterdir() if path.is_file())
    if max_files and max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files found under {input_dir}")

    rng = np.random.RandomState(seed)
    per_file_trajs = []
    for path in files:
        per_file_trajs.append(
            preprocess_file(
                file_path=path,
                boundary=CHENGDU_BOUNDARY,
                lat_size=lat_size,
                lng_size=lng_size,
                lng_grid_num=lng_grid_num,
                shortest=shortest,
                longest=longest,
                max_gap=max_gap,
                timestamp_format=timestamp_format,
                state_value=state_value,
                rng=rng,
                has_header=has_header,
            )
        )

    merge_count = int(merge_first) if int(merge_first) > 0 else len(per_file_trajs)
    trajectories = []
    for trajs in per_file_trajs[:merge_count]:
        trajectories.extend(trajs)

    if min_sd_traj_num > 0:
        train_data, _, test_data, kept_sd = sd_pair_split(
            trajectories,
            min_sd_traj_num=min_sd_traj_num,
            train_ratio=1.0 - float(test_ratio),
            val_ratio=0.0,
            seed=split_seed,
        )
        split_strategy = "sd_pair"
    else:
        train_data, test_data = random_train_test_split(
            trajectories,
            test_size=test_ratio,
            seed=split_seed,
        )
        kept_sd = 0
        split_strategy = "random"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_object_array(output_dir / "train_data_init.npy", to_proto_sequences(train_data))
    save_object_array(output_dir / "test_data_init.npy", to_proto_sequences(test_data))
    metadata = {
        "dataset": "chengdu",
        "input_dir": str(input_dir),
        "grid_size_km": float(grid_size_km),
        "grid_shape": [int(lat_grid_num), int(lng_grid_num)],
        "num_tokens_without_specials": int(lat_grid_num * lng_grid_num),
        "shortest": int(shortest),
        "longest": int(longest),
        "max_gap": float(max_gap),
        "test_ratio": float(test_ratio),
        "min_sd_traj_num": int(min_sd_traj_num),
        "state_value": int(state_value),
        "num_files": int(len(files)),
        "merge_first": int(merge_count),
        "num_trajectories": int(len(trajectories)),
        "num_points": int(sum(len(traj["grid_seq"]) for traj in trajectories)),
        "split_strategy": split_strategy,
        "kept_sd": int(kept_sd),
        "train_size": int(len(train_data)),
        "test_size": int(len(test_data)),
        "seed": int(seed),
        "split_seed": int(split_seed),
        "timestamp_format": str(timestamp_format),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Chengdu trajectories.")
    parser.add_argument("--input_dir", "--data_dir", dest="input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", "--out_dir", dest="output_dir", type=Path, default=Path("data/chengdu"))
    parser.add_argument("--grid_size_km", type=float, default=0.3)
    parser.add_argument("--lat_rows", type=int, default=None)
    parser.add_argument("--lng_cols", type=int, default=None)
    parser.add_argument("--shortest", type=int, default=30)
    parser.add_argument("--longest", type=int, default=100)
    parser.add_argument("--max_gap", "--max_gap_seconds", dest="max_gap", type=float, default=20)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min_sd_traj_num", type=int, default=25)
    parser.add_argument("--state_value", type=int, default=1)
    parser.add_argument("--max_files", type=int, default=7)
    parser.add_argument("--merge_first", type=int, default=7)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--has_header", action="store_true")
    parser.add_argument("--timestamp_format", type=str, default="%Y/%m/%d %H:%M:%S")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = preprocess_chengdu(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        grid_size_km=float(args.grid_size_km),
        shortest=int(args.shortest),
        longest=int(args.longest),
        max_gap=float(args.max_gap),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        min_sd_traj_num=int(args.min_sd_traj_num),
        state_value=int(args.state_value),
        max_files=int(args.max_files),
        merge_first=int(args.merge_first),
        split_seed=int(args.split_seed),
        has_header=bool(args.has_header),
        timestamp_format=str(args.timestamp_format),
        lat_rows=args.lat_rows,
        lng_cols=args.lng_cols,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
