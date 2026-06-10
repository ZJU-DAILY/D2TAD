#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np


DATASET_BOUNDARIES = {
    "porto": {
        "boundary": {
            "min_lat": 41.140092,
            "max_lat": 41.185969,
            "min_lng": -8.690261,
            "max_lng": -8.549155,
        },
        "default_grid_size_km": 0.1,
    },
    "chengdu": {
        "boundary": {
            "min_lat": 30.5,
            "max_lat": 30.8,
            "min_lng": 103.9,
            "max_lng": 104.2,
        },
        "default_grid_size_km": 0.3,
    },
}

ANOMALY_TYPES = ("detour", "switch")


def haversine_km(p1, p2):
    lat1, lng1 = p1
    lat2, lng2 = p2
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def grid_mapping(boundary: dict, grid_size_km: float) -> tuple[int, int]:
    lat_km = haversine_km(
        (boundary["min_lat"], boundary["min_lng"]),
        (boundary["max_lat"], boundary["min_lng"]),
    )
    lng_km = haversine_km(
        (boundary["min_lat"], boundary["min_lng"]),
        (boundary["min_lat"], boundary["max_lng"]),
    )
    lat_grid_num = int(lat_km / grid_size_km) + 1
    lng_grid_num = int(lng_km / grid_size_km) + 1
    return lat_grid_num, lng_grid_num


def load_object_array(path: Path) -> list:
    return np.load(path, allow_pickle=True).tolist()


def save_object_array(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(items, dtype=object))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def compact_float(value: float) -> str:
    return f"{float(value):g}"


def token_of(item):
    if isinstance(item, np.ndarray):
        item = item.tolist()
    if isinstance(item, (list, tuple)):
        if not item:
            return None
        return item[0]
    return item


def time_payload(item):
    if isinstance(item, np.ndarray):
        item = item.tolist()
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        payload = item[1]
        if isinstance(payload, np.ndarray):
            payload = payload.tolist()
        if isinstance(payload, (list, tuple)):
            return list(payload)
        return [payload]
    return None


def replace_token(item, token):
    if isinstance(item, np.ndarray):
        item = item.tolist()
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return [int(token), time_payload(item)]
    return int(token)


def replace_time(item, new_time):
    if isinstance(item, np.ndarray):
        item = item.tolist()
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return [int(token_of(item)), list(new_time)]
    return item


def _timevec_to_epoch(vec) -> int:
    # vec: [hour, minute, second, year, month, day]
    return int(
        calendar.timegm(
            dt.datetime(
                int(vec[3]),
                int(vec[4]),
                int(vec[5]),
                int(vec[0]),
                int(vec[1]),
                int(vec[2]),
            ).timetuple()
        )
    )


def _epoch_to_timevec(epoch_seconds: int) -> list[int]:
    value = dt.datetime.utcfromtimestamp(int(epoch_seconds))
    return [value.hour, value.minute, value.second, value.year, value.month, value.day]


def _time_calculate(vec, seconds: int) -> list[int]:
    return _epoch_to_timevec(_timevec_to_epoch(vec) + int(seconds))


def _convert_grid(point, map_size):
    x, y = divmod(int(point), int(map_size[1]))
    return [x, y]


def _grid_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def perturb_point(point, level, map_size, rng, offset=None):
    x, y = divmod(int(point), int(map_size[1]))
    if offset is None:
        offsets = [
            [0, 1],
            [1, 0],
            [-1, 0],
            [0, -1],
            [1, 1],
            [-1, -1],
            [-1, 1],
            [1, -1],
        ]
        x_offset, y_offset = offsets[rng.randint(0, len(offsets))]
    else:
        x_offset, y_offset = offset
    next_x = x + x_offset * int(level)
    next_y = y + y_offset * int(level)
    if 0 <= next_x < map_size[0] and 0 <= next_y < map_size[1]:
        x = next_x
        y = next_y
    return int(x * map_size[1] + y)


def perturb_time(traj, st_loc, end_loc, time_offset, interval):
    out = list(traj)
    for idx in range(st_loc, end_loc):
        payload = time_payload(out[idx])
        if payload is None or len(payload) < 6:
            return out
        shift = int((idx - st_loc + 1) * float(time_offset) * int(interval))
        out[idx] = replace_time(out[idx], _time_calculate(payload, shift))
    for idx in range(end_loc, len(out)):
        payload = time_payload(out[idx])
        if payload is None or len(payload) < 6:
            return out
        shift = int((end_loc - st_loc) * float(time_offset) * int(interval))
        out[idx] = replace_time(out[idx], _time_calculate(payload, shift))
    return out


def _switch_offset(traj, st_loc, ed_loc, map_size, rng):
    st_point = int(token_of(traj[st_loc]))
    ed_point = int(token_of(traj[ed_loc]))
    st_x, st_y = divmod(st_point, int(map_size[1]))
    ed_x, ed_y = divmod(ed_point, int(map_size[1]))
    offset = [st_x - ed_x, st_y - ed_y]

    div0 = abs(offset[0]) if offset[0] != 0 else 1
    div1 = abs(offset[1]) if offset[1] != 0 else 1

    if rng.random() < 0.5:
        return [-offset[0] / div0, offset[1] / div1]
    return [offset[0] / div0, -offset[1] / div1]


def _select_outlier_indices(num_trajs: int, ratio: float, rng) -> np.ndarray:
    if num_trajs <= 0 or ratio <= 0:
        return np.asarray([], dtype=np.int64)
    outlier_num = int(num_trajs * float(ratio))
    outlier_num = min(num_trajs, outlier_num)
    return rng.choice(num_trajs, size=outlier_num, replace=False).astype(np.int64)


def _observed_prefix(seq, observed_ratio: float):
    keep = int(len(seq) * float(observed_ratio))
    keep = max(1, min(len(seq), keep))
    return list(seq[:keep])


def generate_mstoatd_outliers(
    trajs,
    selected_idx,
    *,
    anomaly_type: str,
    level: int,
    point_prob: float,
    observed_ratio: float,
    rng,
    map_size: tuple[int, int],
    interval: int,
    no_time_perturb: bool,
):
    selected_set = {int(idx) for idx in selected_idx}
    outlier_trajs = []
    perturb_labels = []

    for idx, raw_traj in enumerate(trajs):
        traj = raw_traj.tolist() if isinstance(raw_traj, np.ndarray) else list(raw_traj)
        perturb_labels_full = [0] * len(traj)
        anomaly_len = max(1, int(len(traj) * float(point_prob)))
        anomaly_st_loc = int(rng.randint(1, max(2, len(traj) - anomaly_len - 1)))
        anomaly_ed_loc = min(len(traj) - 1, anomaly_st_loc + anomaly_len)

        if idx not in selected_set:
            perturbed = traj
        elif anomaly_type == "detour":
            perturbed = (
                traj[:anomaly_st_loc]
                + [
                    replace_token(p, perturb_point(token_of(p), level, map_size, rng))
                    for p in traj[anomaly_st_loc:anomaly_ed_loc]
                ]
                + traj[anomaly_ed_loc:]
            )
            for pos in range(anomaly_st_loc, anomaly_ed_loc):
                perturb_labels_full[pos] = 1

            if not no_time_perturb:
                dis = max(
                    _grid_distance(
                        _convert_grid(token_of(traj[anomaly_st_loc]), map_size),
                        _convert_grid(token_of(traj[anomaly_ed_loc - 1]), map_size),
                    ),
                    1,
                )
                time_offset = (int(level) * 2) / dis
                perturbed = perturb_time(
                    perturbed,
                    anomaly_st_loc,
                    anomaly_ed_loc,
                    time_offset,
                    interval,
                )
        elif anomaly_type == "switch":
            num_points_to_process = min(len(traj), 64)
            if num_points_to_process < 5:
                perturbed = traj
            else:
                interior = num_points_to_process - 2
                switch_len = max(1, int(interior * float(point_prob)))
                switch_len = min(switch_len, num_points_to_process - 4)
                switch_st_loc = int(rng.randint(1, num_points_to_process - switch_len - 2))
                switch_ed_loc = int(switch_st_loc + switch_len)
                offset = _switch_offset(traj, switch_st_loc, switch_ed_loc, map_size, rng)
                perturbed = (
                    traj[:switch_st_loc]
                    + [
                        replace_token(
                            p,
                            perturb_point(token_of(p), level, map_size, rng, offset=offset),
                        )
                        for p in traj[switch_st_loc:switch_ed_loc]
                    ]
                    + traj[switch_ed_loc:]
                )
                for pos in range(switch_st_loc, switch_ed_loc):
                    perturb_labels_full[pos] = 1
        else:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

        full = _observed_prefix(perturbed, observed_ratio)
        outlier_trajs.append(full)
        perturb_labels.append([int(v) for v in perturb_labels_full[: len(full)]])

    return outlier_trajs, perturb_labels


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "anomaly_type",
        "point_prob",
        "level",
        "num_sequences",
        "num_outliers",
        "outlier_ratio",
        "output_file",
        "idx_file",
        "label_file",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MST-OATD-style detour/switch trajectory anomalies from preprocessed test trajectories."
    )
    parser.add_argument("--dataset_name", type=str, default="porto", choices=sorted(DATASET_BOUNDARIES))
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--grid_size_km", type=float, default=None)
    parser.add_argument("--lat_rows", type=int, default=0)
    parser.add_argument("--lng_cols", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--observed_ratio", type=float, default=1.0)
    parser.add_argument("--anomaly_types", nargs="+", default=list(ANOMALY_TYPES), choices=ANOMALY_TYPES)
    parser.add_argument("--detour_point_prob", type=float, default=0.1)
    parser.add_argument("--switch_point_prob", type=float, default=0.3)
    parser.add_argument("--detour_level", type=int, default=3)
    parser.add_argument("--switch_level", type=int, default=3)
    parser.add_argument("--sampling_interval", type=int, default=15)
    parser.add_argument("--no_time_perturb", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    trajs = load_object_array(input_path)
    dataset_cfg = DATASET_BOUNDARIES[args.dataset_name]
    grid_size_km = (
        float(args.grid_size_km)
        if args.grid_size_km is not None
        else float(dataset_cfg["default_grid_size_km"])
    )
    if args.lat_rows > 0 and args.lng_cols > 0:
        map_size = (int(args.lat_rows), int(args.lng_cols))
    elif args.lat_rows > 0 or args.lng_cols > 0:
        raise ValueError("--lat_rows and --lng_cols must be provided together.")
    else:
        map_size = grid_mapping(dataset_cfg["boundary"], grid_size_km)

    summary_rows = []
    for anomaly_type in args.anomaly_types:
        if anomaly_type == "detour":
            point_prob = float(args.detour_point_prob)
            level = int(args.detour_level)
        elif anomaly_type == "switch":
            point_prob = float(args.switch_point_prob)
            level = int(args.switch_level)
        else:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

        rng = np.random.RandomState(args.seed)
        selected_idx = _select_outlier_indices(len(trajs), args.ratio, rng)
        outlier_trajs, perturb_labels = generate_mstoatd_outliers(
            trajs,
            selected_idx,
            anomaly_type=anomaly_type,
            level=level,
            point_prob=point_prob,
            observed_ratio=args.observed_ratio,
            rng=rng,
            map_size=map_size,
            interval=args.sampling_interval,
            no_time_perturb=bool(args.no_time_perturb),
        )

        tag = f"{anomaly_type}_a{compact_float(point_prob)}_d{level}"
        data_path = target_dir / f"outliers_data_{tag}_nomiss.npy"
        label_path = target_dir / f"outliers_perturb_label_{tag}_nomiss.npy"
        idx_path = target_dir / f"outliers_idx_{tag}.npy"

        save_object_array(data_path, outlier_trajs)
        save_object_array(label_path, perturb_labels)
        np.save(idx_path, selected_idx.astype(np.int64))

        summary_rows.append(
            {
                "anomaly_type": anomaly_type,
                "point_prob": compact_float(point_prob),
                "level": int(level),
                "num_sequences": int(len(outlier_trajs)),
                "num_outliers": int(len(selected_idx)),
                "outlier_ratio": (
                    float(len(selected_idx) / len(outlier_trajs)) if outlier_trajs else 0.0
                ),
                "output_file": data_path.name,
                "idx_file": idx_path.name,
                "label_file": label_path.name,
            }
        )
        print(
            f"[{anomaly_type}] wrote {data_path.name}, {idx_path.name}, {label_path.name} "
            f"(outliers={len(selected_idx)}, ratio={args.ratio:g})",
            flush=True,
        )

    metadata = {
        "dataset_name": args.dataset_name,
        "task": "trajectory anomaly construction",
        "source_file": str(input_path),
        "boundary": dataset_cfg["boundary"],
        "grid_size_km": grid_size_km,
        "grid_shape": [int(map_size[0]), int(map_size[1])],
        "ratio": float(args.ratio),
        "observed_ratio": float(args.observed_ratio),
        "anomaly_types": list(args.anomaly_types),
        "detour_point_prob": float(args.detour_point_prob),
        "switch_point_prob": float(args.switch_point_prob),
        "detour_level": int(args.detour_level),
        "switch_level": int(args.switch_level),
        "sampling_interval": int(args.sampling_interval),
        "time_perturb": not bool(args.no_time_perturb),
        "seed": int(args.seed),
        "output_convention": "outliers_data_{type}_a{point_prob}_d{level}_nomiss.npy",
    }
    save_json(target_dir / "anomaly_metadata.json", metadata)
    write_summary(target_dir / "anomaly_summary.csv", summary_rows)
    print(f"saved anomaly datasets to {target_dir}", flush=True)


if __name__ == "__main__":
    main()
