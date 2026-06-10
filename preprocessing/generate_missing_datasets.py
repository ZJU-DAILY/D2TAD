#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
from collections import deque
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

MISSING_TYPES = ("SR-TR", "SR-TC", "SC-TR", "SC-TC")
DEFAULT_MISS_RATES = (0.1, 0.3, 0.5, 0.7, 0.9)


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


def token_of(item):
    if isinstance(item, np.ndarray):
        item = item.tolist()
    if isinstance(item, (list, tuple)):
        if not item:
            return None
        return item[0]
    return item


def time_payload(item) -> list:
    if isinstance(item, np.ndarray):
        item = item.tolist()
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        payload = item[1]
        if isinstance(payload, np.ndarray):
            payload = payload.tolist()
        return list(payload) if isinstance(payload, (list, tuple)) else [payload]
    return []


def keep_slots_with_placeholders(seq, drop_indices):
    if not drop_indices:
        return list(seq)
    drop_set = set(int(idx) for idx in drop_indices)
    out = []
    for idx, item in enumerate(seq):
        if idx in drop_set:
            out.append([None, time_payload(item)])
        else:
            out.append(item)
    return out


def build_community_map(lat_grid_num, lng_grid_num, block_rows, block_cols):
    community_map = np.full(lat_grid_num * lng_grid_num, -1, dtype=np.int32)
    community_to_nodes = []
    community_rows = int(math.ceil(lat_grid_num / block_rows))
    community_cols = int(math.ceil(lng_grid_num / block_cols))
    cid = 0
    for row0 in range(0, lat_grid_num, block_rows):
        for col0 in range(0, lng_grid_num, block_cols):
            nodes = []
            for row in range(row0, min(lat_grid_num, row0 + block_rows)):
                for col in range(col0, min(lng_grid_num, col0 + block_cols)):
                    nid = row * lng_grid_num + col
                    nodes.append(nid)
                    community_map[nid] = cid
            community_to_nodes.append(np.asarray(nodes, dtype=np.int32))
            cid += 1

    adjacency = {idx: set() for idx in range(cid)}
    for crow in range(community_rows):
        for ccol in range(community_cols):
            cur = crow * community_cols + ccol
            if cur >= cid:
                continue
            for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nrow = crow + drow
                ncol = ccol + dcol
                if 0 <= nrow < community_rows and 0 <= ncol < community_cols:
                    nxt = nrow * community_cols + ncol
                    if nxt < cid:
                        adjacency[cur].add(nxt)
    return community_map, community_to_nodes, adjacency, community_rows, community_cols


def _drop_num(seq_len: int, miss_rate: float) -> int:
    if seq_len <= 0 or miss_rate <= 0:
        return 0
    return min(seq_len, max(0, int(seq_len * float(miss_rate))))


def choose_random_indices(seq_len, miss_rate, rng):
    drop_num = _drop_num(seq_len, miss_rate)
    if drop_num == 0:
        return []
    candidates = np.arange(seq_len)
    drop_idx = rng.choice(candidates, size=drop_num, replace=False).tolist()
    return sorted(int(idx) for idx in drop_idx)


def choose_block_indices(seq_len, miss_rate, rng):
    drop_num = _drop_num(seq_len, miss_rate)
    if drop_num == 0:
        return []
    start_min = 0
    start_max = seq_len - drop_num
    if start_max < start_min:
        return list(range(seq_len))
    start = int(rng.randint(start_min, start_max + 1))
    return list(range(start, start + drop_num))


def choose_spatial_random_indices(seq, miss_rate, community_map, community_adj, rng):
    seq_len = len(seq)
    drop_num = _drop_num(seq_len, miss_rate)
    if drop_num == 0:
        return []

    candidates = list(range(seq_len))
    comm_by_idx = {
        idx: int(community_map[int(token_of(seq[idx]))])
        for idx in candidates
    }
    present_comms = sorted(set(comm_by_idx.values()))
    seed = int(rng.choice(np.asarray(present_comms)))

    queue = deque([seed])
    visited = {seed}
    selected_comms = []
    pool = []
    while queue and len(pool) < drop_num:
        cur = queue.popleft()
        selected_comms.append(cur)
        selected_set = set(selected_comms)
        pool = [idx for idx in candidates if comm_by_idx[idx] in selected_set]
        neigh = list(community_adj[cur])
        rng.shuffle(neigh)
        for nxt in neigh:
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)

    if len(pool) < drop_num:
        pool = candidates
    chosen = rng.choice(np.asarray(pool), size=drop_num, replace=False).tolist()
    return sorted(int(idx) for idx in chosen)


def cell_to_xy(cell_id, width):
    return divmod(int(cell_id), int(width))


def count_components(communities, community_adj):
    communities = set(int(comm) for comm in communities)
    if not communities:
        return 0
    remaining = set(communities)
    components = 0
    while remaining:
        start = remaining.pop()
        components += 1
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in community_adj[cur]:
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
    return components


def choose_spatial_block_indices(seq, miss_rate, community_map, community_adj, lng_grid_num, rng):
    seq_len = len(seq)
    drop_num = _drop_num(seq_len, miss_rate)
    if drop_num == 0:
        return []

    start_min = 0
    start_max = seq_len - drop_num
    if start_max < start_min:
        return list(range(seq_len))

    connected_starts = []
    fallback_scores = []
    for start in range(start_min, start_max + 1):
        end = start + drop_num
        window_cells = [int(token_of(seq[idx])) for idx in range(start, end)]
        comms = {int(community_map[cell]) for cell in window_cells}
        components = count_components(comms, community_adj)
        coords = [cell_to_xy(cell, lng_grid_num) for cell in window_cells]
        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]
        bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        if components == 1:
            connected_starts.append(start)
        fallback_scores.append((components, len(comms), bbox_area, start))

    if connected_starts:
        start = int(rng.choice(np.asarray(connected_starts)))
        return list(range(start, start + drop_num))

    fallback_scores.sort(key=lambda item: (item[0], item[1], item[2]))
    best_components = fallback_scores[0][0]
    best_comm = fallback_scores[0][1]
    best_area = fallback_scores[0][2]
    best_starts = [
        start
        for components, comm_cnt, area, start in fallback_scores
        if components == best_components and comm_cnt == best_comm and area == best_area
    ]
    start = int(rng.choice(np.asarray(best_starts)))
    return list(range(start, start + drop_num))


def compute_drop_indices(seq, miss_type, miss_rate, community_map, community_adj, lng_grid_num, rng):
    if miss_type == "SR-TR":
        return choose_random_indices(len(seq), miss_rate, rng)
    if miss_type == "SR-TC":
        return choose_block_indices(len(seq), miss_rate, rng)
    if miss_type == "SC-TR":
        return choose_spatial_random_indices(seq, miss_rate, community_map, community_adj, rng)
    if miss_type == "SC-TC":
        return choose_spatial_block_indices(seq, miss_rate, community_map, community_adj, lng_grid_num, rng)
    raise ValueError(f"Unknown miss_type: {miss_type}")


def summarize_array(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.percentile(arr, 50)),
        "max": float(arr.max()),
    }


def summarize_trajs(trajs):
    return summarize_array([len(traj) for traj in trajs])


def build_summary_row(base_trajs, missing_trajs, missing_indices, miss_type, miss_rate):
    total_points = sum(len(traj) for traj in base_trajs)
    total_missing = sum(len(indices) for indices in missing_indices)
    length_stats = summarize_trajs(missing_trajs)
    return {
        "miss_type": miss_type,
        "miss_rate": miss_rate,
        "num_sequences": len(missing_trajs),
        "actual_missing_over_points": (total_missing / total_points) if total_points else 0.0,
        "actual_missing_over_candidates": (total_missing / total_points) if total_points else 0.0,
        "mean_len": length_stats.get("mean", 0.0),
        "median_len": length_stats.get("median", 0.0),
        "min_len": length_stats.get("min", 0.0),
        "max_len": length_stats.get("max", 0.0),
    }


def rate_tag(rate: float) -> str:
    return f"{float(rate):.2f}"


def parse_multi_values(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out = []
    for value in values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Generate four missing-data variants.")
    parser.add_argument("--dataset_name", type=str, default="porto", choices=sorted(DATASET_BOUNDARIES))
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--target_dir", type=Path, required=True)
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--outlier_ids_file", type=Path, default=None)
    parser.add_argument("--grid_size_km", type=float, default=None)
    parser.add_argument("--spatial_block_rows", "--block_rows", dest="spatial_block_rows", type=int, default=3)
    parser.add_argument("--spatial_block_cols", "--block_cols", dest="spatial_block_cols", type=int, default=3)
    parser.add_argument("--community_mode", type=str, default="block", choices=["block"])
    parser.add_argument("--miss_types", nargs="+", default=list(MISSING_TYPES))
    parser.add_argument("--miss_rates", nargs="+", default=[str(value) for value in DEFAULT_MISS_RATES])
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_cfg = DATASET_BOUNDARIES[args.dataset_name]
    grid_size_km = (
        float(args.grid_size_km)
        if args.grid_size_km is not None
        else float(dataset_cfg["default_grid_size_km"])
    )
    lat_grid_num, lng_grid_num = grid_mapping(dataset_cfg["boundary"], grid_size_km)
    community_map, community_nodes, community_adj, community_rows, community_cols = build_community_map(
        lat_grid_num=lat_grid_num,
        lng_grid_num=lng_grid_num,
        block_rows=int(args.spatial_block_rows),
        block_cols=int(args.spatial_block_cols),
    )

    miss_types = tuple(parse_multi_values(args.miss_types))
    invalid = [name for name in miss_types if name not in MISSING_TYPES]
    if invalid:
        raise ValueError(f"Unsupported missing type(s): {invalid}")
    miss_rates = [float(value) for value in parse_multi_values(args.miss_rates)]

    input_file = args.input_file.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or input_file.stem
    seqs = load_object_array(input_file)

    summaries = []
    scenario_rows = []
    for miss_type in miss_types:
        for miss_rate in miss_rates:
            rng = np.random.RandomState(
                int(args.seed)
                + int(round(float(miss_rate) * 1000))
                + sum(ord(ch) for ch in miss_type)
            )
            missing_data = []
            missing_indices = []
            for seq in seqs:
                if isinstance(seq, np.ndarray):
                    seq = seq.tolist()
                else:
                    seq = list(seq)
                drop_indices = compute_drop_indices(
                    seq,
                    miss_type=miss_type,
                    miss_rate=miss_rate,
                    community_map=community_map,
                    community_adj=community_adj,
                    lng_grid_num=lng_grid_num,
                    rng=rng,
                )
                missing_data.append(keep_slots_with_placeholders(seq, drop_indices))
                missing_indices.append(np.asarray(drop_indices, dtype=np.int16))

            tag = f"{prefix}_{miss_type}_{rate_tag(miss_rate)}"
            save_object_array(target_dir / f"{tag}.npy", missing_data)
            save_object_array(target_dir / f"missing_idx_{tag}.npy", missing_indices)
            scenario_rows.append(
                build_summary_row(
                    seqs,
                    missing_data,
                    missing_indices,
                    miss_type,
                    miss_rate,
                )
            )
            summaries.append(
                {
                    "missing_type": miss_type,
                    "missing_rate": float(miss_rate),
                    "mean_missing_points": float(np.mean([len(x) for x in missing_indices]))
                    if missing_indices
                    else 0.0,
                    "output_data": f"{tag}.npy",
                    "output_missing_idx": f"missing_idx_{tag}.npy",
                }
            )

    if args.outlier_ids_file is not None:
        dst = target_dir / f"outlier_ids_{prefix}.npy"
        shutil.copyfile(args.outlier_ids_file.expanduser().resolve(), dst)

    metadata = {
        "dataset_name": args.dataset_name,
        "input_file": str(input_file),
        "grid_size_km": float(grid_size_km),
        "grid_shape": [int(lat_grid_num), int(lng_grid_num)],
        "spatial_block_rows": int(args.spatial_block_rows),
        "spatial_block_cols": int(args.spatial_block_cols),
        "community_mode": "block",
        "community_method": "grid_block",
        "community_grid_shape": [int(community_rows), int(community_cols)],
        "num_spatial_communities": int(len(community_nodes)),
        "missing_types": list(miss_types),
        "missing_rates": miss_rates,
        "missing_value_format": "replace missing points with [None, time_vec] while preserving trajectory length",
        "sc_tr_strategy": "random seed community + BFS connected expansion, then random point sampling inside region",
        "sc_tc_strategy": "enumerate fixed-length contiguous windows, prefer spatially connected community windows, random among ties, fallback to lowest components/communities/bbox",
        "seed": int(args.seed),
        "outputs": summaries,
    }
    save_json(target_dir / "missing_metadata.json", metadata)

    summary_path = target_dir / "scenario_stats.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "miss_type",
                "miss_rate",
                "num_sequences",
                "actual_missing_over_points",
                "actual_missing_over_candidates",
                "mean_len",
                "median_len",
                "min_len",
                "max_len",
            ],
        )
        writer.writeheader()
        writer.writerows(scenario_rows)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
