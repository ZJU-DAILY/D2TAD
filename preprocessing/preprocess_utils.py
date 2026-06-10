import datetime as dt
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np


PORTO_BOUNDARY = {
    "min_lat": 41.140092,
    "max_lat": 41.185969,
    "min_lng": -8.690261,
    "max_lng": -8.549155,
}

CHENGDU_BOUNDARY = {
    "min_lat": 30.5,
    "max_lat": 30.8,
    "min_lng": 103.9,
    "max_lng": 104.2,
}


def in_boundary(lat: float, lng: float, boundary: dict) -> bool:
    return (
        boundary["min_lng"] < lng < boundary["max_lng"]
        and boundary["min_lat"] < lat < boundary["max_lat"]
    )


def haversine_km(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1, lng1 = p1
    lat2, lng2 = p2
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def grid_mapping(boundary: dict, grid_size_km: float) -> tuple[float, float, int, int]:
    lat_km = haversine_km(
        (boundary["min_lat"], boundary["min_lng"]),
        (boundary["max_lat"], boundary["min_lng"]),
    )
    lng_km = haversine_km(
        (boundary["min_lat"], boundary["min_lng"]),
        (boundary["min_lat"], boundary["max_lng"]),
    )
    lat_size = (boundary["max_lat"] - boundary["min_lat"]) / lat_km * grid_size_km
    lng_size = (boundary["max_lng"] - boundary["min_lng"]) / lng_km * grid_size_km
    lat_rows = int(lat_km / grid_size_km) + 1
    lng_cols = int(lng_km / grid_size_km) + 1
    return lat_size, lng_size, lat_rows, lng_cols


def point_to_token(lat: float, lng: float, boundary: dict, lat_size: float, lng_size: float, lng_cols: int) -> int:
    row = int((lat - boundary["min_lat"]) / lat_size)
    col = int((lng - boundary["min_lng"]) / lng_size)
    return int(row * lng_cols + col)


def cutting_trajs(traj: list, longest: int, shortest: int, rng: random.Random) -> list[list]:
    out = []
    rest = list(traj)
    while len(rest) > longest:
        cut_len = rng.randint(shortest, longest)
        out.append(rest[:cut_len])
        rest = rest[cut_len:]
    if len(rest) >= shortest:
        out.append(rest)
    return out


def porto_time_vector(timestamp: int) -> list[int]:
    value = dt.datetime.fromtimestamp(int(timestamp))
    return [value.hour, value.minute, value.second, value.year, value.month, value.day]


def chengdu_time_vector(text: str) -> list[int]:
    parsed = time.strptime(str(text), "%Y/%m/%d %H:%M:%S")
    return [
        parsed.tm_hour,
        parsed.tm_min,
        parsed.tm_sec,
        parsed.tm_year,
        parsed.tm_mon,
        parsed.tm_mday,
    ]


def timestamp_gap_seconds(left: str, right: str) -> float:
    left_dt = dt.datetime.strptime(str(left), "%Y/%m/%d %H:%M:%S")
    right_dt = dt.datetime.strptime(str(right), "%Y/%m/%d %H:%M:%S")
    return (right_dt - left_dt).total_seconds()


def parse_polyline(text: str) -> Iterable[tuple[float, float]]:
    for lng, lat in json.loads(text):
        yield float(lat), float(lng)


def save_object_array(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(items, dtype=object))
