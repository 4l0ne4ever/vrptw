#!/usr/bin/env python3
"""
Utility script to regenerate realistic Hanoi test datasets.

Customers are sampled around well-known urban districts to avoid
placing points in rivers or lakes, ensuring map overlays make sense.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd


DATASET_CONFIG = [
    ("hanoi_small_8_customers", 8),
    ("hanoi_medium_15_customers", 15),
    ("hanoi_large_25_customers", 25),
    ("hanoi_xlarge_100_customers", 100),
]


@dataclass(frozen=True)
class RectZone:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# Safe zones - chỉ bao phủ khu vực đô thị có đường, tránh sông hồ
# Đã được điều chỉnh cẩn thận để không trùng với WATER_ZONES
RECT_ZONES: Sequence[RectZone] = [
    # Ba Dinh - tránh Hồ Tây và Trúc Bạch, chỉ phần phía đông
    RectZone("Ba Dinh", 21.030, 21.050, 105.825, 105.838),
    # Cầu Giấy - tránh Hồ Tây, chỉ phần phía đông
    RectZone("Cau Giay", 21.018, 21.040, 105.775, 105.798),
    # Đống Đa - khu vực trung tâm an toàn, tránh Hồ Hoàn Kiếm
    RectZone("Dong Da", 21.005, 21.028, 105.815, 105.845),
    # Hai Bà Trưng - tránh sông Hồng, chỉ phần phía tây
    RectZone("Hai Ba Trung", 20.998, 21.018, 105.855, 105.868),
    # Thanh Xuân - khu vực an toàn
    RectZone("Thanh Xuan", 20.998, 21.012, 105.792, 105.820),
    # Tây Hồ - chỉ phần phía bắc xa Hồ Tây
    RectZone("Tay Ho", 21.080, 21.095, 105.760, 105.785),
    # Hà Đông - khu vực an toàn
    RectZone("Ha Dong", 20.950, 20.975, 105.752, 105.775),
    # Nam Từ Liêm - tránh Hồ Tây, chỉ phần phía đông
    RectZone("Nam Tu Liem", 21.008, 21.035, 105.760, 105.788),
    # Đông Anh - khu vực an toàn
    RectZone("Dong Anh", 21.085, 21.105, 105.835, 105.875),
]

# Vùng nước cần tránh - bao gồm sông Hồng, các hồ lớn
WATER_ZONES: Sequence[RectZone] = [
    # Hồ Tây (West Lake) - hồ lớn nhất
    RectZone("West Lake", 21.035, 21.090, 105.800, 105.850),
    # Hồ Trúc Bạch
    RectZone("Truc Bach", 21.038, 21.055, 105.835, 105.850),
    # Hồ Linh Đàm
    RectZone("Linh Dam", 20.965, 20.990, 105.845, 105.890),
    # Hồ Yên Sở
    RectZone("Yen So", 20.960, 20.990, 105.880, 105.925),
    # Sông Hồng - đoạn chảy qua Hà Nội (rộng khoảng 1-2km)
    RectZone("Red River North", 21.020, 21.110, 105.860, 105.920),
    RectZone("Red River South", 20.950, 21.020, 105.870, 105.930),
    # Hồ Hoàn Kiếm (nếu trong phạm vi)
    RectZone("Hoan Kiem", 21.025, 21.032, 105.848, 105.855),
    # Các hồ nhỏ khác
    RectZone("Small Lake 1", 21.015, 21.025, 105.840, 105.850),
    RectZone("Small Lake 2", 20.980, 20.995, 105.860, 105.875),
]

# Depot - đặt ở khu vực trung tâm an toàn, tránh sông hồ
# Vị trí: Đống Đa, khu vực có đường, không có hồ
DEPOT = {
    "id": 0,
    "x": 105.8300,  # Điều chỉnh để tránh tất cả vùng nước
    "y": 21.0150,   # Điều chỉnh để tránh các hồ
    "demand": 0,
    "ready_time": 0.0,
    "due_date": 1000.0,
    "service_time": 0.0,
}

DATA_DIR = Path("data/test_datasets")
MIN_DISTANCE_KM = 0.25  # Ensure customers dispersed (giảm xuống để dễ tạo dataset lớn)
VEHICLE_CAPACITY = 200


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _point_in_rect(lat: float, lon: float, rect: RectZone) -> bool:
    return rect.lat_min <= lat <= rect.lat_max and rect.lon_min <= lon <= rect.lon_max


def _is_water(lat: float, lon: float) -> bool:
    return any(_point_in_rect(lat, lon, rect) for rect in WATER_ZONES)


def random_point_in_zone(zone: RectZone) -> Tuple[float, float]:
    """Draw a random point inside a rectangular safe zone."""
    # Tăng số lần thử để đảm bảo tìm được điểm không trong nước
    for _ in range(5000):
        lat = random.uniform(zone.lat_min, zone.lat_max)
        lon = random.uniform(zone.lon_min, zone.lon_max)
        if not _is_water(lat, lon):
            return lat, lon
    # Nếu không tìm được, thử với vùng nhỏ hơn ở giữa zone
    center_lat = (zone.lat_min + zone.lat_max) / 2
    center_lon = (zone.lon_min + zone.lon_max) / 2
    shrink = 0.3  # Thu nhỏ 30%
    for _ in range(5000):
        lat = random.uniform(
            center_lat - (zone.lat_max - zone.lat_min) * shrink / 2,
            center_lat + (zone.lat_max - zone.lat_min) * shrink / 2
        )
        lon = random.uniform(
            center_lon - (zone.lon_max - zone.lon_min) * shrink / 2,
            center_lon + (zone.lon_max - zone.lon_min) * shrink / 2
        )
        if not _is_water(lat, lon):
            return lat, lon
    raise RuntimeError(f"Could not sample non-water point inside zone {zone.name}")


def _build_zone_sequence(num_customers: int) -> List[RectZone]:
    """Build sequence encouraging coverage but avoiding overuse of a single zone."""
    zones = list(RECT_ZONES)
    random.shuffle(zones)
    sequence: List[RectZone] = zones[:min(len(zones), num_customers)]
    while len(sequence) < num_customers:
        sequence.append(random.choice(RECT_ZONES))
    random.shuffle(sequence)
    return sequence[:num_customers]


def sample_customer(existing: List[Tuple[float, float]], zone: RectZone) -> Tuple[float, float]:
    """Sample a customer location in provided zone ensuring spacing."""
    # Tăng số lần thử cho dataset lớn
    max_attempts = 8000 if len(existing) > 50 else 4000
    for _ in range(max_attempts):
        lat, lon = random_point_in_zone(zone)
        if all(haversine_km(lat, lon, ex_lat, ex_lon) >= MIN_DISTANCE_KM for ex_lat, ex_lon in existing):
            return lat, lon
    raise RuntimeError(f"Could not sample customer inside zone {zone.name} with spacing constraint.")


def validate_dataset(dataset: dict) -> Tuple[bool, List[str]]:
    """Validate dataset - check for water points and return issues."""
    issues = []
    
    # Check depot
    depot = dataset["depot"]
    if _is_water(depot["y"], depot["x"]):
        issues.append(f"Depot at ({depot['x']:.6f}, {depot['y']:.6f}) is in water")
    
    # Check all customers
    for customer in dataset["customers"]:
        lat, lon = customer["y"], customer["x"]
        if _is_water(lat, lon):
            issues.append(f"Customer {customer['id']} at ({lon:.6f}, {lat:.6f}) is in water")
    
    return len(issues) == 0, issues


def build_dataset(num_customers: int) -> Tuple[dict, List[str]]:
    """Build dataset dictionary and return zone usage log."""
    existing_coords = [(DEPOT["y"], DEPOT["x"])]
    customers = []
    zones_used = []
    total_demand = 0
    zone_sequence = _build_zone_sequence(num_customers)

    for cid, zone in zip(range(1, num_customers + 1), zone_sequence):
        lat, lon = sample_customer(existing_coords, zone)
        existing_coords.append((lat, lon))
        zones_used.append(zone.name)
        demand = random.randint(5, 30)
        total_demand += demand
        customers.append(
            {
                "id": cid,
                "x": round(lon, 6),
                "y": round(lat, 6),
                "demand": demand,
                "ready_time": 0.0,
                "due_date": 1000.0,
                "service_time": 10.0,
            }
        )

    num_vehicles = max(1, math.ceil(total_demand / VEHICLE_CAPACITY))

    dataset = {
        "depot": DEPOT,
        "customers": customers,
        "vehicle_capacity": VEHICLE_CAPACITY,
        "num_vehicles": num_vehicles,
        "metadata": {
            "name": f"Hanoi Dataset ({num_customers} customers)",
            "source": "generated",
            "format": "hanoi_mockup",
            "num_customers": num_customers,
            "safe_zones": sorted(set(zones_used)),
        },
        "problem_config": {
            "vehicle_capacity": VEHICLE_CAPACITY,
            "num_vehicles": num_vehicles,
            "traffic_factor": 1.0,
        },
    }
    
    # Validate dataset
    is_valid, issues = validate_dataset(dataset)
    if not is_valid:
        raise RuntimeError(f"Dataset validation failed:\n" + "\n".join(issues))

    return dataset, zones_used


def write_dataset(dataset: dict, base_name: str) -> None:
    """Write JSON, CSV, and Excel versions of a dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / f"{base_name}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)

    rows = [
        {
            "id": dataset["depot"]["id"],
            "x": dataset["depot"]["x"],
            "y": dataset["depot"]["y"],
            "demand": dataset["depot"]["demand"],
            "ready_time": dataset["depot"]["ready_time"],
            "due_date": dataset["depot"]["due_date"],
            "service_time": dataset["depot"]["service_time"],
            "vehicle_capacity": dataset["vehicle_capacity"],
            "num_vehicles": dataset["num_vehicles"],
        }
    ]
    rows.extend(
        {
            "id": customer["id"],
            "x": customer["x"],
            "y": customer["y"],
            "demand": customer["demand"],
            "ready_time": customer["ready_time"],
            "due_date": customer["due_date"],
            "service_time": customer["service_time"],
            "vehicle_capacity": dataset["vehicle_capacity"],
            "num_vehicles": dataset["num_vehicles"],
        }
        for customer in dataset["customers"]
    )

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / f"{base_name}.csv", index=False)
    df.to_excel(DATA_DIR / f"{base_name}.xlsx", index=False)


def clean_previous_files() -> None:
    """Remove previous dataset files to avoid confusion."""
    if not DATA_DIR.exists():
        return
    for pattern in ("*.json", "*.csv", "*.xlsx"):
        for file in DATA_DIR.glob(pattern):
            if file.stem.startswith("hanoi_"):
                file.unlink()


def regenerate() -> None:
    """Regenerate all datasets defined in DATASET_CONFIG."""
    random.seed(20251117)
    clean_previous_files()

    for base_name, num_customers in DATASET_CONFIG:
        dataset, zones_used = build_dataset(num_customers)
        write_dataset(dataset, base_name)
        zones_str = ", ".join(sorted(set(zones_used)))
        print(f"[OK] {base_name} -> {num_customers} customers ({zones_str})")


if __name__ == "__main__":
    regenerate()

