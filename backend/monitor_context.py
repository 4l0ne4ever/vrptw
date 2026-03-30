"""
Context builder for monitoring tab map payload.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

from config import VRP_CONFIG
from src.simulation.route_geometry import get_route_polyline


def build_monitor_context(artifact_path: str, replay_slot: int = 0) -> Dict[str, Any]:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError("Artifacts not found")
    with path.open("rb") as f:
        artifacts = pickle.load(f)

    problem = artifacts.get("problem")
    result = artifacts.get("result", {})
    solution = result.get("solution", {})
    routes = solution.get("routes", [])

    meta = getattr(problem, "metadata", {}) or {}
    dt = str(meta.get("dataset_type") or getattr(problem, "dataset_type", "") or "").strip().lower()
    is_solomon = dt.startswith("solomon")
    start_minutes = float(problem.depot.ready_time) if is_solomon else float(VRP_CONFIG.get("time_window_start", 480))

    customer_vehicle_index: Dict[str, int] = {}
    for ri, route in enumerate(routes):
        for nid in route or []:
            n = int(nid)
            if n != 0:
                customer_vehicle_index[str(n)] = ri

    depot = {
        "id": 0,
        "lat": float(problem.depot.y),
        "lon": float(problem.depot.x),
    }
    customers: List[Dict[str, Any]] = []
    for c in problem.customers:
        customers.append(
            {
                "id": int(c.id),
                "lat": float(c.y),
                "lon": float(c.x),
                "demand": float(c.demand),
                "ready_time": float(c.ready_time),
                "due_date": float(c.due_date),
                "service_time": float(c.service_time),
            }
        )

    def node_lon_lat(node_id: int) -> tuple[float, float]:
        if node_id == 0:
            return (float(problem.depot.x), float(problem.depot.y))
        customer = problem.get_customer_by_id(int(node_id))
        if not customer:
            return (float(problem.depot.x), float(problem.depot.y))
        return (float(customer.x), float(customer.y))

    polylines: List[Dict[str, Any]] = []
    total_points = 0
    for idx, route in enumerate(routes):
        coords: List[Dict[str, float]] = []
        # Chain OSRM polylines segment-by-segment to ensure roads, not straight lines.
        for j in range(len(route) - 1):
            from_node = int(route[j])
            to_node = int(route[j + 1])
            origin = node_lon_lat(from_node)
            destination = node_lon_lat(to_node)
            seg_poly = get_route_polyline(origin, destination)
            if not seg_poly:
                continue
            if coords and seg_poly:
                last = coords[-1]
                first = seg_poly[0]
                if float(last["lat"]) == float(first["lat"]) and float(last["lon"]) == float(first["lon"]):
                    coords.extend(seg_poly[1:])
                else:
                    coords.extend(seg_poly)
            else:
                coords.extend(seg_poly)

        polylines.append({"route_id": idx + 1, "coordinates": coords})
        total_points += len(coords)

    return {
        "replay_slot": replay_slot,
        "depot": depot,
        "customers": customers,
        "polylines": polylines,
        "plan_revision": int(artifacts.get("plan_revision", 0)),
        "replay_duration_h": round(total_points * 0.05, 3),
        "start_minutes": start_minutes,
        "customer_vehicle_index": customer_vehicle_index,
    }

