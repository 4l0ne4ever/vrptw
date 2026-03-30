"""
Route playback: each vehicle keeps its own route and state (planned/actual time,
segment index, polyline steps). Vehicles do not share movement state.

The outer loop only orders *events* by simulation time (next travel point or
post-service) and advances wall-clock pacing between event times so one global
simulation timeline matches the UI clock; that is the only cross-vehicle coupling.

OSRM segment geometry when enabled; travel time scaled by `backend.traffic_state`;
TW checks at customer arrival.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional, Tuple

from backend.realtime import outbound_queue
from backend.traffic_state import get_current_observation, record_runtime_signal
from config import VRP_CONFIG
from src.messaging.kafka_telemetry import emit_alert, emit_telemetry
from src.runtime_settings import REPLAY_DWELL_SPEEDUP, REPLAY_DWELL_STEP_MIN, REPLAY_WAIT_TARGET_EVENTS
from src.simulation.route_geometry import get_route_segment, haversine_distance_m

logger = logging.getLogger(__name__)


def _node_lon_lat(problem, node_id: int) -> Tuple[float, float]:
    if node_id == 0:
        return float(problem.depot.x), float(problem.depot.y)
    c = problem.get_customer_by_id(int(node_id))
    if not c:
        return float(problem.depot.x), float(problem.depot.y)
    return float(c.x), float(c.y)


def _detect_dataset_type(problem) -> str:
    meta = getattr(problem, "metadata", {}) or {}
    dt = meta.get("dataset_type") or getattr(problem, "dataset_type", None) or "hanoi"
    return str(dt).strip().lower()


MAX_TELEMETRY_POINTS_PER_SEGMENT = 180
DWELL_TELEMETRY_STEP_MIN = max(0.01, float(REPLAY_DWELL_STEP_MIN))
DWELL_SPEEDUP = max(1.0, float(REPLAY_DWELL_SPEEDUP))
WAIT_TARGET_EVENTS = max(1, int(REPLAY_WAIT_TARGET_EVENTS))


def _warm_route_segment_cache(problem: Any, routes: List[List[int]]) -> None:
    for route in routes:
        if not route or len(route) < 2:
            continue
        for j in range(len(route) - 1):
            o = _node_lon_lat(problem, int(route[j]))
            d = _node_lon_lat(problem, int(route[j + 1]))
            get_route_segment(o, d)


class _VehicleReplayer:
    __slots__ = (
        "problem",
        "vehicle_id",
        "route",
        "run_id",
        "replay_slot",
        "start_minutes",
        "is_solomon",
        "planned_minutes",
        "actual_minutes",
        "seg_i",
        "from_node",
        "to_node",
        "step_points",
        "n_substeps",
        "dt_base_min",
        "next_pidx",
        "prev_emit",
        "last_dt_s",
        "done",
        "post_service_time",
        "post_service_start_time",
        "post_planned_end_time",
        "post_wait_step_min",
        "post_customer_id",
        "post_violation",
        "post_lateness",
    )

    def __init__(
        self,
        problem: Any,
        vehicle_id: int,
        route: List[int],
        run_id: str,
        replay_slot: int,
        start_minutes: float,
        is_solomon: bool,
    ) -> None:
        self.problem = problem
        self.vehicle_id = vehicle_id
        self.route = route
        self.run_id = run_id
        self.replay_slot = replay_slot
        self.start_minutes = start_minutes
        self.is_solomon = is_solomon

        self.planned_minutes = start_minutes
        self.actual_minutes = start_minutes
        self.seg_i = 0
        self.from_node = 0
        self.to_node = 0
        self.step_points: List[Dict[str, Any]] = []
        self.n_substeps = 1
        self.dt_base_min = 0.0
        self.next_pidx = 0
        self.prev_emit: Optional[Dict[str, float]] = None
        self.last_dt_s = 0.0
        self.done = len(route) < 2
        self.post_service_time: Optional[float] = None
        self.post_service_start_time: Optional[float] = None
        self.post_planned_end_time: Optional[float] = None
        self.post_wait_step_min: float = DWELL_TELEMETRY_STEP_MIN
        self.post_customer_id: Optional[int] = None
        self.post_violation = False
        self.post_lateness = 0.0

        if not self.done:
            self._load_segment(0)

    def elapsed_h(self, m: float) -> float:
        return max(0.0, (m - self.start_minutes) / 60.0)

    def _load_segment(self, seg_i: int) -> None:
        self.seg_i = seg_i
        route = self.route
        self.from_node = int(route[seg_i])
        self.to_node = int(route[seg_i + 1])
        origin = _node_lon_lat(self.problem, self.from_node)
        destination = _node_lon_lat(self.problem, self.to_node)
        seg = get_route_segment(origin, destination)
        poly: List[Dict[str, Any]] = list(seg["polyline"])
        duration_s = float(seg["duration_s"])

        if len(poly) < 2:
            poly = [
                {"lat": origin[1], "lon": origin[0]},
                {"lat": destination[1], "lon": destination[0]},
            ]
        if duration_s <= 0.0:
            dist_m = haversine_distance_m(origin[0], origin[1], destination[0], destination[1])
            duration_s = max(0.1, dist_m / 8.33)

        if len(poly) <= MAX_TELEMETRY_POINTS_PER_SEGMENT:
            step_points = poly
        else:
            stride = max(1, int(len(poly) / MAX_TELEMETRY_POINTS_PER_SEGMENT))
            step_points = [poly[i] for i in range(0, len(poly), stride)]
            if step_points[-1] != poly[-1]:
                step_points.append(poly[-1])

        self.step_points = step_points
        self.n_substeps = max(1, len(step_points) - 1)
        self.dt_base_min = (duration_s / 60.0) / self.n_substeps
        self.next_pidx = 0 if seg_i == 0 else 1

    def _is_last_overall_point(self, pidx: int) -> bool:
        return self.seg_i == len(self.route) - 2 and pidx == len(self.step_points) - 1 and self.to_node == 0

    def next_event_time(self) -> Optional[float]:
        if self.done:
            return None
        if self.post_service_time is not None:
            remaining = max(0.0, float(self.post_service_time - self.actual_minutes))
            return float(self.actual_minutes + min(DWELL_TELEMETRY_STEP_MIN, remaining))
        if not self.step_points:
            return None
        pidx = self.next_pidx
        if pidx >= len(self.step_points):
            return None
        if pidx > 0:
            factor = max(
                0.1,
                float(
                    get_current_observation(
                        self.run_id, sim_time_h=self.elapsed_h(self.actual_minutes)
                    ).factor
                ),
            )
            return float(self.actual_minutes + self.dt_base_min * factor)
        return float(self.actual_minutes)

    def _schedule_post_service(self) -> None:
        to_node = self.to_node
        if to_node == 0:
            self.post_service_time = None
            self.post_service_start_time = None
            self.post_planned_end_time = None
            self.post_wait_step_min = DWELL_TELEMETRY_STEP_MIN
            self.post_customer_id = None
            self.post_violation = False
            self.post_lateness = 0.0
            return
        customer = self.problem.get_customer_by_id(to_node)
        if not customer:
            self.post_service_time = None
            self.post_service_start_time = None
            self.post_planned_end_time = None
            self.post_wait_step_min = DWELL_TELEMETRY_STEP_MIN
            self.post_customer_id = None
            self.post_violation = False
            self.post_lateness = 0.0
            self._advance_segment()
            return
        planned_arrival = self.planned_minutes
        actual_arrival = self.actual_minutes
        planned_start_service = max(planned_arrival, float(customer.ready_time))
        actual_start_service = max(actual_arrival, float(customer.ready_time))
        self.post_service_time = float(actual_start_service + float(customer.service_time))
        self.post_service_start_time = float(actual_start_service)
        self.post_planned_end_time = float(planned_start_service + float(customer.service_time))
        wait_total_min = max(0.0, actual_start_service - actual_arrival)
        self.post_wait_step_min = max(DWELL_TELEMETRY_STEP_MIN, wait_total_min / WAIT_TARGET_EVENTS)
        self.post_customer_id = to_node

        if self.is_solomon:
            violated = (actual_arrival < float(customer.ready_time)) or (actual_arrival > float(customer.due_date))
            if actual_arrival < float(customer.ready_time):
                lateness = float(customer.ready_time) - actual_arrival
            else:
                lateness = actual_arrival - float(customer.due_date)
        else:
            violated = actual_start_service > float(customer.due_date)
            lateness = actual_start_service - float(customer.due_date) if violated else 0.0

        self.post_violation = bool(violated)
        self.post_lateness = float(max(0.0, lateness))

        wait_min = wait_total_min
        service_min = float(customer.service_time)
        if wait_min > 0.0:
            logger.info(
                "Replay vehicle %s waiting at customer %s for %.2f min (arr=%.2f, ready=%.2f)",
                self.vehicle_id,
                to_node,
                wait_min,
                actual_arrival,
                float(customer.ready_time),
            )
        if service_min > 0.0:
            logger.info(
                "Replay vehicle %s service at customer %s for %.2f min",
                self.vehicle_id,
                to_node,
                service_min,
            )

    def _emit_stationary_telemetry(self, status: str) -> None:
        point = self.prev_emit
        if point is None:
            lon, lat = _node_lon_lat(self.problem, self.to_node)
            point = {"lat": float(lat), "lon": float(lon)}
            self.prev_emit = point
        clock_minutes = float(self.actual_minutes % 1440.0)

        # Remaining time (minutes of simulation time) in current dwell phase.
        dwell_remaining_minutes: Optional[float] = None
        if status == "waiting" and self.post_service_start_time is not None:
            dwell_remaining_minutes = max(0.0, float(self.post_service_start_time - self.actual_minutes))
        elif status in {"servicing", "service_done"} and self.post_service_time is not None:
            dwell_remaining_minutes = max(0.0, float(self.post_service_time - self.actual_minutes))

        payload = {
            "run_id": self.run_id,
            "replay_slot": self.replay_slot,
            "vehicle_id": self.vehicle_id,
            "lat": float(point["lat"]),
            "lon": float(point["lon"]),
            "sim_time_h": self.elapsed_h(self.actual_minutes),
            "clock_minutes": clock_minutes,
            "speed_kmh": 0.0,
            "status": status,
            "next_customer_id": self.to_node,
            "eta_h": self.elapsed_h(self.actual_minutes),
            "planned_arrival_h": self.elapsed_h(self.planned_minutes),
            "arrived_customer_id": None,
            "dwell_remaining_minutes": dwell_remaining_minutes,
        }
        obs = get_current_observation(self.run_id, sim_time_h=self.elapsed_h(self.actual_minutes))
        payload["traffic_factor"] = obs.factor
        payload["traffic_source"] = obs.source
        emit_telemetry(payload)
        record_runtime_signal(
            self.run_id,
            delay_minutes=max(0.0, self.actual_minutes - self.planned_minutes),
            violation=False,
        )

    def _advance_post_service_chunk(self) -> bool:
        """
        Returns True when dwell (wait/service) is complete for current customer.
        """
        if self.post_service_time is None:
            return True
        remaining_actual = max(0.0, float(self.post_service_time - self.actual_minutes))
        if remaining_actual <= 1e-9:
            return True
        in_waiting_phase = self.post_service_start_time is not None and self.actual_minutes < (
            self.post_service_start_time - 1e-9
        )
        if in_waiting_phase:
            step_actual = min(remaining_actual, self.post_wait_step_min)
        else:
            step_actual = min(DWELL_TELEMETRY_STEP_MIN, remaining_actual)
        self.actual_minutes += step_actual

        if self.post_planned_end_time is not None:
            remaining_planned = max(0.0, float(self.post_planned_end_time - self.planned_minutes))
            self.planned_minutes += min(step_actual, remaining_planned)

        if self.post_service_start_time is not None and self.actual_minutes < (self.post_service_start_time - 1e-9):
            status = "waiting"
        elif self.actual_minutes < (float(self.post_service_time) - 1e-9):
            status = "servicing"
        else:
            status = "service_done"
        self._emit_stationary_telemetry(status)
        return self.actual_minutes >= float(self.post_service_time) - 1e-9

    def _finish_post_service(self) -> None:
        cid = self.post_customer_id
        was_violated = self.post_violation
        lateness = self.post_lateness
        self.post_service_time = None
        self.post_service_start_time = None
        self.post_planned_end_time = None
        self.post_wait_step_min = DWELL_TELEMETRY_STEP_MIN
        self.post_customer_id = None
        self.post_violation = False
        self.post_lateness = 0.0
        if cid is None:
            return
        if was_violated:
            emit_alert(
                {
                    "run_id": self.run_id,
                    "replay_slot": self.replay_slot,
                    "type": "tw_violation",
                    "vehicle_id": self.vehicle_id,
                    "customer_id": cid,
                    "lateness_minutes": float(lateness),
                    "sim_time_h": self.elapsed_h(self.actual_minutes),
                }
            )
            record_runtime_signal(self.run_id, delay_minutes=max(0.0, lateness), violation=True)

    def _advance_segment(self) -> None:
        next_seg = self.seg_i + 1
        if next_seg >= len(self.route) - 1:
            self.done = True
            return
        self._load_segment(next_seg)

    def emit_one_step(self, stop_event: Optional[Event]) -> str:
        """
        Returns 'telemetry', 'post_service', or 'done' (vehicle finished).
        """
        if self.done:
            return "done"

        if self.post_service_time is not None:
            if self._advance_post_service_chunk():
                self._finish_post_service()
                self._advance_segment()
                return "post_service"
            return "telemetry"

        pidx = self.next_pidx
        if pidx >= len(self.step_points):
            self.done = True
            return "done"

        if pidx > 0:
            factor = max(
                0.1,
                float(
                    get_current_observation(
                        self.run_id, sim_time_h=self.elapsed_h(self.actual_minutes)
                    ).factor
                ),
            )
            self.planned_minutes += self.dt_base_min
            self.actual_minutes += self.dt_base_min * factor
            self.last_dt_s = (self.dt_base_min * 60.0) * factor
        else:
            self.last_dt_s = 0.0

        is_last_overall = self._is_last_overall_point(pidx)
        is_arrival_customer = pidx == len(self.step_points) - 1 and self.to_node != 0
        if is_last_overall:
            status = "done"
        elif is_arrival_customer:
            status = "arrived"
        else:
            status = "en_route"
        point = self.step_points[pidx]
        speed_kmh = 0.0
        if self.prev_emit is not None and self.last_dt_s > 1e-6:
            dist_m = haversine_distance_m(
                self.prev_emit["lon"],
                self.prev_emit["lat"],
                float(point["lon"]),
                float(point["lat"]),
            )
            speed_kmh = (dist_m / 1000.0) / (self.last_dt_s / 3600.0)
        self.prev_emit = {"lat": float(point["lat"]), "lon": float(point["lon"])}

        clock_minutes = float(self.actual_minutes % 1440.0)
        payload = {
            "run_id": self.run_id,
            "replay_slot": self.replay_slot,
            "vehicle_id": self.vehicle_id,
            "lat": float(point["lat"]),
            "lon": float(point["lon"]),
            "sim_time_h": self.elapsed_h(self.actual_minutes),
            "clock_minutes": clock_minutes,
            "speed_kmh": float(round(speed_kmh, 2)),
            "status": status,
            "next_customer_id": self.to_node,
            "eta_h": self.elapsed_h(self.actual_minutes),
            "planned_arrival_h": self.elapsed_h(self.planned_minutes),
            "arrived_customer_id": self.to_node if is_arrival_customer else None,
        }
        obs = get_current_observation(self.run_id, sim_time_h=self.elapsed_h(self.actual_minutes))
        payload["traffic_factor"] = obs.factor
        payload["traffic_source"] = obs.source
        emit_telemetry(payload)
        record_runtime_signal(
            self.run_id,
            delay_minutes=max(0.0, self.actual_minutes - self.planned_minutes),
            violation=False,
        )

        self.next_pidx = pidx + 1
        if self.next_pidx >= len(self.step_points):
            if self.to_node != 0:
                self._schedule_post_service()
            else:
                self._advance_segment()

        return "telemetry"


def replay_solution(
    run_id: str,
    artifact_path: str,
    replay_slot: int = 0,
    hours_per_real_second: float = 1.0,
    stop_event: Optional[Event] = None,
) -> None:
    path = Path(artifact_path)
    if not path.exists():
        outbound_queue.put({"type": "monitor_error", "run_id": run_id, "message": "Artifacts not found"})
        return

    try:
        with path.open("rb") as f:
            artifacts = pickle.load(f)

        problem = artifacts["problem"]
        routes = artifacts["result"]["solution"]["routes"]

        _warm_route_segment_cache(problem, routes)
    except Exception as exc:
        outbound_queue.put({"type": "monitor_error", "run_id": run_id, "message": str(exc)})
        outbound_queue.put({"type": "sim_complete", "run_id": run_id, "replay_slot": replay_slot, "cancelled": True})
        return

    speed_x = max(float(hours_per_real_second), 0.1)
    dataset_type = _detect_dataset_type(problem)
    is_solomon = dataset_type.startswith("solomon")
    start_minutes = float(problem.depot.ready_time) if is_solomon else float(VRP_CONFIG.get("time_window_start", 480))

    vehicles: List[_VehicleReplayer] = []
    for vidx, route in enumerate(routes):
        if not route:
            continue
        vehicles.append(
            _VehicleReplayer(
                problem=problem,
                vehicle_id=vidx + 1,
                route=route,
                run_id=run_id,
                replay_slot=replay_slot,
                start_minutes=start_minutes,
                is_solomon=is_solomon,
            )
        )

    last_event_t: Optional[float] = None

    while vehicles:
        if stop_event is not None and stop_event.is_set():
            outbound_queue.put(
                {"type": "sim_complete", "run_id": run_id, "replay_slot": replay_slot, "cancelled": True}
            )
            return

        best: Optional[_VehicleReplayer] = None
        best_t: Optional[float] = None
        for vr in vehicles:
            t = vr.next_event_time()
            if t is None:
                continue
            if best is None or best_t is None:
                best_t = t
                best = vr
            elif t < best_t or (t == best_t and vr.vehicle_id < best.vehicle_id):
                best_t = t
                best = vr

        if best is None or best_t is None:
            break

        if last_event_t is not None:
            effective_speed_x = speed_x * (DWELL_SPEEDUP if best.post_service_time is not None else 1.0)
            sleep_s = max(0.02, (best_t - last_event_t) * 60.0 / effective_speed_x)
            elapsed_sleep = 0.0
            while elapsed_sleep < sleep_s:
                if stop_event is not None and stop_event.is_set():
                    outbound_queue.put(
                        {"type": "sim_complete", "run_id": run_id, "replay_slot": replay_slot, "cancelled": True}
                    )
                    return
                step = min(0.2, sleep_s - elapsed_sleep)
                time.sleep(step)
                elapsed_sleep += step

        last_event_t = best_t
        kind = best.emit_one_step(stop_event)

        if kind == "done":
            vehicles.remove(best)
            continue

        if stop_event is not None and stop_event.is_set():
            outbound_queue.put(
                {"type": "sim_complete", "run_id": run_id, "replay_slot": replay_slot, "cancelled": True}
            )
            return

        vehicles = [vr for vr in vehicles if not vr.done]

    outbound_queue.put({"type": "sim_complete", "run_id": run_id, "replay_slot": replay_slot, "cancelled": False})
