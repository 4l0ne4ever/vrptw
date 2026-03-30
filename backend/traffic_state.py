"""
In-memory traffic state store.

Traffic factors are applied during replay:
- Base model: `instance_factor` (fixed) or `adaptive` (time-of-day).
- Manual injections: override base factor only within a [from_h, to_h] time window.
"""

from __future__ import annotations

import threading
import math
from dataclasses import dataclass
from typing import Dict

from config import VRP_CONFIG


@dataclass
class TrafficObservation:
    factor: float
    source: str
    model_key: str


@dataclass
class ManualOverride:
    factor: float
    source: str
    label: str
    from_h: float
    to_h: float


_lock = threading.Lock()

_base_model_key: Dict[str, str] = {}
_instance_factor: Dict[str, float] = {}
_anchor_minutes: Dict[str, float] = {}

_manual_override: Dict[str, ManualOverride] = {}
_delay_ema_minutes: Dict[str, float] = {}
_violation_score: Dict[str, float] = {}


def _normalize_model_key(model_key: str) -> str:
    key = str(model_key).strip().lower()
    if key in {"instance_factor", "adaptive"}:
        return key
    return "instance_factor"


def _adaptive_traffic_factor(
    time_minutes: float,
    *,
    traffic_factor_peak: float,
    traffic_factor_normal: float,
    traffic_factor_low: float,
    delay_ema_minutes: float,
    violation_score: float,
) -> float:
    """
    Continuous adaptive factor:
    - smooth time-of-day baseline (not stepwise constants),
    - short-period micro fluctuation for real-time variation,
    - runtime feedback from delay/violations.
    """
    t = float(time_minutes % 1440.0)

    # Smooth baseline from low -> normal -> peak using Gaussian bumps around rush hours.
    peak_morning = math.exp(-((t - 8.0 * 60.0) ** 2) / (2.0 * (75.0 ** 2)))
    peak_evening = math.exp(-((t - 17.5 * 60.0) ** 2) / (2.0 * (90.0 ** 2)))
    peak_mix = min(1.0, peak_morning + peak_evening)

    # Night valley around ~2:00.
    night_valley = math.exp(-((t - 2.0 * 60.0) ** 2) / (2.0 * (120.0 ** 2)))

    baseline = traffic_factor_normal + (traffic_factor_peak - traffic_factor_normal) * peak_mix
    baseline -= (traffic_factor_normal - traffic_factor_low) * night_valley

    # Real-time fluctuation every ~30 minutes to avoid constant plateaus on short replays.
    micro_wave = 0.04 * math.sin((2.0 * math.pi * t) / 30.0)

    # Runtime feedback: more delay/violations -> slightly heavier traffic.
    delay_boost = min(0.25, max(0.0, delay_ema_minutes) / 120.0)
    violation_boost = min(0.20, max(0.0, violation_score) * 0.02)

    return baseline + micro_wave + delay_boost + violation_boost


def apply_model_key(run_id: str, model_key: str, instance_factor: float = 1.0, *, anchor_minutes: float = 0.0) -> None:
    """
    Set base traffic model for a run.

    `anchor_minutes` is used to convert sim_time_h -> time-of-day minutes.
    """
    normalized = _normalize_model_key(model_key)
    with _lock:
        _base_model_key[run_id] = normalized
        _instance_factor[run_id] = float(instance_factor)
        _anchor_minutes[run_id] = float(anchor_minutes)
        _delay_ema_minutes[run_id] = 0.0
        _violation_score[run_id] = 0.0


def inject_event(
    run_id: str,
    factor: float,
    source: str,
    *,
    from_h: float = 0.0,
    to_h: float = 1e6,
    label: str = "manual_override",
) -> None:
    with _lock:
        _manual_override[run_id] = ManualOverride(
            factor=float(factor),
            source=str(source),
            label=str(label),
            from_h=float(from_h),
            to_h=float(to_h),
        )


def record_runtime_signal(run_id: str, *, delay_minutes: float = 0.0, violation: bool = False) -> None:
    """
    Feed real-time signals from replay so adaptive mode responds to current conditions.
    """
    with _lock:
        prev_delay = float(_delay_ema_minutes.get(run_id, 0.0))
        alpha = 0.2
        _delay_ema_minutes[run_id] = (1.0 - alpha) * prev_delay + alpha * max(0.0, float(delay_minutes))

        prev_viol = float(_violation_score.get(run_id, 0.0))
        decayed = prev_viol * 0.95
        _violation_score[run_id] = decayed + (1.0 if violation else 0.0)


def get_current_observation(run_id: str, sim_time_h: float = 0.0) -> TrafficObservation:
    with _lock:
        base_key = _base_model_key.get(run_id, "instance_factor")
        instance_factor = _instance_factor.get(run_id, 1.0)
        anchor_minutes = _anchor_minutes.get(run_id, 0.0)
        manual = _manual_override.get(run_id)
        delay_ema = _delay_ema_minutes.get(run_id, 0.0)
        violation_score = _violation_score.get(run_id, 0.0)

    time_minutes = anchor_minutes + float(sim_time_h) * 60.0

    if base_key == "adaptive":
        factor = _adaptive_traffic_factor(
            time_minutes,
            traffic_factor_peak=float(VRP_CONFIG.get("traffic_factor_peak", 1.8)),
            traffic_factor_normal=float(VRP_CONFIG.get("traffic_factor_normal", 1.2)),
            traffic_factor_low=float(VRP_CONFIG.get("traffic_factor_low", 1.0)),
            delay_ema_minutes=float(delay_ema),
            violation_score=float(violation_score),
        )
    else:
        factor = instance_factor

    # Apply manual override only inside its active window.
    if manual is not None and manual.from_h <= sim_time_h <= manual.to_h:
        return TrafficObservation(factor=manual.factor, source=manual.source, model_key="manual_override")

    return TrafficObservation(factor=max(0.1, float(factor)), source=base_key, model_key=base_key)

