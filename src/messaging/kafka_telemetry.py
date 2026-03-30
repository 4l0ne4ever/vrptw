"""Kafka producer for telemetry and alert events."""

from __future__ import annotations

from typing import Any, Dict

from src.messaging.kafka_producer import _send


def emit_telemetry(payload: Dict[str, Any]) -> None:
    try:
        from backend.realtime import outbound_queue

        outbound_queue.put({"type": "telemetry", **payload})
    except Exception:
        pass
    _send("vrptw-telemetry", payload)


def emit_alert(payload: Dict[str, Any]) -> None:
    _send("vrptw-tw-alerts", payload)


def emit_replan_event(payload: Dict[str, Any]) -> None:
    _send("vrptw-replan-events", payload)


def emit_traffic_update(payload: Dict[str, Any]) -> None:
    _send("vrptw-traffic-updates", payload)

