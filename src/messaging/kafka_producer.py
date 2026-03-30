"""
Kafka producer for solver progress and convergence events.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

_PRODUCER = None


def _get_producer():
    global _PRODUCER
    if _PRODUCER is not None:
        return _PRODUCER
    try:
        from kafka import KafkaProducer

        _PRODUCER = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    except Exception:
        _PRODUCER = False
    return _PRODUCER


def _send(topic: str, payload: Dict[str, Any]) -> None:
    producer = _get_producer()
    if not producer:
        return
    try:
        producer.send(topic, payload)
    except Exception:
        return


def emit_convergence(run_id: str, generation: int, best_fitness: float, avg_fitness: float) -> None:
    _send(
        "vrptw-convergence-log",
        {
            "run_id": run_id,
            "generation": generation,
            "best_fitness": float(best_fitness),
            "avg_fitness": float(avg_fitness),
        },
    )


def emit_solver_progress(run_id: str, message: str) -> None:
    _send("vrptw-convergence-log", {"kind": "solver_progress", "run_id": run_id, "message": message})

