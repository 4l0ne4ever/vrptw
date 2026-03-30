"""
Kafka consumer bridge that forwards topic events to outbound_queue.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from backend.realtime import outbound_queue

logger = logging.getLogger(__name__)

TOPICS = (
    "vrptw-convergence-log",
    "vrptw-tw-alerts",
    "vrptw-replan-events",
    "vrptw-traffic-updates",
)


def _normalize_message(topic: str, value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if topic == "vrptw-convergence-log":
        if value.get("kind") == "solver_progress":
            return {
                "type": "solver_progress",
                "run_id": value.get("run_id"),
                "message": value.get("message", ""),
            }
        return {
            "type": "convergence",
            "run_id": value.get("run_id"),
            "generation": value.get("generation"),
            "best_fitness": value.get("best_fitness"),
            "avg_fitness": value.get("avg_fitness"),
        }
    if topic == "vrptw-tw-alerts":
        return {"type": "alert", "run_id": value.get("run_id"), "data": value}
    if topic == "vrptw-replan-events":
        return {"type": value.get("type", "replan_event"), **value}
    if topic == "vrptw-traffic-updates":
        return {"type": "traffic_update", **value}
    return None


def start_kafka_forwarder() -> Optional[threading.Thread]:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    group_id = os.getenv("KAFKA_CONSUMER_GROUP", "vrptw-fastapi")
    try:
        from kafka import KafkaConsumer
    except Exception as exc:
        logger.warning("Kafka consumer dependency unavailable: %s", exc)
        return None

    def _loop() -> None:
        while True:
            consumer = None
            try:
                consumer = KafkaConsumer(
                    *TOPICS,
                    bootstrap_servers=bootstrap,
                    group_id=group_id,
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                )
                logger.info("Kafka forwarder connected: %s", bootstrap)
                for msg in consumer:
                    normalized = _normalize_message(msg.topic, msg.value)
                    if normalized is not None:
                        outbound_queue.put(normalized)
            except Exception as exc:
                logger.warning("Kafka forwarder unavailable (%s), retrying...", exc)
                time.sleep(2.0)
            finally:
                if consumer is not None:
                    try:
                        consumer.close()
                    except Exception:
                        pass

    t = threading.Thread(target=_loop, name="kafka-forwarder", daemon=True)
    t.start()
    return t

