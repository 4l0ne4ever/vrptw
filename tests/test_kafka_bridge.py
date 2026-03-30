from backend.kafka_bridge import _normalize_message


def test_normalize_convergence_and_progress():
    progress = _normalize_message(
        "vrptw-convergence-log",
        {"kind": "solver_progress", "run_id": "r1", "message": "step"},
    )
    assert progress == {"type": "solver_progress", "run_id": "r1", "message": "step"}

    convergence = _normalize_message(
        "vrptw-convergence-log",
        {"run_id": "r1", "generation": 2, "best_fitness": 10.0, "avg_fitness": 12.0},
    )
    assert convergence == {
        "type": "convergence",
        "run_id": "r1",
        "generation": 2,
        "best_fitness": 10.0,
        "avg_fitness": 12.0,
    }


def test_normalize_other_topics():
    assert _normalize_message("vrptw-telemetry", {"run_id": "r1", "vehicle_id": 1}) is None

    alert = _normalize_message("vrptw-tw-alerts", {"run_id": "r1", "type": "tw_violation"})
    assert alert["type"] == "alert"
    assert alert["data"]["type"] == "tw_violation"

    replan = _normalize_message("vrptw-replan-events", {"type": "replan_complete", "run_id": "r1"})
    assert replan["type"] == "replan_complete"

    traffic = _normalize_message("vrptw-traffic-updates", {"run_id": "r1", "factor": 1.2})
    assert traffic["type"] == "traffic_update"
    assert traffic["factor"] == 1.2

    assert _normalize_message("unknown-topic", {"x": 1}) is None

