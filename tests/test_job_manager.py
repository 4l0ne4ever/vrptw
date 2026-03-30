import time
from pathlib import Path

import backend.job_manager as jm


class DummyProblem:
    pass


def test_job_manager_state_transitions(monkeypatch, tmp_path):
    monkeypatch.setattr(jm, "_RUNS_DIR", Path(tmp_path))
    jm._RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def fake_build_problem(_request):
        return DummyProblem()

    def fake_run_ga_with_events(job, _problem, _ga_config):
        return {
            "kpis": {"total_distance": 10.0},
            "solution": {"routes": [[0, 1, 0]], "chromosome": [1], "fitness": 1.0, "total_distance": 10.0},
            "evolution": [{"generation": 1, "best_fitness": 1.0, "avg_fitness": 1.2}],
        }

    monkeypatch.setattr(jm, "_build_problem", fake_build_problem)
    monkeypatch.setattr(jm, "_run_ga_with_events", fake_run_ga_with_events)
    monkeypatch.setattr(jm, "_serialize_artifacts", lambda *_args, **_kwargs: None)

    job = jm.create_job({"dataset": "C101", "dataset_type": "solomon"})
    assert job.state == jm.JobState.PENDING
    assert jm.get_job(job.run_id) is not None
    assert any(x.run_id == job.run_id for x in jm.list_jobs())

    jm.start_run_thread(job)

    deadline = time.time() + 5.0
    while time.time() < deadline:
        current = jm.get_job(job.run_id)
        if current and current.state in (jm.JobState.COMPLETE, jm.JobState.ERROR):
            break
        time.sleep(0.05)

    current = jm.get_job(job.run_id)
    assert current is not None
    assert current.state == jm.JobState.COMPLETE
    assert current.result is not None
    assert current.result["solution"]["routes"] == [[0, 1, 0]]

