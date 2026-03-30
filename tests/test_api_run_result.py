from fastapi.testclient import TestClient

import backend.main as api
from backend.job_manager import JobState, get_job


def test_run_then_poll_result(monkeypatch):
    def fake_start_run_thread(job):
        job.state = JobState.COMPLETE
        job.result = {"run_id": job.run_id, "kpis": {"total_distance": 42.0}, "solution": {"routes": [[0, 1, 0]]}}

    monkeypatch.setattr(api, "start_run_thread", fake_start_run_thread)
    client = TestClient(api.app)

    run_resp = client.post(
        "/run",
        json={
            "dataset": "C101",
            "dataset_type": "solomon",
            "population_size": 20,
            "generations": 5,
        },
    )
    assert run_resp.status_code == 200
    run_id = run_resp.json()["run_id"]
    assert get_job(run_id) is not None

    result_resp = client.get(f"/result/{run_id}")
    assert result_resp.status_code == 200
    body = result_resp.json()
    assert body["status"] == "complete"
    assert body["result"]["kpis"]["total_distance"] == 42.0

