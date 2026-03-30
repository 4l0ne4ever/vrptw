"""
FastAPI entrypoint for fullstack VRPTW runtime.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import VRP_CONFIG

from backend.kafka_bridge import start_kafka_forwarder
from backend.job_manager import (
    JobState,
    build_quick_comparison,
    create_job,
    get_job,
    get_run_progress,
    start_monitor_replay_thread,
    start_run_thread,
    stop_monitor_replay,
    try_begin_replan,
)
from backend.monitor_context import build_monitor_context
from backend.realtime import outbound_queue
from backend.traffic_state import inject_event
from src.messaging.kafka_telemetry import emit_traffic_update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VRPTW Fullstack API", version="0.1.0")


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)

    async def broadcast_json(self, payload: Dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.send_json(payload)
            except Exception:
                await self.disconnect(ws)


manager = ConnectionManager()
_pump_task: Optional[asyncio.Task] = None


class RunRequest(BaseModel):
    dataset: str
    dataset_type: Optional[str] = "test"
    traffic_factor: float = Field(default=VRP_CONFIG.get("traffic_factor", 1.0), ge=0.1, le=20.0)
    traffic_model: str = Field(default="adaptive")
    population_size: int = Field(default=100, ge=2)
    generations: int = Field(default=1000, ge=1)
    crossover_prob: float = Field(default=0.9, ge=0.0, le=1.0)
    mutation_prob: float = Field(default=0.15, ge=0.0, le=1.0)
    tournament_size: int = Field(default=5, ge=2)
    elitism_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    use_split_algorithm: bool = True
    seed: Optional[int] = None
    time_limit: Optional[float] = None


class MonitorStartRequest(BaseModel):
    run_id: str
    replay_slot: int = 0
    hours_per_real_second: float = 1.0


class MonitorStopRequest(BaseModel):
    run_id: str


class ReplanRequest(BaseModel):
    run_id: str
    replay_slot: int = 0
    sim_time_h: float = 0.0


class QuickCompareRequest(BaseModel):
    run_id: str


class TrafficInjectRequest(BaseModel):
    run_id: str
    factor: float = Field(ge=0.1, le=5.0)
    from_h: float = Field(ge=0.0, le=1e6, default=0.0)
    to_h: float = Field(ge=0.0, le=1e6, default=1e6)
    label: str = "manual_override"
    source: str = "manual_override"


def _to_jsonable(value: Any) -> Any:
    # FastAPI/Pydantic may fail on numpy scalar types; normalize recursively.
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_jsonable(v) for v in value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value


_TEST_DATASETS_DIR = Path("data/test_datasets")


def _list_instances() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not _TEST_DATASETS_DIR.exists():
        return items
    for p in sorted(_TEST_DATASETS_DIR.glob("*.json")):
        items.append({"key": p.stem, "dataset": p.stem, "dataset_type": "test"})
    return items


def _to_vrp_json(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    cols = {c.lower().strip(): c for c in df.columns}
    for req in ["id", "x", "y", "demand"]:
        if req not in cols:
            raise HTTPException(status_code=400, detail=f"Missing column: {req}")
    out = df.rename(columns={v: k for k, v in cols.items()})
    depot_rows = out[out["id"] == 0]
    if depot_rows.empty:
        raise HTTPException(status_code=400, detail="Missing depot row id=0")
    depot = depot_rows.iloc[0]
    customers_df = out[out["id"] != 0]
    customers = []
    for _, row in customers_df.iterrows():
        customers.append(
            {
                "id": int(row["id"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "demand": float(row["demand"]),
                "ready_time": float(row.get("ready_time", 0.0)),
                "due_date": float(row.get("due_date", 1000.0)),
                "service_time": float(row.get("service_time", 10.0)),
            }
        )
    vehicle_capacity = float(out["vehicle_capacity"].iloc[0]) if "vehicle_capacity" in out else 200.0
    num_vehicles = int(out["num_vehicles"].iloc[0]) if "num_vehicles" in out else max(1, len(customers) // 8)
    return {
        "depot": {
            "id": 0,
            "x": float(depot["x"]),
            "y": float(depot["y"]),
            "demand": 0.0,
            "ready_time": float(depot.get("ready_time", 0.0)),
            "due_date": float(depot.get("due_date", 1000.0)),
            "service_time": 0.0,
        },
        "customers": customers,
        "vehicle_capacity": vehicle_capacity,
        "num_vehicles": num_vehicles,
        "metadata": {"name": name, "dataset_type": "upload"},
        "problem_config": {"vehicle_capacity": vehicle_capacity, "num_vehicles": num_vehicles},
    }


async def _pump_outbound_queue() -> None:
    while True:
        item = await asyncio.to_thread(outbound_queue.get)
        await manager.broadcast_json(item)


@app.on_event("startup")
async def startup_event() -> None:
    global _pump_task
    start_kafka_forwarder()
    _pump_task = asyncio.create_task(_pump_outbound_queue())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _pump_task
    if _pump_task is not None:
        _pump_task.cancel()
        _pump_task = None


origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/instances")
def instances() -> Dict[str, Any]:
    return {"items": _list_instances()}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename or "").suffix.lower()
        stem = Path(file.filename or "uploaded").stem
        if ext not in {".json", ".csv", ".xlsx", ".xls"}:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        raw = await file.read()

        if ext == ".json":
            payload = json.loads(raw.decode("utf-8"))
        elif ext == ".csv":
            payload = _to_vrp_json(pd.read_csv(io.BytesIO(raw)), stem)
        else:
            payload = _to_vrp_json(pd.read_excel(io.BytesIO(raw)), stem)

        saved = uploads_dir / f"{stem}.json"
        with saved.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {
            "dataset_id": saved.stem,
            "dataset_type": "upload",
            "num_customers": len(payload.get("customers", [])),
            "metadata": payload.get("metadata", {"name": saved.stem}),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload processing failed: {exc}") from exc


@app.post("/run")
def run_solver(req: RunRequest) -> Dict[str, str]:
    try:
        job = create_job(req.model_dump())
        start_run_thread(job)
        return {"run_id": job.run_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start run: {exc}") from exc


@app.get("/result/{run_id}")
def result(run_id: str) -> Dict[str, Any]:
    job = get_job(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Run not found")
    if job.state in (JobState.PENDING, JobState.RUNNING):
        return {"status": "running", "run_id": run_id}
    if job.state == JobState.ERROR:
        raise HTTPException(status_code=500, detail=job.error or "Run failed")
    return {"status": "complete", "run_id": run_id, "result": _to_jsonable(job.result)}


@app.get("/progress/{run_id}")
def progress(run_id: str) -> Dict[str, Any]:
    payload = get_run_progress(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found")
    return _to_jsonable(payload)


@app.post("/monitor/start")
def monitor_start(req: MonitorStartRequest) -> Dict[str, Any]:
    try:
        start_monitor_replay_thread(req.run_id, req.replay_slot, req.hours_per_real_second)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/monitor/stop")
def monitor_stop(req: MonitorStopRequest) -> Dict[str, Any]:
    try:
        stop_monitor_replay(req.run_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitor: {exc}") from exc


@app.get("/monitor/context")
def monitor_context(run_id: str, replay_slot: int = 0) -> Dict[str, Any]:
    job = get_job(run_id)
    if not job or not job.run_dir:
        raise HTTPException(status_code=404, detail="Run not found")
    artifact_path = Path(job.run_dir) / "artifacts.pkl"
    try:
        return build_monitor_context(str(artifact_path), replay_slot=replay_slot)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/monitor/replan")
def monitor_replan(req: ReplanRequest) -> Dict[str, Any]:
    try:
        try_begin_replan(req.run_id, req.replay_slot, req.sim_time_h)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/compare/quick")
def compare_quick(req: QuickCompareRequest) -> Dict[str, Any]:
    try:
        payload = build_quick_comparison(req.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run quick comparison: {exc}") from exc
    return {"ok": True, "comparison": _to_jsonable(payload)}


@app.post("/monitor/traffic/inject")
def monitor_traffic_inject(req: TrafficInjectRequest) -> Dict[str, Any]:
    try:
        inject_event(
            req.run_id,
            req.factor,
            req.source,
            from_h=req.from_h,
            to_h=req.to_h,
            label=req.label,
        )
        payload = {
            "run_id": req.run_id,
            "factor": req.factor,
            "source": req.source,
            "from_h": req.from_h,
            "to_h": req.to_h,
            "label": req.label,
        }
        emit_traffic_update(payload)
        outbound_queue.put({"type": "traffic_update", **payload})
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to inject traffic: {exc}") from exc


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(ws)


frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    # Optional production mode: serve built SPA from backend.
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

