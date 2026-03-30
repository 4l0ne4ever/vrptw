"""
Thread-safe job manager for FastAPI runtime.
"""

from __future__ import annotations

import os
import pickle
import threading
import traceback
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from config import VRP_CONFIG

from backend.realtime import outbound_queue
from backend.traffic_state import apply_model_key
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.nearest_neighbor import NearestNeighborHeuristic
from src.data_processing.json_loader import JSONDatasetLoader
from src.data_processing.distance import DistanceCalculator
from src.evaluation.metrics import KPICalculator
from src.models.vrp_model import VRPProblem, create_vrp_problem_from_dict
from src.messaging.kafka_producer import emit_convergence, emit_solver_progress
from src.messaging.kafka_telemetry import emit_replan_event
from src.simulation.replay import replay_solution


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Job:
    run_id: str
    state: JobState = JobState.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    request: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    baseline_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    run_dir: Optional[str] = None
    thread: Optional[threading.Thread] = None
    replay_thread: Optional[threading.Thread] = None
    replay_stop_event: Optional[threading.Event] = None
    replan_in_progress: bool = False
    plan_revision: int = 0
    last_replan_at: float = 0.0
    live_convergence: List[Dict[str, Any]] = field(default_factory=list)
    solver_progress_message: str = ""


_jobs: Dict[str, Job] = {}
_lock = threading.Lock()

_RUNS_DIR = Path(os.getenv("VRPTW_RUNS_DIR", "results/runs"))
_RUNS_DIR.mkdir(parents=True, exist_ok=True)
_REPLAN_COOLDOWN_S = int(os.getenv("REPLAN_COOLDOWN_S", "120"))
_QUICK_COMPARE_MIN_GENS = 80
_QUICK_COMPARE_MAX_GENS = 300
_QUICK_COMPARE_MIN_POP = 40
_QUICK_COMPARE_MAX_POP = 120
_QUICK_COMPARE_GEN_RATIO = 0.35
_QUICK_COMPARE_POP_RATIO = 0.6


def create_job(request: Dict[str, Any]) -> Job:
    run_id = str(uuid4())
    run_dir = _RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    job = Job(run_id=run_id, request=request, run_dir=str(run_dir))
    with _lock:
        _jobs[run_id] = job
    return job


def get_job(run_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(run_id)


def list_jobs() -> List[Job]:
    with _lock:
        return list(_jobs.values())


def _build_problem(request: Dict[str, Any]) -> VRPProblem:
    dataset_name = request["dataset"]
    dataset_type = request.get("dataset_type")
    traffic_factor = float(request.get("traffic_factor", 1.0))
    traffic_model = str(request.get("traffic_model", "adaptive")).strip().lower()
    is_solomon = str(dataset_type).strip().lower().startswith("solomon")
    use_adaptive_traffic = (traffic_model == "adaptive") and not is_solomon
    if dataset_type == "upload":
        upload_file = Path("data/uploads") / f"{dataset_name}.json"
        if not upload_file.exists():
            raise FileNotFoundError(f"Uploaded dataset not found: {dataset_name}")
        with upload_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        coords = [(data["depot"]["x"], data["depot"]["y"])]
        coords.extend((c["x"], c["y"]) for c in data["customers"])
        calculator = DistanceCalculator(
            traffic_factor=traffic_factor,
            use_adaptive=use_adaptive_traffic,
            dataset_type="hanoi",
        )
        distance_matrix = calculator.calculate_distance_matrix(coords)
        problem = create_vrp_problem_from_dict(data, distance_matrix, use_adaptive_traffic=use_adaptive_traffic)
        problem.set_distance_calculator(calculator)
        return problem
    loader = JSONDatasetLoader()
    data = loader.load_dataset(dataset_name, dataset_type)
    coords = [(data["depot"]["x"], data["depot"]["y"])]
    coords.extend((c["x"], c["y"]) for c in data["customers"])

    calculator = DistanceCalculator(
        traffic_factor=traffic_factor,
        use_adaptive=use_adaptive_traffic,
        dataset_type=str(dataset_type or "hanoi"),
    )
    distance_matrix = calculator.calculate_distance_matrix(coords)
    problem = create_vrp_problem_from_dict(data, distance_matrix, use_adaptive_traffic=use_adaptive_traffic)
    problem.set_distance_calculator(calculator)
    return problem


def _serialize_artifacts(job: Job, payload: Dict[str, Any]) -> None:
    if not job.run_dir:
        return
    artifact_path = Path(job.run_dir) / "artifacts.pkl"
    with artifact_path.open("wb") as f:
        pickle.dump(payload, f)


def _load_artifacts(job: Job) -> Dict[str, Any]:
    if not job.run_dir:
        raise ValueError("Run has no artifact directory")
    artifact_path = Path(job.run_dir) / "artifacts.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError("Artifacts file not found")
    with artifact_path.open("rb") as f:
        return pickle.load(f)


def _run_ga_with_events(job: Job, problem: VRPProblem, ga_config: Dict[str, Any]) -> Dict[str, Any]:
    req = job.request or {}
    raw_seed = req.get("seed")
    if raw_seed is not None:
        s = int(raw_seed)
        random.seed(s)
        np.random.seed(s)

    wall_limit_s: Optional[float] = None
    wall_raw = req.get("time_limit")
    if wall_raw is not None:
        try:
            w = float(wall_raw)
            if w > 0:
                wall_limit_s = w
        except (TypeError, ValueError):
            pass
    t0 = time.monotonic()

    ga = GeneticAlgorithm(problem, ga_config)
    ga.initialize_population()

    evolution_data: List[Dict[str, Any]] = []
    max_generations = ga_config["generations"]
    for generation in range(max_generations):
        if wall_limit_s is not None and (time.monotonic() - t0) >= wall_limit_s:
            break
        ga._create_next_generation()
        ga._update_statistics()
        best = ga.population.get_best_individual()
        avg_fit = ga.population.get_avg_fitness()
        emit_convergence(job.run_id, generation + 1, best.fitness, avg_fit)
        if generation % 10 == 0:
            progress_msg = f"Generation {generation + 1}/{max_generations}"
            emit_solver_progress(job.run_id, progress_msg)
            with _lock:
                job.solver_progress_message = progress_msg
        evolution_data.append(
            {
                "generation": generation + 1,
                "best_fitness": best.fitness,
                "best_distance": best.total_distance,
                "avg_fitness": avg_fit,
            }
        )
        with _lock:
            # Keep memory bounded while preserving recent trend for UI.
            job.live_convergence = evolution_data[-600:]
        if ga._check_convergence():
            break

    best_solution = ga.population.get_best_individual()
    kpis = KPICalculator(problem).calculate_kpis(best_solution)
    time_window_details: List[Dict[str, Any]] = []
    try:
        # Compute per-stop time window compliance for frontend display.
        meta = getattr(problem, "metadata", {}) or {}
        dt = str(meta.get("dataset_type") or getattr(problem, "dataset_type", "hanoi")).strip().lower()
        is_solomon = dt.startswith("solomon")
        from src.optimization.matrix_preprocessor import MatrixPreprocessor
        from src.optimization.vidal_evaluator import VidalEvaluator

        pre = MatrixPreprocessor(problem)
        _dist_matrix, time_matrix = pre.normalize_matrices()
        evaluator = VidalEvaluator(problem, time_matrix, _dist_matrix)

        start_minutes = float(problem.depot.ready_time) if is_solomon else float(VRP_CONFIG.get("time_window_start", 480))

        for route_idx, route in enumerate(best_solution.routes):
            if not route or len(route) < 3:
                continue
            current_time = start_minutes
            stops: List[Dict[str, Any]] = []
            for pos in range(1, len(route) - 1):
                prev_node = int(route[pos - 1])
                cust_id = int(route[pos])
                customer = problem.get_customer_by_id(cust_id)
                if not customer:
                    continue

                travel = float(evaluator._get_time(prev_node, cust_id))
                arrival = current_time + travel

                ready = float(customer.ready_time)
                due = float(customer.due_date)
                start_service = max(arrival, ready)

                violated = False
                violation_type = None
                lateness_minutes = 0.0

                if is_solomon:
                    if arrival < ready:
                        violated = True
                        violation_type = "early"
                        lateness_minutes = ready - arrival
                    elif arrival > due:
                        violated = True
                        violation_type = "late"
                        lateness_minutes = arrival - due
                else:
                    if start_service > due:
                        violated = True
                        violation_type = "late"
                        lateness_minutes = start_service - due

                stops.append(
                    {
                        "customer_id": cust_id,
                        "position": pos,
                        "arrival_h": arrival / 60.0,
                        "start_service_h": start_service / 60.0,
                        "ready_time": ready,
                        "due_date": due,
                        "violated": violated,
                        "violation_type": violation_type,
                        "lateness_minutes": lateness_minutes,
                    }
                )

                current_time = start_service + float(customer.service_time)

            time_window_details.append({"vehicle_id": route_idx + 1, "stops": stops})
    except Exception:
        time_window_details = []

    return {
        "kpis": kpis,
        "solution": {
            "routes": best_solution.routes,
            "chromosome": best_solution.chromosome,
            "fitness": best_solution.fitness,
            "total_distance": best_solution.total_distance,
            "penalty": getattr(best_solution, "penalty", 0.0),
            "time_window_details": time_window_details,
        },
        "evolution": evolution_data,
    }


def _run_with_seed(seed: Optional[int], fn):
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        if seed is not None:
            s = int(seed)
            random.seed(s)
            np.random.seed(s)
        return fn()
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _quick_compare_config(request: Dict[str, Any], *, use_split_algorithm: bool) -> Dict[str, Any]:
    base_gens = int(request.get("generations", 1000))
    base_pop = int(request.get("population_size", 100))
    quick_gens = int(base_gens * _QUICK_COMPARE_GEN_RATIO)
    quick_pop = int(base_pop * _QUICK_COMPARE_POP_RATIO)
    quick_gens = max(_QUICK_COMPARE_MIN_GENS, min(_QUICK_COMPARE_MAX_GENS, quick_gens))
    quick_pop = max(_QUICK_COMPARE_MIN_POP, min(_QUICK_COMPARE_MAX_POP, quick_pop))
    return {
        "population_size": quick_pop,
        "generations": quick_gens,
        "crossover_prob": float(request.get("crossover_prob", 0.9)),
        "mutation_prob": float(request.get("mutation_prob", 0.15)),
        "tournament_size": int(request.get("tournament_size", 5)),
        "elitism_rate": float(request.get("elitism_rate", 0.1)),
        "use_split_algorithm": bool(use_split_algorithm),
    }


def _summarize_kpis(label: str, kpis: Dict[str, Any], runtime_s: float) -> Dict[str, Any]:
    cv = kpis.get("constraint_violations", {}) or {}
    return {
        "label": label,
        "total_distance": float(kpis.get("total_distance", 0.0) or 0.0),
        "total_cost": float(kpis.get("total_cost", 0.0) or 0.0),
        "num_routes": int(kpis.get("num_routes", 0) or 0),
        "is_feasible": bool(kpis.get("is_feasible", False)),
        "tw_violations": int(cv.get("time_window_violations", 0) or 0),
        "runtime_s": float(runtime_s),
    }


def _summarize_result(label: str, result: Optional[Dict[str, Any]], runtime_s: float = 0.0) -> Optional[Dict[str, Any]]:
    if not result:
        return None
    return _summarize_kpis(label, result.get("kpis", {}) or {}, runtime_s=runtime_s)


def _run_quick_ga(problem: VRPProblem, request: Dict[str, Any], *, use_split_algorithm: bool) -> Dict[str, Any]:
    cfg = _quick_compare_config(request, use_split_algorithm=use_split_algorithm)
    t0 = time.monotonic()
    ga = GeneticAlgorithm(problem, cfg)
    ga.initialize_population()
    for _ in range(cfg["generations"]):
        ga._create_next_generation()
        ga._update_statistics()
        if ga._check_convergence():
            break
    best = ga.population.get_best_individual()
    kpis = KPICalculator(problem).calculate_kpis(best)
    return _summarize_kpis(
        "HGA" if use_split_algorithm else "GA",
        kpis,
        runtime_s=time.monotonic() - t0,
    )


def _run_quick_nn(problem: VRPProblem) -> Dict[str, Any]:
    t0 = time.monotonic()
    nn = NearestNeighborHeuristic(problem).solve(problem.num_vehicles)
    kpis = KPICalculator(problem).calculate_kpis(nn)
    return _summarize_kpis("NN", kpis, runtime_s=time.monotonic() - t0)


def _run_quick_replan_like(problem: VRPProblem, request: Dict[str, Any]) -> Dict[str, Any]:
    # Mimic monitor re-plan style: shorter GA run while keeping the same operators.
    cfg = _quick_compare_config(
        request,
        use_split_algorithm=bool(request.get("use_split_algorithm", True)),
    )
    cfg["generations"] = max(40, int(cfg["generations"] * 0.5))
    t0 = time.monotonic()
    ga = GeneticAlgorithm(problem, cfg)
    ga.initialize_population()
    for _ in range(cfg["generations"]):
        ga._create_next_generation()
        ga._update_statistics()
        if ga._check_convergence():
            break
    best = ga.population.get_best_individual()
    kpis = KPICalculator(problem).calculate_kpis(best)
    return _summarize_kpis("Co re-plan (mau nhanh)", kpis, runtime_s=time.monotonic() - t0)


def build_quick_comparison(run_id: str) -> Dict[str, Any]:
    job = get_job(run_id)
    if not job:
        raise ValueError("Run not found")
    if job.state != JobState.COMPLETE:
        raise ValueError("Run must be complete before quick comparison")

    request = dict(job.request or {})
    seed = request.get("seed")
    with _lock:
        baseline_result = dict(job.baseline_result) if job.baseline_result else None
        current_result = dict(job.result) if job.result else None
        plan_revision = int(job.plan_revision)
        replan_running = bool(job.replan_in_progress)

    base_problem = _build_problem(request)
    algo_rows: List[Dict[str, Any]] = []
    algo_rows.append(_run_with_seed(seed, lambda: _run_quick_ga(base_problem, request, use_split_algorithm=True)))
    algo_rows.append(_run_with_seed(seed, lambda: _run_quick_ga(base_problem, request, use_split_algorithm=False)))
    algo_rows.append(_run_with_seed(seed, lambda: _run_quick_nn(base_problem)))

    request_dynamic = dict(request)
    request_dynamic["traffic_model"] = "adaptive"
    request_static = dict(request)
    request_static["traffic_model"] = "instance_factor"
    dynamic_problem = _build_problem(request_dynamic)
    static_problem = _build_problem(request_static)
    traffic_rows = [
        _run_with_seed(seed, lambda: _run_quick_ga(dynamic_problem, request_dynamic, use_split_algorithm=True)),
        _run_with_seed(seed, lambda: _run_quick_ga(static_problem, request_static, use_split_algorithm=True)),
    ]
    traffic_rows[0]["label"] = "HGA + traffic dong"
    traffic_rows[1]["label"] = "HGA + traffic tinh"
    dynamic_replan_row = _run_with_seed(seed, lambda: _run_quick_replan_like(dynamic_problem, request_dynamic))
    static_baseline_row = dict(traffic_rows[1])
    static_baseline_row["label"] = "Toi uu tinh (1 lan, traffic tinh)"
    monitoring_like_row = dict(dynamic_replan_row)
    monitoring_like_row["label"] = "Monitoring-like (traffic dong + re-plan)"

    replan_rows: List[Dict[str, Any]] = []
    no_replan_actual = _summarize_result("Khong re-plan (actual)", baseline_result)
    if no_replan_actual is not None:
        replan_rows.append(no_replan_actual)
    if plan_revision > 0:
        with_replan_actual = _summarize_result("Co re-plan (actual)", current_result)
        if with_replan_actual is not None:
            replan_rows.append(with_replan_actual)
    else:
        simulated_base = _run_with_seed(
            seed,
            lambda: _run_quick_ga(
                base_problem,
                request,
                use_split_algorithm=bool(request.get("use_split_algorithm", True)),
            ),
        )
        simulated_base["label"] = "Khong re-plan (mau nhanh)"
        replan_rows = [simulated_base, dynamic_replan_row]

    if len(replan_rows) < 2:
        simulated_base = _run_with_seed(
            seed,
            lambda: _run_quick_ga(
                base_problem,
                request,
                use_split_algorithm=bool(request.get("use_split_algorithm", True)),
            ),
        )
        simulated_base["label"] = "Khong re-plan (mau nhanh)"
        if not replan_rows:
            replan_rows.append(simulated_base)
        if len(replan_rows) == 1:
            replan_rows.append(dynamic_replan_row)

    monitoring_rows = [static_baseline_row, monitoring_like_row]

    return {
        "run_id": run_id,
        "algorithm_compare": algo_rows,
        "traffic_compare": traffic_rows,
        "replan_compare": {
            "rows": replan_rows,
            "has_replan": plan_revision > 0,
            "plan_revision": plan_revision,
            "replan_running": replan_running,
        },
        "monitoring_compare": {
            "rows": monitoring_rows,
            "supports_monitoring": True,
            "static_optimization": True,
        },
    }


def start_run_thread(job: Job) -> None:
    def _run() -> None:
        with _lock:
            job.state = JobState.RUNNING
            job.started_at = datetime.utcnow()
            job.live_convergence = []
            job.solver_progress_message = "Starting run"
        try:
            request = job.request
            emit_solver_progress(job.run_id, "Initializing problem")
            with _lock:
                job.solver_progress_message = "Initializing problem"
            problem = _build_problem(request)

            ga_config = {
                "population_size": int(request.get("population_size", 100)),
                "generations": int(request.get("generations", 1000)),
                "crossover_prob": float(request.get("crossover_prob", 0.9)),
                "mutation_prob": float(request.get("mutation_prob", 0.15)),
                "tournament_size": int(request.get("tournament_size", 5)),
                "elitism_rate": float(request.get("elitism_rate", 0.1)),
                "use_split_algorithm": bool(request.get("use_split_algorithm", True)),
            }

            emit_solver_progress(job.run_id, "Initializing population")
            with _lock:
                job.solver_progress_message = "Initializing population"
            ga_payload = _run_ga_with_events(job, problem, ga_config)
            result = {
                "run_id": job.run_id,
                "dataset": request.get("dataset"),
                "dataset_type": request.get("dataset_type"),
                "run_config": {
                    "traffic_model": request.get("traffic_model"),
                    "use_split_algorithm": bool(request.get("use_split_algorithm", True)),
                    "population_size": int(request.get("population_size", 100)),
                    "generations": int(request.get("generations", 1000)),
                    "time_limit": request.get("time_limit"),
                },
                "kpis": ga_payload["kpis"],
                "solution": ga_payload["solution"],
                "evolution": ga_payload["evolution"],
            }

            _serialize_artifacts(
                job,
                {
                    "run_id": job.run_id,
                    "request": request,
                    "result": result,
                    "problem": problem,
                    "plan_revision": job.plan_revision,
                },
            )

            with _lock:
                job.state = JobState.COMPLETE
                job.result = result
                job.baseline_result = result
                job.completed_at = datetime.utcnow()
                job.solver_progress_message = "Run complete"

            outbound_queue.put({"type": "run_complete", "run_id": job.run_id})
        except Exception as exc:
            with _lock:
                job.state = JobState.ERROR
                job.error = f"{exc}\n{traceback.format_exc()}"
                job.completed_at = datetime.utcnow()
                job.solver_progress_message = f"Run error: {exc}"
            outbound_queue.put({"type": "run_error", "run_id": job.run_id, "message": str(exc)})

    t = threading.Thread(target=_run, name=f"run-{job.run_id}", daemon=True)
    with _lock:
        job.thread = t
    t.start()


def get_run_progress(run_id: str) -> Optional[Dict[str, Any]]:
    job = get_job(run_id)
    if not job:
        return None
    with _lock:
        return {
            "run_id": run_id,
            "status": job.state.value,
            "solver_progress_message": job.solver_progress_message,
            "convergence": list(job.live_convergence),
        }


def start_monitor_replay_thread(run_id: str, replay_slot: int = 0, hours_per_real_second: float = 1.0) -> None:
    job = get_job(run_id)
    if not job:
        raise ValueError("Run not found")
    if job.state != JobState.COMPLETE:
        raise ValueError("Run is not complete")
    if not job.run_dir:
        raise ValueError("Run has no artifact directory")

    artifact_path = Path(job.run_dir) / "artifacts.pkl"
    traffic_model = str(job.request.get("traffic_model", "adaptive"))
    instance_factor = float(job.request.get("traffic_factor", VRP_CONFIG.get("traffic_factor", 1.0)))

    try:
        artifacts = _load_artifacts(job)
        problem = artifacts.get("problem")
        dataset_type = job.request.get("dataset_type") or getattr(problem, "dataset_type", "hanoi")
        is_solomon = str(dataset_type).strip().lower().startswith("solomon")
        anchor_minutes = float(getattr(problem, "depot", None).ready_time) if is_solomon else float(
            VRP_CONFIG.get("time_window_start", 480)
        )
    except Exception:
        anchor_minutes = float(VRP_CONFIG.get("time_window_start", 480))

    apply_model_key(run_id, traffic_model, instance_factor=instance_factor, anchor_minutes=anchor_minutes)

    prev_stop: Optional[threading.Event] = None
    prev_thread: Optional[threading.Thread] = None
    with _lock:
        prev_stop = job.replay_stop_event
        prev_thread = job.replay_thread
        if prev_stop is not None:
            prev_stop.set()
    if prev_thread is not None and prev_thread.is_alive():
        prev_thread.join(timeout=10.0)

    stop_event = threading.Event()

    def _run_replay() -> None:
        replay_solution(
            run_id=run_id,
            artifact_path=str(artifact_path),
            replay_slot=replay_slot,
            hours_per_real_second=hours_per_real_second,
            stop_event=stop_event,
        )

    t = threading.Thread(target=_run_replay, name=f"replay-{run_id}", daemon=True)
    with _lock:
        job.replay_thread = t
        job.replay_stop_event = stop_event
    t.start()


def stop_monitor_replay(run_id: str) -> None:
    job = get_job(run_id)
    if not job:
        return
    with _lock:
        if job.replay_stop_event is not None:
            job.replay_stop_event.set()
    outbound_queue.put({"type": "monitor_stopped", "run_id": run_id})


def try_begin_replan(run_id: str, replay_slot: int, sim_time_h: float) -> None:
    job = get_job(run_id)
    if not job:
        raise ValueError("Run not found")
    if job.state != JobState.COMPLETE:
        raise ValueError("Run must be complete before replan")

    with _lock:
        if job.replan_in_progress:
            raise ValueError("Replan already in progress")
        now = time.time()
        if now - job.last_replan_at < _REPLAN_COOLDOWN_S:
            remain = int(_REPLAN_COOLDOWN_S - (now - job.last_replan_at))
            raise ValueError(f"Replan cooldown active ({remain}s remaining)")
        job.replan_in_progress = True
        job.last_replan_at = now

    def _replan() -> None:
        try:
            emit_replan_event(
                {
                    "type": "replan_started",
                    "run_id": run_id,
                    "replay_slot": replay_slot,
                    "sim_time_h": sim_time_h,
                    "plan_revision": job.plan_revision + 1,
                }
            )
            outbound_queue.put(
                {
                    "type": "replan_started",
                    "run_id": run_id,
                    "replay_slot": replay_slot,
                    "sim_time_h": sim_time_h,
                    "plan_revision": job.plan_revision + 1,
                }
            )

            artifacts = _load_artifacts(job)
            problem = artifacts["problem"]
            request = job.request
            base_generations = int(request.get("generations", 1000))
            ga_config = {
                "population_size": int(request.get("population_size", 100)),
                "generations": max(50, int(base_generations * 0.4)),
                "crossover_prob": float(request.get("crossover_prob", 0.9)),
                "mutation_prob": float(request.get("mutation_prob", 0.15)),
                "tournament_size": int(request.get("tournament_size", 5)),
                "elitism_rate": float(request.get("elitism_rate", 0.1)),
                "use_split_algorithm": bool(request.get("use_split_algorithm", True)),
            }
            emit_solver_progress(run_id, "Replan in progress")
            ga_payload = _run_ga_with_events(job, problem, ga_config)
            result = {
                "run_id": run_id,
                "dataset": request.get("dataset"),
                "dataset_type": request.get("dataset_type"),
                "run_config": {
                    "traffic_model": request.get("traffic_model"),
                    "use_split_algorithm": bool(request.get("use_split_algorithm", True)),
                    "population_size": int(request.get("population_size", 100)),
                    "generations": max(50, int(base_generations * 0.4)),
                    "time_limit": request.get("time_limit"),
                },
                "kpis": ga_payload["kpis"],
                "solution": ga_payload["solution"],
                "evolution": ga_payload["evolution"],
            }

            with _lock:
                if job.baseline_result is None and job.result is not None:
                    job.baseline_result = job.result
                job.plan_revision += 1
                current_revision = job.plan_revision
                job.result = result

            _serialize_artifacts(
                job,
                {
                    "run_id": run_id,
                    "request": request,
                    "result": result,
                    "problem": problem,
                    "plan_revision": current_revision,
                },
            )

            emit_replan_event(
                {
                    "type": "replan_complete",
                    "run_id": run_id,
                    "replay_slot": replay_slot,
                    "sim_time_h": sim_time_h,
                    "plan_revision": current_revision,
                }
            )
            outbound_queue.put(
                {
                    "type": "replan_complete",
                    "run_id": run_id,
                    "replay_slot": replay_slot,
                    "sim_time_h": sim_time_h,
                    "plan_revision": current_revision,
                }
            )
        except Exception as exc:
            emit_replan_event(
                {
                    "type": "replan_error",
                    "run_id": run_id,
                    "replay_slot": replay_slot,
                    "message": str(exc),
                }
            )
            outbound_queue.put(
                {
                    "type": "replan_error",
                    "run_id": run_id,
                    "replay_slot": replay_slot,
                    "message": str(exc),
                }
            )
        finally:
            with _lock:
                job.replan_in_progress = False

    t = threading.Thread(target=_replan, name=f"replan-{run_id}", daemon=True)
    t.start()

