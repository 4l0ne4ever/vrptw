"""
Run summary component aggregates before/during/after run information.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import streamlit as st

from src.evaluation.metrics import KPICalculator
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem


CONFIG_FIELDS = [
    ("population_size", "Population"),
    ("generations", "Target Generations"),
    ("num_vehicles", "Max Vehicles"),
    ("crossover_prob", "Crossover"),
    ("mutation_prob", "Mutation"),
    ("tournament_size", "Tournament"),
    ("elitism_rate", "Elitism"),
    ("use_split_algorithm", "Split Algorithm"),
    ("penalty_weight", "Penalty Weight"),
]


def render_run_summary(
    problem: VRPProblem,
    solution: Individual,
    config: Optional[Dict],
    statistics: Optional[Dict],
    dataset_metadata: Optional[Dict] = None,
    dataset_type: str = "hanoi"
):
    """
    Render synchronized summary of everything that happened during the run.
    """
    if solution is None or solution.is_empty():
        st.info("No solution available to summarize.")
        return

    dataset_info = _build_dataset_info(problem, dataset_metadata)
    statistics = statistics or {}
    effective_config = (statistics.get("config_used") or config or {}).copy()

    kpi_calculator = KPICalculator(problem)
    kpis = kpi_calculator.calculate_kpis(
        solution,
        execution_time=statistics.get("execution_time")
    )

    st.subheader("Run Summary")

    # Before run
    st.markdown("#### Before Run")
    _render_dataset_snapshot(dataset_info)
    _render_config_snapshot(effective_config, dataset_type=dataset_type)

    # During run
    st.markdown("#### During Run")
    _render_execution_snapshot(statistics, effective_config)

    # After run
    st.markdown("#### After Run")
    _render_result_snapshot(solution, kpis)


def _build_dataset_info(problem: VRPProblem, metadata: Optional[Dict]) -> Dict:
    info = problem.get_problem_info()
    ready_times = [c.ready_time for c in problem.customers]
    due_times = [c.due_date for c in problem.customers]
    service_times = [c.service_time for c in problem.customers]
    
    # Use dataset time windows (don't override with VRP_CONFIG for Solomon)
    min_ready_time = min(ready_times) if ready_times else 0
    max_due_date = max(due_times) if due_times else 0
    
    # Only use VRP_CONFIG for Hanoi mode if time windows are default/unset
    from config import VRP_CONFIG
    if metadata and metadata.get('dataset_type', 'hanoi').lower() == 'hanoi':
        if min_ready_time == 0 and max_due_date == 1000:
            min_ready_time = VRP_CONFIG.get('time_window_start', 480)
            max_due_date = VRP_CONFIG.get('time_window_end', 1200)
    
    return {
        "name": (metadata or {}).get("name") or (metadata or {}).get("id") or "Dataset",
        "num_customers": info["num_customers"],
        "vehicle_capacity": info["vehicle_capacity"],
        "num_vehicles": info["num_vehicles"],
        "total_demand": info["total_demand"],
        "min_ready_time": min_ready_time,
        "max_due_date": max_due_date,
        "avg_service_time": sum(service_times) / len(service_times) if service_times else 0,
    }


def _render_dataset_snapshot(dataset_info: Dict):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset", dataset_info["name"])
        st.metric("Customers", dataset_info["num_customers"])
    with col2:
        st.metric("Vehicle Capacity", f"{dataset_info['vehicle_capacity']:.0f}")
        st.metric("Fleet Size", dataset_info["num_vehicles"])
    with col3:
        st.metric("Total Demand", f"{dataset_info['total_demand']:.0f}")
        # Format time window as hours:minutes
        tw_start = dataset_info['min_ready_time']
        tw_end = dataset_info['max_due_date']
        tw_start_hour = int(tw_start // 60)
        tw_start_min = int(tw_start % 60)
        tw_end_hour = int(tw_end // 60)
        tw_end_min = int(tw_end % 60)
        st.metric(
            "Time Window Range",
            f"{tw_start_hour:02d}:{tw_start_min:02d} → {tw_end_hour:02d}:{tw_end_min:02d}"
        )


def _render_config_snapshot(config: Dict, dataset_type: str = "hanoi"):
    if not config:
        st.info("No GA configuration provided.")
        return

    # GA Configuration
    rows = []
    for field, label in CONFIG_FIELDS:
        if field in config:
            rows.append((label, _format_config_value(config[field])))

    if rows:
        st.write("**GA Configuration**")
        col_count = min(3, len(rows))
        cols = st.columns(col_count)
        for idx, (label, value) in enumerate(rows):
            with cols[idx % col_count]:
                st.metric(label, value)
    
    # VRP Configuration (only show Hanoi-specific settings for Hanoi mode)
    if dataset_type.lower() == "hanoi":
        from config import VRP_CONFIG
        vrp_rows = []
        
        # Traffic configuration
        if VRP_CONFIG.get('use_adaptive_traffic'):
            vrp_rows.append(("Traffic Mode", "Adaptive"))
            vrp_rows.append(("Peak Traffic Factor", f"{VRP_CONFIG.get('traffic_factor_peak', 1.8):.2f}"))
            vrp_rows.append(("Normal Traffic Factor", f"{VRP_CONFIG.get('traffic_factor_normal', 1.2):.2f}"))
        else:
            vrp_rows.append(("Traffic Factor", f"{VRP_CONFIG.get('traffic_factor', 1.0):.2f}"))
        
        # Time window (for Hanoi, use config)
        tw_start = VRP_CONFIG.get('time_window_start', 480)
        tw_end = VRP_CONFIG.get('time_window_end', 1200)
        tw_start_hour = int(tw_start // 60)
        tw_start_min = int(tw_start % 60)
        tw_end_hour = int(tw_end // 60)
        tw_end_min = int(tw_end % 60)
        vrp_rows.append(("Time Window", f"{tw_start_hour:02d}:{tw_start_min:02d} - {tw_end_hour:02d}:{tw_end_min:02d}"))
        
        # Service time
        service_time = VRP_CONFIG.get('service_time', 12)
        vrp_rows.append(("Service Time", f"{service_time} min"))
        
        # Cost configuration
        if VRP_CONFIG.get('use_waiting_fee'):
            waiting_fee = VRP_CONFIG.get('waiting_fee_per_minute', 300)
            vrp_rows.append(("Waiting Fee", f"{waiting_fee:,} ₫/min"))
        
        cod_ratio = VRP_CONFIG.get('cod_ratio', 0.75)
        vrp_rows.append(("COD Ratio", f"{cod_ratio*100:.0f}%"))
        
        if vrp_rows:
            st.write("**VRP Configuration**")
        col_count = min(3, len(vrp_rows))
        cols = st.columns(col_count)
        for idx, (label, value) in enumerate(vrp_rows):
            with cols[idx % col_count]:
                st.metric(label, value)


def _render_execution_snapshot(statistics: Dict, config: Dict):
    planned_generations = config.get("generations")
    executed_generations = statistics.get("generations", 0)
    progress = None
    if planned_generations:
        progress = executed_generations / planned_generations

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Generations",
            executed_generations,
            delta=None if planned_generations is None else f"target {planned_generations}"
        )
    with col2:
        st.metric("Total Evaluations", statistics.get("total_evaluations", 0))
    with col3:
        st.metric("Runtime", _format_duration(statistics.get("execution_time", 0.0)))

    extra_col1, extra_col2 = st.columns(2)
    with extra_col1:
        convergence = statistics.get("convergence_generation")
        if convergence is not None:
            st.metric("Converged At", convergence)
        else:
            st.metric("Converged At", "Not reached")
    with extra_col2:
        st.metric("Population Size", statistics.get("population_size", config.get("population_size", 0)))

    if progress is not None:
        st.progress(min(max(progress, 0.0), 1.0))


def _render_result_snapshot(solution: Individual, kpis: Dict):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Distance", f"{solution.total_distance:.2f}")
    with col2:
        st.metric("Fitness", f"{solution.fitness:.4f}")
    with col3:
        st.metric("Routes Used", solution.get_route_count())
    with col4:
        st.metric("Feasible", "Yes" if kpis.get("is_feasible") else "No")

    # Cost breakdown
    st.markdown("**Cost Breakdown**")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        shipping_cost = kpis.get('shipping_total_cost', 0.0)
        st.metric("Shipping Cost", f"{shipping_cost:,.0f} ₫")
    with col6:
        total_cost = kpis.get('total_cost', shipping_cost)
        st.metric("Total Cost", f"{total_cost:,.0f} ₫", 
                 delta=f"{(total_cost - shipping_cost):,.0f} ₫" if total_cost > shipping_cost else None)
    with col7:
        st.metric("Cost / Customer", f"{kpis.get('total_cost_per_customer', kpis.get('shipping_cost_per_customer', 0.0)):,.0f} ₫")
    with col8:
        st.metric("Penalty", f"{kpis.get('penalty', 0.0):.2f}")
    
    # Operational costs breakdown (if available)
    if kpis.get('fuel_cost') or kpis.get('driver_cost') or kpis.get('vehicle_fixed_cost'):
        with st.expander("Operational Costs Breakdown", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Fuel Cost", f"{kpis.get('fuel_cost', 0.0):,.0f} ₫")
            with col2:
                st.metric("Driver Cost", f"{kpis.get('driver_cost', 0.0):,.0f} ₫")
            with col3:
                st.metric("Vehicle Fixed", f"{kpis.get('vehicle_fixed_cost', 0.0):,.0f} ₫")
            with col4:
                st.metric("Total Operational", f"{kpis.get('total_operational_cost', 0.0):,.0f} ₫")
    
    # Constraint violations breakdown (if solution is not feasible)
    if not kpis.get("is_feasible", True):
        violations = kpis.get('constraint_violations', {})
        if violations:
            with st.expander("⚠️ Constraint Violations Details", expanded=True):
                st.warning("Solution has constraint violations. Details below:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Violation Types:**")
                    if violations.get('capacity_violations'):
                        st.error("❌ Capacity Violations")
                    if violations.get('vehicle_count_violations'):
                        st.error("❌ Vehicle Count Violations")
                    if violations.get('customer_visit_violations'):
                        st.error("❌ Customer Visit Violations")
                    if violations.get('depot_violations'):
                        st.error("❌ Depot Violations")
                    if violations.get('time_window_violations'):
                        st.error("❌ Time Window Violations")
                
                with col2:
                    total_violations = violations.get('total_violations', 0)
                    st.metric("Total Penalty", f"{total_violations:.2f}")
                    if total_violations > 0:
                        st.caption("Higher penalty = more severe violations")
                
                # Show penalty breakdown if available
                if solution.penalty > 0:
                    st.write(f"**Penalty Value**: {solution.penalty:.2f}")
                    st.caption("This penalty is applied to fitness calculation to penalize infeasible solutions")


def _format_config_value(value) -> str:
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _format_duration(seconds: float) -> str:
    seconds = seconds or 0.0
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

