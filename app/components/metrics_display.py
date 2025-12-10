"""
Metrics display component for VRP solutions.
"""

import streamlit as st
from typing import Dict, Optional
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.evaluation.metrics import KPICalculator


def render_metrics_panel(
    solution: Individual,
    problem: VRPProblem,
    statistics: Optional[Dict] = None,
    nn_solution: Optional[Individual] = None,
    bks_distance: Optional[float] = None
):
    """
    Render metrics display panel.
    
    Args:
        solution: VRP solution
        problem: VRP problem instance
        statistics: Optional GA statistics
        nn_solution: Optional Nearest Neighbor solution for comparison
        bks_distance: Optional Best Known Solution distance
    """
    if solution.is_empty():
        st.warning("No solution available to display metrics.")
        return
    
    # Calculate KPIs
    kpi_calculator = KPICalculator(problem)
    kpis = kpi_calculator.calculate_kpis(
        solution,
        execution_time=statistics.get('execution_time') if statistics else None
    )
    
    # Basic Metrics Section
    st.subheader("Basic Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Distance",
            f"{solution.total_distance:.2f}"
        )
    
    with col2:
        num_routes = solution.get_route_count()
        st.metric(
            "Number of Routes",
            num_routes
        )
    
    with col3:
        avg_route_length = solution.total_distance / num_routes if num_routes > 0 else 0
        st.metric(
            "Average Route Length",
            f"{avg_route_length:.2f}"
        )
    
    with col4:
        num_customers = solution.get_customer_count()
        st.metric(
            "Total Customers",
            num_customers
        )
    
    # Quality Metrics Section
    with st.expander("Quality Metrics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Capacity Utilization
            avg_utilization = kpis.get('avg_utilization', 0)
            st.write("**Capacity Utilization**")
            st.progress(avg_utilization / 100 if avg_utilization <= 100 else 1.0)
            st.caption(f"Average: {avg_utilization:.1f}%")
            
            # Load Balance
            load_balance = kpis.get('load_balance_index', 0)
            st.write("**Load Balance Index**")
            st.progress(load_balance)
            st.caption(f"Index: {load_balance:.3f} (higher is better)")
        
        with col2:
            # Efficiency Score
            efficiency = kpis.get('efficiency_score', 0)
            st.write("**Efficiency Score**")
            st.progress(efficiency)
            st.caption(f"Score: {efficiency:.3f} (higher is better)")
            
            # Feasibility
            is_feasible = kpis.get('is_feasible', False)
            st.write("**Feasibility**")
            if is_feasible:
                st.success("✓ Feasible Solution")
            else:
                st.error("✗ Infeasible Solution")
                violations = kpis.get('constraint_violations', {})
                if violations:
                    total = violations.get('total_violations', 0)
                    st.caption(f"Violations: {total}")
    
    # Comparison Metrics Section (Enhanced with NN comparison)
    if nn_solution or bks_distance is not None:
        with st.expander("Algorithm Comparison: GA vs Nearest Neighbor", expanded=True):
            st.markdown("### GA vs Nearest Neighbor (NN) Comparison")
            
            comparison_data = []
            
            # GA vs Nearest Neighbor - Detailed comparison
            if nn_solution and not nn_solution.is_empty():
                nn_distance = nn_solution.total_distance
                ga_distance = solution.total_distance
                improvement = ((nn_distance - ga_distance) / nn_distance) * 100
                
                # Calculate additional metrics for comparison
                nn_routes = nn_solution.get_route_count()
                ga_routes = solution.get_route_count()
                
                # VRPTW metrics for both
                nn_tw_metrics = _calculate_vrptw_metrics(nn_solution, problem)
                ga_tw_metrics = _calculate_vrptw_metrics(solution, problem)
                
                # Display side-by-side comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Nearest Neighbor (Baseline)")
                    st.metric("Total Distance", f"{nn_distance:.2f}")
                    st.metric("Number of Routes", nn_routes)
                    st.metric("Time Window Violations", nn_tw_metrics['num_violations'])
                    st.metric("Compliance Rate", f"{nn_tw_metrics['compliance_rate']:.1f}%")
                
                with col2:
                    st.markdown("#### Genetic Algorithm (Optimized)")
                    st.metric("Total Distance", f"{ga_distance:.2f}", delta=f"{improvement:+.2f}%")
                    st.metric("Number of Routes", ga_routes, delta=ga_routes - nn_routes)
                    st.metric("Time Window Violations", ga_tw_metrics['num_violations'], 
                             delta=ga_tw_metrics['num_violations'] - nn_tw_metrics['num_violations'])
                    st.metric("Compliance Rate", f"{ga_tw_metrics['compliance_rate']:.1f}%",
                             delta=f"{ga_tw_metrics['compliance_rate'] - nn_tw_metrics['compliance_rate']:+.1f}%")
                
                with col3:
                    st.markdown("#### Improvement Summary")
                    if improvement > 0:
                        st.success(f"✓ Distance: {improvement:.2f}% better")
                    elif improvement < 0:
                        st.error(f"✗ Distance: {abs(improvement):.2f}% worse")
                        st.caption(" For small datasets (≤20 customers), NN can be competitive. GA may need more generations or prioritize feasibility over pure distance.")
                    else:
                        st.info("-> Distance: Same")
                    
                    route_diff = ga_routes - nn_routes
                    if route_diff < 0:
                        st.success(f"✓ Routes: {abs(route_diff)} fewer")
                    elif route_diff > 0:
                        st.warning(f"⚠ Routes: {route_diff} more")
                    else:
                        st.info("-> Routes: Same")
                    
                    violation_diff = ga_tw_metrics['num_violations'] - nn_tw_metrics['num_violations']
                    if violation_diff < 0:
                        st.success(f"✓ Violations: {abs(violation_diff)} fewer")
                    elif violation_diff > 0:
                        st.error(f"✗ Violations: {violation_diff} more")
                    else:
                        st.info("-> Violations: Same")
                
                comparison_data.append({
                    'Metric': 'vs Nearest Neighbor',
                    'GA Distance': ga_distance,
                    'Baseline Distance': nn_distance,
                    'Improvement': improvement
                })
            
            # GA vs BKS
            if bks_distance is not None:
                ga_distance = solution.total_distance
                improvement = ((bks_distance - ga_distance) / bks_distance) * 100
                
                comparison_data.append({
                    'Metric': 'vs Best Known',
                    'GA Distance': ga_distance,
                    'Baseline Distance': bks_distance,
                    'Improvement': improvement
                })
            
            if comparison_data:
                import pandas as pd
                df = pd.DataFrame(comparison_data)
                
                # Display comparison table
                st.markdown("#### Detailed Comparison Table")
                st.dataframe(
                    df.style.format({
                        'GA Distance': '{:.2f}',
                        'Baseline Distance': '{:.2f}',
                        'Improvement': '{:+.2f}%'
                    }),
                    use_container_width=True
                )
    
    # VRPTW Metrics Section (Simplified for non-technical users)
    tw_metrics = _calculate_vrptw_metrics(solution, problem)
    
    st.markdown("---")
    st.subheader("Time Window Evaluation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        violations = tw_metrics['num_violations']
        if violations == 0:
            st.success(f"✓ **Time Window Compliance: 100%**")
            st.caption("All customers served on time")
        else:
            st.error(f"✗ **Time Window Violations: {violations}**")
            st.caption(f"{tw_metrics['late_arrivals']} late, {tw_metrics['early_arrivals']} early")
    
    with col2:
        compliance = tw_metrics['compliance_rate']
        st.metric(
            "Compliance Rate",
            f"{compliance:.1f}%",
            help="Percentage of customers served within time windows (early arrivals count as compliant, only late = violation)"
        )
        if compliance == 100.0:
            st.success("Perfect compliance!")
        elif compliance >= 95.0:
            st.info("Good compliance")
        else:
            st.warning("Needs improvement")
    
    with col3:
        avg_duration = tw_metrics['avg_route_duration']
        st.metric(
            "Average Route Duration",
            f"{avg_duration:.1f}",
            help="Average time to complete a route"
        )
    
    # Show detailed metrics only if there are violations or user expands
    if tw_metrics['num_violations'] > 0:
        with st.expander("Detailed Time Window Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Time Window Analysis**")
                st.write(f"- Late arrivals (violations): {tw_metrics['late_arrivals']} ✗")
                st.write(f"- Early arrivals (compliant): {tw_metrics['early_arrivals']} ✓")
                st.write(f"- On-time arrivals (compliant): {tw_metrics['on_time_arrivals']} ✓")
                if tw_metrics['max_lateness'] > 0:
                    st.write(f"- Max lateness: {tw_metrics['max_lateness']:.2f}")
            
            with col2:
                st.write("**Route Timing**")
                st.write(f"- Average route duration: {tw_metrics['avg_route_duration']:.2f}")
                st.write(f"- Max route duration: {tw_metrics['max_route_duration']:.2f}")
                st.write(f"- Total travel time: {tw_metrics['total_travel_time']:.2f}")
                st.write(f"- Total service time: {tw_metrics['total_service_time']:.2f}")
    
    # Detailed Statistics Section
    if statistics:
        with st.expander("Detailed Statistics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**GA Statistics**")
                st.write(f"- Generations: {statistics.get('generations', 'N/A')}")
                st.write(f"- Total Evaluations: {statistics.get('total_evaluations', 'N/A')}")
                if 'execution_time' in statistics:
                    st.write(f"- Execution Time: {statistics['execution_time']:.2f}s")
                if 'convergence_generation' in statistics:
                    st.write(f"- Converged at: Generation {statistics['convergence_generation']}")
            
            with col2:
                st.write("**Solution Quality**")
                st.write(f"- Fitness: {solution.fitness:.6f}")
                st.write(f"- Penalty: {solution.penalty:.2f}")
                if 'diversity' in statistics:
                    st.write(f"- Population Diversity: {statistics['diversity']:.4f}")


def _calculate_vrptw_metrics(solution: Individual, problem: VRPProblem) -> Dict:
    """
    Calculate VRPTW-specific metrics.
    
    Args:
        solution: VRP solution
        problem: VRP problem instance
        
    Returns:
        Dictionary with VRPTW metrics
    """
    if solution.is_empty() or not solution.routes:
        return {
            'num_violations': 0,
            'total_waiting_time': 0.0,
            'avg_service_time': 0.0,
            'compliance_rate': 100.0,
            'early_arrivals': 0,
            'late_arrivals': 0,
            'on_time_arrivals': 0,
            'max_lateness': 0.0,
            'total_lateness': 0.0,
            'avg_route_duration': 0.0,
            'max_route_duration': 0.0,
            'total_travel_time': 0.0,
            'total_service_time': 0.0,
            'total_idle_time': 0.0
        }
    
    num_violations = 0
    total_waiting_time = 0.0
    total_service_time = 0.0
    total_travel_time = 0.0
    early_arrivals = 0
    late_arrivals = 0
    on_time_arrivals = 0
    max_lateness = 0.0
    total_lateness = 0.0
    route_durations = []
    
    for route in solution.routes:
        if len(route) < 2:  # Skip empty routes
            continue
        
        current_time = 0.0
        route_start_time = 0.0
        
        for i in range(len(route)):
            customer_id = route[i]
            
            if customer_id == 0:  # Depot
                if i == 0:
                    route_start_time = 0.0
                    current_time = 0.0
                else:
                    # Return to depot
                    prev_id = route[i-1]
                    travel_time = problem.get_distance(prev_id, 0)
                    total_travel_time += travel_time
                    current_time += travel_time
                    route_durations.append(current_time - route_start_time)
                continue
            
            # Get customer
            customer = problem.get_customer_by_id(customer_id)
            if customer is None:
                continue
            
            # Travel time from previous location
            if i > 0:
                prev_id = route[i-1]
                travel_time = problem.get_distance(prev_id, customer_id)
                total_travel_time += travel_time
                current_time += travel_time
            
            # Check time window
            if current_time < customer.ready_time:
                # Early arrival - wait
                waiting_time = customer.ready_time - current_time
                total_waiting_time += waiting_time
                current_time = customer.ready_time
                early_arrivals += 1
            elif current_time > customer.due_date:
                # Late arrival - violation
                lateness = current_time - customer.due_date
                total_lateness += lateness
                max_lateness = max(max_lateness, lateness)
                num_violations += 1
                late_arrivals += 1
            else:
                # On time
                on_time_arrivals += 1
            
            # Service time
            total_service_time += customer.service_time
            current_time += customer.service_time
    
    # Calculate averages
    num_customers = solution.get_customer_count()
    avg_service_time = total_service_time / num_customers if num_customers > 0 else 0.0
    # Early arrivals are compliant (they wait until ready_time, no violation)
    # Only late arrivals (after due_date) are violations
    compliance_rate = ((on_time_arrivals + early_arrivals) / num_customers * 100) if num_customers > 0 else 100.0
    avg_route_duration = sum(route_durations) / len(route_durations) if route_durations else 0.0
    max_route_duration = max(route_durations) if route_durations else 0.0
    
    # Idle time = waiting time
    total_idle_time = total_waiting_time
    
    return {
        'num_violations': num_violations,
        'total_waiting_time': total_waiting_time,
        'avg_service_time': avg_service_time,
        'compliance_rate': compliance_rate,
        'early_arrivals': early_arrivals,
        'late_arrivals': late_arrivals,
        'on_time_arrivals': on_time_arrivals,
        'max_lateness': max_lateness,
        'total_lateness': total_lateness,
        'avg_route_duration': avg_route_duration,
        'max_route_duration': max_route_duration,
        'total_travel_time': total_travel_time,
        'total_service_time': total_service_time,
        'total_idle_time': total_idle_time
    }

