"""
History page to view past optimization runs and best results.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.database_service import DatabaseService
from app.database.crud import get_all_best_results, get_optimization_runs, get_optimization_run
from app.core.exceptions import DatabaseError
from app.config.database import SessionLocal

st.set_page_config(
    page_title="History - VRP-GA",
    page_icon=None,
    layout="wide"
)

st.title("Optimization History")

st.info("""
View your optimization history and best results for each dataset.
Best results are automatically updated when you achieve better solutions.
""")

# Initialize database service (for compatibility, but we'll use SessionLocal directly)
try:
    db_service = DatabaseService()
except Exception as e:
    st.error(f"Database error: {str(e)}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["Best Results", "All Runs"])

with tab1:
    st.markdown("### Best Results by Dataset")
    st.markdown("The best optimization result achieved for each dataset.")
    
    try:
        db = SessionLocal()
        best_results = get_all_best_results(db, limit=100)
        db.close()
        
        if not best_results:
            st.info("No best results found. Run optimizations to see results here.")
        else:
            # Filter options
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input("Search datasets", placeholder="Search by name...", key="best_search")
            with col2:
                filter_type = st.selectbox(
                    "Filter by type",
                    options=["All", "hanoi", "hanoi_mockup"],
                    index=0,
                    key="best_filter"
                )
            
            # Filter results
            filtered_results = best_results
            if search_term:
                filtered_results = [r for r in filtered_results if search_term.lower() in r.dataset_name.lower()]
            if filter_type != "All":
                # Need to check dataset type from dataset_id
                db = SessionLocal()
                from app.database.crud import get_dataset
                filtered_results = [
                    r for r in filtered_results
                    if get_dataset(db, r.dataset_id) and (
                        get_dataset(db, r.dataset_id).type == filter_type or
                        (filter_type == "hanoi" and get_dataset(db, r.dataset_id).type in ["hanoi", "hanoi_mockup"])
                    )
                ]
                db.close()
            
            st.markdown(f"**Found {len(filtered_results)} best result(s)**")

            # Display best results
            for best_result in filtered_results:
                with st.expander(
                    f"**{best_result.dataset_name}** - "
                    f"Distance: {best_result.total_distance:.2f} km, "
                    f"Violations: {best_result.time_window_violations}, "
                    f"Routes: {best_result.num_routes}",
                    expanded=False
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Metrics**")
                        st.write(f"**Total Distance:** {best_result.total_distance:.2f} km")
                        st.write(f"**Number of Routes:** {best_result.num_routes}")
                        st.write(f"**Time Window Violations:** {best_result.time_window_violations}")
                        st.write(f"**Compliance Rate:** {best_result.compliance_rate:.1f}%")
                    
                    with col2:
                        st.markdown("**Quality**")
                        st.write(f"**Fitness:** {best_result.fitness:.6f}")
                        st.write(f"**Penalty:** {best_result.penalty:.2f}")
                        if best_result.gap_vs_bks is not None:
                            st.write(f"**Gap vs BKS:** {best_result.gap_vs_bks:.2f}%")
                            st.write(f"**BKS Distance:** {best_result.bks_distance:.2f} km")
                    
                    with col3:
                        st.markdown("**Timeline**")
                        achieved_date = datetime.fromisoformat(str(best_result.achieved_at)).strftime('%Y-%m-%d %H:%M:%S')
                        st.write(f"**Achieved:** {achieved_date}")
                        updated_date = datetime.fromisoformat(str(best_result.updated_at)).strftime('%Y-%m-%d %H:%M:%S')
                        st.write(f"**Updated:** {updated_date}")
                    
                    # View run details button
                    if st.button("View Run Details", key=f"view_run_{best_result.run_id}"):
                        st.session_state['selected_run_id'] = best_result.run_id
                        st.rerun()
                    
    except Exception as e:
        st.error(f"Error loading best results: {str(e)}")

with tab2:
    st.markdown("### All Optimization Runs")
    st.markdown("Complete history of all optimization runs.")
    
    try:
        db = SessionLocal()
        all_runs = get_optimization_runs(db, limit=100)
        db.close()
        
        if not all_runs:
            st.info("No optimization runs found. Run optimizations to see history here.")
        else:
            # Filter options
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input("Search runs", placeholder="Search by name...", key="runs_search")
            with col2:
                status_filter = st.selectbox(
                    "Filter by status",
                    options=["All", "completed", "failed", "stopped"],
                    index=0,
                    key="runs_filter"
                )
            
            # Filter runs
            filtered_runs = all_runs
            if search_term:
                filtered_runs = [r for r in filtered_runs if search_term.lower() in r.name.lower()]
            if status_filter != "All":
                filtered_runs = [r for r in filtered_runs if r.status == status_filter]
            
            st.markdown(f"**Found {len(filtered_runs)} run(s)**")
            
            # Display runs
            for run in filtered_runs:
                try:
                    results_data = json.loads(run.results_json)
                    distance = results_data.get('total_distance', 0)
                    violations = results_data.get('time_window_violations', 0)
                    routes = results_data.get('num_routes', 0)
                except:
                    distance = 0
                    violations = 0
                    routes = 0
                
                status_color = {
                    'completed': '🟢',
                    'failed': '🔴',
                    'stopped': '🟡',
                    'running': '🔵'
                }.get(run.status, '⚪')
                
                with st.expander(
                    f"{status_color} **{run.name}** - "
                    f"Distance: {distance:.2f} km, "
                    f"Violations: {violations}, "
                    f"Status: {run.status}",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Run Information**")
                        st.write(f"**Status:** {run.status}")
                        if run.started_at:
                            start_date = datetime.fromisoformat(str(run.started_at)).strftime('%Y-%m-%d %H:%M:%S')
                            st.write(f"**Started:** {start_date}")
                        if run.completed_at:
                            end_date = datetime.fromisoformat(str(run.completed_at)).strftime('%Y-%m-%d %H:%M:%S')
                            st.write(f"**Completed:** {end_date}")
                        created_date = datetime.fromisoformat(str(run.created_at)).strftime('%Y-%m-%d %H:%M:%S')
                        st.write(f"**Created:** {created_date}")
                    
                    with col2:
                        st.markdown("**Results**")
                        st.write(f"**Distance:** {distance:.2f} km")
                        st.write(f"**Routes:** {routes}")
                        st.write(f"**Violations:** {violations}")
                    
                    if run.notes:
                        st.markdown("**Notes:**")
                        st.write(run.notes)
                    
                    if run.error_message:
                        st.error(f"**Error:** {run.error_message}")
                    
                    # View details button
                    if st.button("View Details", key=f"view_details_{run.id}"):
                        st.session_state['selected_run_id'] = run.id
                        st.rerun()
                    
    except Exception as e:
        st.error(f"Error loading runs: {str(e)}")

# Show run details if selected
if 'selected_run_id' in st.session_state:
    st.markdown("---")
    st.markdown("###  Run Details")
    
    try:
        db = SessionLocal()
        run = get_optimization_run(db, st.session_state['selected_run_id'])
        db.close()
        
        if run:
            # Parse results
            try:
                results_data = json.loads(run.results_json)
                parameters_data = json.loads(run.parameters_json) if run.parameters_json else {}
            except:
                results_data = {}
                parameters_data = {}
            
            # Main Results Section
            st.markdown("#### 🎯 Optimization Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                distance = results_data.get('total_distance', 0)
                st.metric("📏 Total Distance", f"{distance:.2f} km")
            with col2:
                routes = results_data.get('num_routes', 0)
                st.metric("🚚 Number of Routes", routes)
            with col3:
                violations = results_data.get('time_window_violations', 0)
                # Try to get from statistics if main value looks wrong
                if violations > 1000000 or violations < 0:
                    # Likely penalty value instead of count - try to get from statistics
                    stats = results_data.get('statistics', {})
                    if isinstance(stats, dict):
                        violations = stats.get('time_window_violations', 0)
                        # Also try constraint_violations
                        if violations > 1000000 or violations < 0:
                            constraint_violations = stats.get('constraint_violations', {})
                            if isinstance(constraint_violations, dict):
                                violations = constraint_violations.get('time_window_violations', 0)
                
                # Display violations count (not penalty)
                if violations > 1000000 or violations < 0:
                    # Still looks wrong - show as N/A with warning
                    st.metric(" Violations", "N/A", 
                             help="Violations data may be incorrect (showing penalty value instead of count)")
                else:
                    st.metric(" Violations", int(violations))
            with col4:
                compliance = results_data.get('compliance_rate', 0)
                st.metric(" Compliance Rate", f"{compliance:.1f}%")
            
            # Additional Metrics
            st.markdown("#### 📈 Additional Metrics")
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                fitness = results_data.get('fitness', 0)
                st.metric("💪 Fitness", f"{fitness:.2f}")
            with col6:
                penalty = results_data.get('penalty', 0)
                if penalty > 0:
                    st.metric("💰 Penalty", f"{penalty:,.0f}")
                else:
                    st.metric("💰 Penalty", "0 (No violations)")
            with col7:
                gap_vs_bks = results_data.get('gap_vs_bks')
                # Try to get from statistics if not in main results
                if gap_vs_bks is None:
                    stats = results_data.get('statistics', {})
                    if isinstance(stats, dict):
                        gap_vs_bks = stats.get('gap_vs_bks') or stats.get('bks_gap_percent')
                        bks_distance = stats.get('bks_distance')
                        if gap_vs_bks is not None and bks_distance:
                            results_data['bks_distance'] = bks_distance
                
                if gap_vs_bks is not None:
                    gap = float(gap_vs_bks)
                    bks_dist = results_data.get('bks_distance', 0)
                    if bks_dist:
                        st.metric("🎯 Gap vs BKS", f"{gap:+.2f}%", 
                                 help=f"Best Known Solution: {bks_dist:.2f} km")
                    else:
                        st.metric("🎯 Gap vs BKS", f"{gap:+.2f}%")
                else:
                    st.metric("🎯 Gap vs BKS", "N/A", 
                             help="BKS data not available for this dataset")
            with col8:
                # Try to get runtime from multiple sources
                runtime_seconds = None
                
                # Priority 1: Calculate from started_at and completed_at
                if run.started_at and run.completed_at:
                    runtime_seconds = (run.completed_at - run.started_at).total_seconds()
                # Priority 2: Get from statistics
                elif results_data.get('statistics', {}).get('execution_time'):
                    runtime_seconds = results_data['statistics']['execution_time']
                # Priority 3: Get from results_data directly
                elif results_data.get('execution_time'):
                    runtime_seconds = results_data['execution_time']
                
                if runtime_seconds is not None and runtime_seconds > 0:
                    minutes = int(runtime_seconds // 60)
                    seconds = int(runtime_seconds % 60)
                    if minutes > 0:
                        st.metric("⏱️ Runtime", f"{minutes}m {seconds}s")
                    else:
                        st.metric("⏱️ Runtime", f"{seconds}s")
                else:
                    st.metric("⏱️ Runtime", "N/A", 
                             help="Runtime data not available")
            
            # Comparison with Nearest Neighbor Baseline
            st.markdown("####  Comparison with Baseline")
            try:
                from app.database.crud import get_dataset
                db = SessionLocal()
                dataset = get_dataset(db, run.dataset_id)
                db.close()
                
                if dataset:
                    # Load problem from dataset
                    try:
                        dataset_data = json.loads(dataset.data_json)
                        from app.services.data_service import DataService
                        data_service = DataService()
                        problem = data_service.create_vrp_problem(dataset_data, dataset.type)
                        
                        if problem:
                            # Calculate NN baseline
                            from src.algorithms.nearest_neighbor import NearestNeighborHeuristic
                            nn_heuristic = NearestNeighborHeuristic(problem)
                            nn_solution = nn_heuristic.solve()
                            
                            nn_distance = nn_solution.total_distance
                            ga_distance = results_data.get('total_distance', 0)
                            
                            if nn_distance > 0 and ga_distance > 0:
                                improvement = ((nn_distance - ga_distance) / nn_distance) * 100
                                improvement_abs = nn_distance - ga_distance
                                
                                comp_col1, comp_col2, comp_col3 = st.columns(3)
                                with comp_col1:
                                    st.metric("🔵 NN Baseline", f"{nn_distance:.2f} km",
                                             help="Nearest Neighbor baseline distance")
                                with comp_col2:
                                    st.metric("🟢 GA Solution", f"{ga_distance:.2f} km",
                                             help="Genetic Algorithm solution distance")
                                with comp_col3:
                                    if improvement > 0:
                                        st.metric("📈 Improvement", f"{improvement:+.2f}%",
                                                 delta=f"{improvement_abs:.2f} km",
                                                 help=f"GA improved by {improvement:.2f}% vs NN baseline")
                                    else:
                                        st.metric("📉 Improvement", f"{improvement:+.2f}%",
                                                 delta=f"{improvement_abs:.2f} km",
                                                 help="GA solution vs NN baseline")
                    except Exception as e:
                        st.warning(f"Could not calculate NN baseline: {str(e)}")
            except Exception as e:
                st.warning(f"Could not load dataset for comparison: {str(e)}")
            
            # Configuration Section
            if parameters_data:
                with st.expander("⚙️ Configuration Parameters", expanded=False):
                    config_cols = st.columns(3)
                    config_items = [
                        ("Population Size", parameters_data.get('population_size')),
                        ("Generations", parameters_data.get('generations')),
                        ("Crossover", parameters_data.get('crossover_prob')),
                        ("Mutation", parameters_data.get('mutation_prob')),
                        ("Tournament", parameters_data.get('tournament_size')),
                        ("Elitism", parameters_data.get('elitism_rate')),
                        ("Split Algorithm", "On" if parameters_data.get('use_split_algorithm') else "Off"),
                        ("Penalty Weight", parameters_data.get('penalty_weight')),
                    ]
                    
                    for idx, (label, value) in enumerate(config_items):
                        with config_cols[idx % 3]:
                            if value is not None:
                                if isinstance(value, float):
                                    st.write(f"**{label}:** {value:.3f}")
                                elif isinstance(value, bool):
                                    st.write(f"**{label}:** {'Yes' if value else 'No'}")
                                else:
                                    st.write(f"**{label}:** {value}")
            
            # Statistics Section (if available)
            if 'statistics' in results_data and results_data['statistics']:
                with st.expander(" Detailed Statistics", expanded=False):
                    stats = results_data['statistics']
                    stats_cols = st.columns(2)
                    with stats_cols[0]:
                        st.write("**GA Statistics:**")
                        st.write(f"- Generations: {stats.get('generations', 'N/A')}")
                        st.write(f"- Total Evaluations: {stats.get('total_evaluations', 'N/A')}")
                        if 'execution_time' in stats:
                            st.write(f"- Execution Time: {stats['execution_time']:.2f}s")
                        if 'convergence_generation' in stats:
                            st.write(f"- Converged at: Generation {stats['convergence_generation']}")
                    with stats_cols[1]:
                        st.write("**Solution Quality:**")
                        st.write(f"- Fitness: {results_data.get('fitness', 0):.6f}")
                        if 'diversity' in stats:
                            st.write(f"- Population Diversity: {stats['diversity']:.4f}")
            
            # Raw JSON (for technical users)
            with st.expander(" Raw JSON Data (Technical)", expanded=False):
                st.json(results_data)
            
            # Close button
            if st.button(" Close Details", use_container_width=True):
                del st.session_state['selected_run_id']
                st.rerun()
        else:
            st.error("Run not found")
            del st.session_state['selected_run_id']
    except Exception as e:
        st.error(f"Error loading run details: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
