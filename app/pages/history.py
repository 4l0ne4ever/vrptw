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

# Initialize database service
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
        db = db_service.get_session()
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
                    options=["All", "solomon", "hanoi_mockup"],
                    index=0,
                    key="best_filter"
                )
            
            # Filter results
            filtered_results = best_results
            if search_term:
                filtered_results = [r for r in filtered_results if search_term.lower() in r.dataset_name.lower()]
            if filter_type != "All":
                # Need to check dataset type from dataset_id
                db = db_service.get_session()
                from app.database.crud import get_dataset
                filtered_results = [
                    r for r in filtered_results
                    if get_dataset(db, r.dataset_id) and get_dataset(db, r.dataset_id).type == filter_type
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
        db = db_service.get_session()
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
    st.markdown("### Run Details")
    
    try:
        db = db_service.get_session()
        run = get_optimization_run(db, st.session_state['selected_run_id'])
        db.close()
        
        if run:
            st.json(json.loads(run.results_json))
            
            if st.button("Close Details"):
                del st.session_state['selected_run_id']
                st.rerun()
        else:
            st.error("Run not found")
            del st.session_state['selected_run_id']
    except Exception as e:
        st.error(f"Error loading run details: {str(e)}")
