"""
Hanoi Mode page for real-world delivery optimization.
"""

import streamlit as st
import sys
import json
import hashlib
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.data_input import (
    render_file_uploader,
    render_dataset_preview,
    render_save_dataset_form,
    render_load_saved_datasets,
    render_sample_data_button
)
from app.components.manual_entry import render_manual_entry_form, clear_manual_entry
from app.components.parameter_config import render_parameter_config
from app.components.progress_tracker import render_progress_tracker, render_optimization_controls
from app.components.pipeline_progress import PipelineProgress
from app.components.run_summary import render_run_summary
from app.services.data_service import DataService
from app.services.optimization_service import OptimizationService
from app.services.history_service import HistoryService
from app.core.exceptions import ValidationError, DatasetError, OptimizationError
from app.core.logger import setup_app_logger

logger = setup_app_logger()

st.set_page_config(
    page_title="Hanoi Mode - VRP-GA",
    page_icon=None,
    layout="wide"
)

st.title("Hanoi Delivery Optimization")

st.info("""
**Hanoi Mode** is designed for real-world delivery route optimization using actual Hanoi coordinates.

Features:
- Real GPS coordinates
- Interactive Folium maps
- Ahamove shipping cost calculation
- Enhanced Hanoi coordinate system
""")

# Initialize session state - tách riêng cho Hanoi mode
if 'hanoi_dataset' not in st.session_state:
    st.session_state.hanoi_dataset = None
if 'hanoi_dataset_hash' not in st.session_state:
    st.session_state.hanoi_dataset_hash = None
if 'hanoi_problem' not in st.session_state:
    st.session_state.hanoi_problem = None
if 'hanoi_optimization_running' not in st.session_state:
    st.session_state.hanoi_optimization_running = False
if 'hanoi_optimization_results' not in st.session_state:
    st.session_state.hanoi_optimization_results = None
if 'hanoi_optimization_service' not in st.session_state:
    st.session_state.hanoi_optimization_service = OptimizationService()

def _update_hanoi_dataset(new_dataset):
    """Update Hanoi dataset and reset problem if changed."""
    dataset_str = json.dumps(new_dataset, sort_keys=True)
    dataset_hash = hashlib.md5(dataset_str.encode()).hexdigest()
    
    # Nếu dataset thay đổi, reset problem và results
    if st.session_state.hanoi_dataset_hash != dataset_hash:
        st.session_state.hanoi_problem = None
        st.session_state.hanoi_optimization_results = None
        st.session_state.hanoi_optimization_running = False
    
    st.session_state.hanoi_dataset = new_dataset
    st.session_state.hanoi_dataset_hash = dataset_hash

# Tabs for different data input methods
tab1, tab2, tab3, tab4 = st.tabs(["Upload File", "Manual Entry", "Load Saved", "Sample Data"])

with tab1:
    st.markdown("### Upload Dataset File")
    uploaded_data = render_file_uploader(dataset_type="hanoi_mockup")
    
    if uploaded_data:
        _update_hanoi_dataset(uploaded_data)
        
        # Show preview
        render_dataset_preview(uploaded_data, dataset_type="hanoi_mockup")
        
        # Save option
        dataset_id = render_save_dataset_form(uploaded_data, dataset_type="hanoi_mockup")
        if dataset_id:
            st.session_state.hanoi_dataset_id = dataset_id

with tab2:
    st.markdown("### Manual Data Entry")
    st.info("Enter customer data manually. Add customers one by one and see them on the map.")
    
    manual_data = render_manual_entry_form(dataset_type="hanoi_mockup")
    
    if manual_data:
        _update_hanoi_dataset(manual_data)
        
        # Show preview
        render_dataset_preview(manual_data, dataset_type="hanoi_mockup")
        
        # Save option
        dataset_id = render_save_dataset_form(manual_data, dataset_type="hanoi_mockup")
        if dataset_id:
            st.session_state.hanoi_dataset_id = dataset_id
            # Sau khi save, load lại dataset đã save
            try:
                from app.services.database_service import DatabaseService
                db_service = DatabaseService()
                saved_dataset = db_service.load_dataset(dataset_id)
                if saved_dataset:
                    _update_hanoi_dataset(saved_dataset)
                    st.success("Dataset đã được lưu và load lại!")
                    clear_manual_entry()
                    st.rerun()
            except Exception as e:
                st.warning(f"Đã lưu dataset nhưng không thể load lại: {e}")

with tab3:
    st.markdown("### Load Saved Dataset")
    loaded_data = render_load_saved_datasets(dataset_type="hanoi_mockup")
    
    if loaded_data:
        _update_hanoi_dataset(loaded_data)
        
        # Show preview
        render_dataset_preview(loaded_data, dataset_type="hanoi_mockup")

with tab4:
    st.markdown("### Use Sample Data")
    sample_data = render_sample_data_button(dataset_type="hanoi_mockup")
    
    if sample_data:
        _update_hanoi_dataset(sample_data)
        
        # Show preview
        render_dataset_preview(sample_data, dataset_type="hanoi_mockup")
        
        # Save option
        dataset_id = render_save_dataset_form(sample_data, dataset_type="hanoi_mockup")
        if dataset_id:
            st.session_state.hanoi_dataset_id = dataset_id

# If dataset is loaded, show optimization section
if st.session_state.hanoi_dataset:
    st.markdown("---")
    
    # Create VRPProblem if not already created
    if st.session_state.hanoi_problem is None:
        try:
            data_service = DataService()
            with PipelineProgress(
                "Creating VRP problem & distance matrix...",
                success_label="VRP problem ready",
                failure_label="VRP problem failed",
                expanded=True,
            ):
                st.session_state.hanoi_problem = data_service.create_vrp_problem(
                    st.session_state.hanoi_dataset,
                    dataset_type="hanoi"
                )
            st.success("VRP problem created successfully!")
        except Exception as e:
            st.error(f"Error creating VRP problem: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.stop()
    
    # Optimization section
    if st.session_state.hanoi_problem:
        # Parameter configuration
        dataset_size = len(st.session_state.hanoi_problem.customers)
        ga_config = render_parameter_config(dataset_size=dataset_size, dataset_type="hanoi")
        
        if ga_config:
            st.markdown("---")
            
            # Optimization controls
            def run_optimization():
                """Run optimization with configured parameters."""
                st.session_state.hanoi_optimization_running = True
                st.session_state.hanoi_optimization_results = None
            
            def stop_optimization():
                """Stop current optimization."""
                st.session_state.hanoi_optimization_service.stop_optimization()
                st.session_state.hanoi_optimization_running = False
                st.rerun()
            
            render_optimization_controls(
                on_run=run_optimization,
                on_stop=stop_optimization,
                is_running=st.session_state.hanoi_optimization_running,
                key_prefix="hanoi"
            )
            
            # Run optimization if requested
            if st.session_state.hanoi_optimization_running and not st.session_state.hanoi_optimization_results:
                # Progress callback
                progress_placeholder = st.empty()
                
                def progress_callback(gen, max_gen, fitness, distance):
                    """Update progress display."""
                    with progress_placeholder.container():
                        render_progress_tracker(gen, max_gen, fitness, distance, is_running=True)
                
                # Run optimization
                try:
                    with st.spinner("Running optimization..."):
                        best_solution, statistics, evolution_data = st.session_state.hanoi_optimization_service.run_optimization(
                            st.session_state.hanoi_problem,
                            ga_config,
                            progress_callback=progress_callback
                        )
                        
                        st.session_state.hanoi_optimization_results = {
                            'solution': best_solution,
                            'statistics': statistics,
                            'evolution_data': evolution_data,
                            'config': ga_config
                        }
                        st.session_state.hanoi_optimization_running = False
                        
                        # Save to history
                        try:
                            history_service = HistoryService()
                            
                            # Get dataset name from metadata or use default
                            dataset_metadata = (st.session_state.hanoi_dataset or {}).get('metadata', {})
                            dataset_name = dataset_metadata.get('name', st.session_state.get('hanoi_dataset_name', 'Hanoi Dataset'))
                            
                            # Ensure we have a valid dataset
                            if not st.session_state.hanoi_dataset:
                                logger.warning("Cannot save to history: hanoi_dataset is None")
                                st.warning("⚠️ Không thể lưu vào history: Dataset không tồn tại")
                            else:
                                # Try to find existing dataset or create new one
                                from app.config.database import SessionLocal
                                from app.database.crud import get_datasets, create_dataset
                                db = SessionLocal()
                                try:
                                    # Try to find dataset by name first (check all types to handle migration)
                                    datasets_all = get_datasets(db, type=None)  # Get all datasets
                                    dataset_id = None
                                    
                                    # Find dataset by name (prefer 'hanoi' type, but accept 'hanoi_mockup' for backward compatibility)
                                    for ds in datasets_all:
                                        if ds.name == dataset_name:
                                            # Prefer 'hanoi' type, but accept 'hanoi_mockup' if exists
                                            if ds.type == 'hanoi':
                                                dataset_id = ds.id
                                                logger.info(f"Found dataset {dataset_name} with type 'hanoi' (ID: {dataset_id})")
                                                break
                                            elif ds.type == 'hanoi_mockup' and dataset_id is None:
                                                dataset_id = ds.id
                                                logger.info(f"Found dataset {dataset_name} with type 'hanoi_mockup' (ID: {dataset_id}), will use this")
                                    
                                    # If not found, try searching only 'hanoi' type
                                    if not dataset_id:
                                        datasets = get_datasets(db, type='hanoi')
                                        for ds in datasets:
                                            if ds.name == dataset_name:
                                                dataset_id = ds.id
                                                logger.info(f"Found dataset {dataset_name} with type 'hanoi' (ID: {dataset_id})")
                                                break
                                    
                                    if not dataset_id:
                                        # Create new dataset entry
                                        dataset_json = json.dumps(st.session_state.hanoi_dataset)
                                        dataset = create_dataset(
                                            db,
                                            name=dataset_name,
                                            description=f"Hanoi delivery: {dataset_name}",
                                            type='hanoi',  # Use consistent type name
                                            data_json=dataset_json
                                        )
                                        dataset_id = dataset.id
                                        logger.info(f"Created new dataset entry: {dataset_name} (ID: {dataset_id})")
                                    
                                    # Save result
                                    run_id, is_new_best = history_service.save_result(
                                        dataset_id=dataset_id,
                                        dataset_name=dataset_name,
                                        solution=best_solution,
                                        statistics=statistics,
                                        config=ga_config,
                                        dataset_type="hanoi"  # Use consistent type name
                                    )
                                    
                                    if run_id:
                                        logger.info(f"Saved optimization run to history: {run_id}")
                                        if is_new_best:
                                            st.session_state['new_best_result'] = True
                                            st.success(f"✅ Đã lưu vào history và đây là kết quả tốt nhất!")
                                        else:
                                            st.info(f"✅ Đã lưu vào history (Run ID: {run_id})")
                                    else:
                                        logger.warning("Failed to save to history: run_id is None")
                                        st.warning("⚠️ Không thể lưu vào history")
                                except Exception as db_error:
                                    logger.error(f"Database error saving to history: {db_error}", exc_info=True)
                                    st.error(f"❌ Lỗi khi lưu vào history: {str(db_error)}")
                                finally:
                                    db.close()
                        except Exception as e:
                            logger.error(f"Failed to save to history: {e}", exc_info=True)
                            st.error(f"❌ Lỗi khi lưu vào history: {str(e)}")
                        
                        st.rerun()
                        
                except OptimizationError as e:
                    st.error(f"Optimization failed: {str(e)}")
                    st.session_state.hanoi_optimization_running = False
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    st.session_state.hanoi_optimization_running = False
            
            # Show results if available
            if st.session_state.hanoi_optimization_results:
                st.markdown("---")
                st.success("Optimization completed!")
                
                results = st.session_state.hanoi_optimization_results
                solution = results['solution']
                stats = results['statistics']
                
                # Run summary ensures all parameters/metrics are in sync
                dataset_metadata = (st.session_state.hanoi_dataset or {}).get('metadata', {})
                render_run_summary(
                    st.session_state.hanoi_problem,
                    solution,
                    results.get('config'),
                    stats,
                    dataset_metadata=dataset_metadata,
                    dataset_type="hanoi"
                )
                
                # Evolution chart
                if results['evolution_data'] and len(results['evolution_data']) > 0:
                    try:
                        import pandas as pd
                        import plotly.graph_objects as go
                        
                        df = pd.DataFrame(results['evolution_data'])
                        
                        # Create figure with two y-axes
                        fig = go.Figure()
                        
                        # Best distance line
                        fig.add_trace(go.Scatter(
                            x=df['generation'],
                            y=df['best_distance'],
                            mode='lines+markers',
                            name='Best Distance',
                            line=dict(color='blue', width=2),
                            marker=dict(size=4)
                        ))
                        
                        # Average fitness line (if available)
                        if 'avg_fitness' in df.columns and df['avg_fitness'].notna().any():
                            # Normalize avg_fitness for display (inverse to show improvement)
                            fig.add_trace(go.Scatter(
                                x=df['generation'],
                                y=df['avg_fitness'],
                                mode='lines',
                                name='Avg Fitness',
                                line=dict(color='green', width=1, dash='dash'),
                                yaxis='y2'
                            ))
                        
                        fig.update_layout(
                            title="Optimization Progress",
                            xaxis_title="Generation",
                            yaxis_title="Distance",
                            yaxis2=dict(
                                title="Fitness",
                                overlaying='y',
                                side='right'
                            ) if 'avg_fitness' in df.columns else None,
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create evolution chart: {e}")
                        # Fallback: show data table
                        st.dataframe(df[['generation', 'best_distance', 'best_fitness']])
                
                # Visualization Section
                st.markdown("---")
                st.subheader("Visualization")
                
                # Import visualization components
                from app.services.visualization_service import VisualizationService
                from app.components.metrics_display import render_metrics_panel
                import streamlit_folium as st_folium
                
                visualization_service = VisualizationService()
                
                # Route visibility controls
                if len(solution.routes) > 1:
                    st.write("**Route Visibility Controls**")
                    route_visibility = []
                    cols = st.columns(min(5, len(solution.routes)))
                    for i, route in enumerate(solution.routes):
                        if route:
                            col_idx = i % len(cols)
                            with cols[col_idx]:
                                visible = st.checkbox(
                                    f"Route {i+1}",
                                    value=True,
                                    key=f"route_vis_{i}"
                                )
                                route_visibility.append(visible)
                else:
                    route_visibility = [True] if solution.routes else []
                
                # Create and display map
                try:
                    folium_map = visualization_service.create_hanoi_map(
                        solution,
                        st.session_state.hanoi_problem,
                        route_visibility if len(route_visibility) == len(solution.routes) else None
                    )
                    
                    # Display map
                    st_folium.st_folium(
                        folium_map,
                        width=700,
                        height=500,
                        returned_objects=[]
                    )
                    
                    # Download button for map
                    map_html = folium_map._repr_html_()
                    st.download_button(
                        label="Download Map (HTML)",
                        data=map_html,
                        file_name="vrp_solution_map.html",
                        mime="text/html"
                    )
                    
                except Exception as e:
                    st.error(f"Error creating map: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                
                # Metrics Panel
                st.markdown("---")
                st.subheader("Solution Metrics")
                
                # Get Nearest Neighbor solution for comparison
                nn_solution = None
                try:
                    from src.algorithms.nearest_neighbor import NearestNeighborHeuristic
                    nn_heuristic = NearestNeighborHeuristic(st.session_state.hanoi_problem)
                    nn_solution = nn_heuristic.solve()
                except Exception as e:
                    st.warning(f"Could not generate Nearest Neighbor baseline: {str(e)}")
                
                # Render metrics panel
                render_metrics_panel(
                    solution,
                    st.session_state.hanoi_problem,
                    statistics=stats,
                    nn_solution=nn_solution
                )
                
                # Comparison Chart
                if nn_solution:
                    st.markdown("---")
                    st.subheader("Solution Comparison")
                    
                    try:
                        comparison_chart = visualization_service.create_comparison_chart(
                            solution,
                            st.session_state.hanoi_problem,
                            nn_solution=nn_solution
                        )
                        st.plotly_chart(comparison_chart, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create comparison chart: {str(e)}")
else:
    st.info("Please upload a dataset, load a saved dataset, or use sample data to get started.")
