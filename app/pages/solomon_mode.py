"""
Solomon Mode page for academic benchmark optimization.
"""

import streamlit as st
import sys
import json
import hashlib
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.components.data_input import (
    render_file_uploader,
    render_dataset_preview,
    render_save_dataset_form,
    render_load_saved_datasets,
    render_sample_data_button
)
from app.components.parameter_config import render_parameter_config
from app.components.progress_tracker import render_progress_tracker, render_optimization_controls
from app.components.pipeline_progress import PipelineProgress
from app.components.metrics_display import render_metrics_panel
from app.components.run_summary import render_run_summary
from app.services.data_service import DataService
from app.services.optimization_service import OptimizationService
from app.services.visualization_service import VisualizationService
from app.core.exceptions import ValidationError, DatasetError, OptimizationError
from src.data_processing.solomon_cache_warmer import SolomonCacheWarmer

st.set_page_config(
    page_title="Solomon Mode - VRP-GA",
    page_icon=None,
    layout="wide"
)

# Warm Solomon cache asynchronously at page load (non-blocking)
SolomonCacheWarmer.ensure_started()

st.title("Solomon Dataset Mode")

st.info("""
**Solomon Mode** is designed for academic benchmarking using standard Solomon instances.

Features:
- BKS (Best-Known Solution) validation
- Performance charts and analytics
- Comparison with baseline algorithms
- Plotly visualization (scatter plots)
""")

# Initialize session state - tách riêng cho Solomon mode
if 'solomon_dataset' not in st.session_state:
    st.session_state.solomon_dataset = None
if 'solomon_dataset_hash' not in st.session_state:
    st.session_state.solomon_dataset_hash = None
if 'solomon_problem' not in st.session_state:
    st.session_state.solomon_problem = None
if 'solomon_optimization_running' not in st.session_state:
    st.session_state.solomon_optimization_running = False
if 'solomon_optimization_results' not in st.session_state:
    st.session_state.solomon_optimization_results = None
if 'solomon_optimization_service' not in st.session_state:
    st.session_state.solomon_optimization_service = OptimizationService()

def _update_solomon_dataset(new_dataset):
    """Update Solomon dataset and reset problem if changed."""
    dataset_str = json.dumps(new_dataset, sort_keys=True)
    dataset_hash = hashlib.md5(dataset_str.encode()).hexdigest()
    
    # Nếu dataset thay đổi, reset problem và results
    if st.session_state.solomon_dataset_hash != dataset_hash:
        st.session_state.solomon_problem = None
        st.session_state.solomon_optimization_results = None
        st.session_state.solomon_optimization_running = False
    
    st.session_state.solomon_dataset = new_dataset
    st.session_state.solomon_dataset_hash = dataset_hash

# Tabs for different data input methods - Only catalog for Solomon
tab1 = st.tabs(["Select from Catalog"])[0]

with tab1:
    st.markdown("### Select from Solomon Catalog")
    
    # List available Solomon datasets
    solomon_dir = Path("data/datasets/solomon")
    if solomon_dir.exists():
        json_files = list(solomon_dir.glob("*.json"))
        if json_files:
            dataset_names = [f.stem for f in json_files]
            selected_dataset = st.selectbox(
                "Select a Solomon instance:",
                options=dataset_names,
                help="Choose a Solomon benchmark instance"
            )
            
            if st.button("Load Dataset", use_container_width=True):
                try:
                    data_service = DataService()
                    dataset_path = solomon_dir / f"{selected_dataset}.json"
                    
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        import json
                        data_dict = json.load(f)
                    
                    _update_solomon_dataset(data_dict)
                    st.success(f"Loaded {selected_dataset} successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
        else:
            st.info("No Solomon datasets found in catalog. Please upload a dataset.")
    else:
        st.info("Solomon dataset directory not found.")

# If dataset is loaded, show optimization section
if st.session_state.solomon_dataset:
    st.markdown("---")
    
    # Create VRPProblem if not already created
    if st.session_state.solomon_problem is None:
        try:
            data_service = DataService()
            with PipelineProgress(
                "Creating VRP problem & distance matrix...",
                success_label="VRP problem ready",
                failure_label="VRP problem failed",
                expanded=True,
            ):
                st.session_state.solomon_problem = data_service.create_vrp_problem(
                    st.session_state.solomon_dataset,
                    dataset_type="solomon"
                )
            st.success("VRP problem created successfully!")
        except Exception as e:
            st.error(f"Error creating VRP problem: {str(e)}")
            st.stop()
    
    # Optimization section
    if st.session_state.solomon_problem:
        # Parameter configuration
        dataset_size = len(st.session_state.solomon_problem.customers)
        ga_config = render_parameter_config(dataset_size=dataset_size, dataset_type="solomon")
        
        if ga_config:
            st.markdown("---")
            
            # Optimization controls
            def run_optimization():
                """Run optimization with configured parameters."""
                st.session_state.solomon_optimization_running = True
                st.session_state.solomon_optimization_results = None
            
            def stop_optimization():
                """Stop current optimization."""
                st.session_state.solomon_optimization_service.stop_optimization()
                st.session_state.solomon_optimization_running = False
                st.rerun()
            
            # Only render controls if not currently running to avoid duplicate buttons
            if not st.session_state.solomon_optimization_running:
                render_optimization_controls(
                    on_run=run_optimization,
                    on_stop=stop_optimization,
                    is_running=False,
                    key_prefix="solomon"
                )
            
            # Run optimization if requested
            if st.session_state.solomon_optimization_running and not st.session_state.solomon_optimization_results:
                # Run optimization without progress callback
                # Streamlit doesn't support real-time UI updates during blocking operations
                # Progress is logged to console/file instead
                try:
                    with st.spinner("Running optimization... (check logs for progress)"):
                        best_solution, statistics, evolution_data = st.session_state.solomon_optimization_service.run_optimization(
                            st.session_state.solomon_problem,
                            ga_config,
                            progress_callback=None
                        )
                    
                    st.session_state.solomon_optimization_results = {
                        'solution': best_solution,
                        'statistics': statistics,
                        'evolution_data': evolution_data,
                        'config': ga_config
                    }
                    st.session_state.solomon_optimization_running = False
                    # Clear progress
                    if 'optimization_progress' in st.session_state:
                        del st.session_state.optimization_progress
                    st.rerun()
                        
                except OptimizationError as e:
                    st.error(f"Optimization failed: {str(e)}")
                    st.session_state.solomon_optimization_running = False
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    st.session_state.solomon_optimization_running = False
            
            # Show results if available
            if st.session_state.solomon_optimization_results:
                st.markdown("---")
                st.success("Optimization completed!")
                
                results = st.session_state.solomon_optimization_results
                solution = results['solution']
                stats = results['statistics']
                
                dataset_metadata = (st.session_state.solomon_dataset or {}).get('metadata', {})
                render_run_summary(
                    st.session_state.solomon_problem,
                    solution,
                    results.get('config'),
                    stats,
                    dataset_metadata=dataset_metadata,
                    dataset_type="solomon"
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
                
                # Visualization Section - Plotly charts for Solomon
                st.markdown("---")
                st.subheader("Visualization")
                
                visualization_service = VisualizationService()
                
                # Create Solomon chart (scatter plot)
                try:
                    solomon_chart = visualization_service.create_solomon_chart(
                        solution,
                        st.session_state.solomon_problem
                    )
                    st.plotly_chart(solomon_chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                
                # Get BKS distance if available
                bks_distance = None
                bks_gap = None
                try:
                    from src.evaluation.bks_validator import BKSValidator
                    dataset_name = st.session_state.solomon_dataset.get('metadata', {}).get('name', '')
                    if dataset_name:
                        bks_validator = BKSValidator()
                        bks_data = bks_validator.get_bks(dataset_name)
                        if bks_data:
                            bks_distance = bks_data.get('distance')
                            if bks_distance:
                                bks_gap = ((solution.total_distance - bks_distance) / bks_distance) * 100
                except Exception as e:
                    st.warning(f"Could not load BKS data: {str(e)}")
                
                # Get Nearest Neighbor solution for comparison
                nn_solution = None
                try:
                    from src.algorithms.nearest_neighbor import NearestNeighborHeuristic
                    nn_heuristic = NearestNeighborHeuristic(st.session_state.solomon_problem)
                    nn_solution = nn_heuristic.solve()
                except Exception as e:
                    st.warning(f"Could not generate Nearest Neighbor baseline: {str(e)}")
                
                # Metrics Panel
                st.markdown("---")
                st.subheader("Solution Metrics")
                
                if bks_distance:
                    gap_text = f"{bks_gap:+.2f}%" if bks_gap is not None else "N/A"
                    bks_col1, bks_col2 = st.columns(2)
                    with bks_col1:
                        st.metric("BKS Distance", f"{bks_distance:.2f}")
                    with bks_col2:
                        st.metric("Gap vs BKS", gap_text)
                
                # Render metrics panel
                render_metrics_panel(
                    solution,
                    st.session_state.solomon_problem,
                    statistics=stats,
                    nn_solution=nn_solution,
                    bks_distance=bks_distance
                )
                
                # Comparison Chart
                if nn_solution or bks_distance:
                    st.markdown("---")
                    st.subheader("Solution Comparison")
                    
                    try:
                        comparison_chart = visualization_service.create_comparison_chart(
                            solution,
                            st.session_state.solomon_problem,
                            nn_solution=nn_solution,
                            bks_distance=bks_distance
                        )
                        st.plotly_chart(comparison_chart, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create comparison chart: {str(e)}")
else:
    st.info("Please upload a Solomon dataset, load a saved dataset, or select from catalog to get started.")
