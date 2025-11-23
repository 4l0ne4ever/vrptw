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
from app.services.history_service import HistoryService
from app.core.exceptions import ValidationError, DatasetError, OptimizationError
from app.core.logger import setup_app_logger
from src.data_processing.solomon_cache_warmer import SolomonCacheWarmer

logger = setup_app_logger()

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
    
    # List available Solomon datasets, sorted by difficulty (easiest to hardest)
    solomon_dir = Path("data/datasets/solomon")
    if solomon_dir.exists():
        json_files = list(solomon_dir.glob("*.json"))
        if json_files:
            # Load BKS data to calculate difficulty
            bks_path = Path("data/solomon_bks.json")
            difficulty_scores = {}
            
            if bks_path.exists():
                try:
                    with open(bks_path, 'r') as f:
                        bks_data = json.load(f)
                    
                    def get_difficulty_score(name, data):
                        """Calculate difficulty: higher distance + more vehicles = harder"""
                        distance = data.get('distance', 0)
                        vehicles = data.get('vehicles', 0)
                        return (distance / 1000) + (vehicles / 10)
                    
                    # Calculate scores for all available datasets
                    for json_file in json_files:
                        name = json_file.stem
                        if name in bks_data:
                            difficulty_scores[name] = get_difficulty_score(name, bks_data[name])
                        else:
                            # If no BKS data, use a default score
                            difficulty_scores[name] = 999.0
                    
                    # Sort by difficulty (easiest first)
                    dataset_names = sorted(
                        [f.stem for f in json_files],
                        key=lambda x: difficulty_scores.get(x, 999.0)
                    )
                except Exception:
                    # Fallback to alphabetical if BKS loading fails
                    dataset_names = sorted([f.stem for f in json_files])
            else:
                # Fallback to alphabetical if BKS file doesn't exist
                dataset_names = sorted([f.stem for f in json_files])
            
            # Create display names with difficulty indicators
            display_options = []
            for name in dataset_names:
                if name in difficulty_scores:
                    score = difficulty_scores[name]
                    # Determine type
                    if name.startswith('C'):
                        dtype = "C (Clustered - Easier)"
                    elif name.startswith('RC'):
                        dtype = "RC (Mixed - Hardest)"
                    else:
                        dtype = "R (Random - Medium)"
                    
                    # Add difficulty indicator
                    if score < 1.5:
                        diff_label = "🟢 Easy"
                    elif score < 2.0:
                        diff_label = "🟡 Medium"
                    elif score < 2.5:
                        diff_label = "🟠 Hard"
                    else:
                        diff_label = "🔴 Very Hard"
                    
                    display_options.append(f"{name} - {dtype} - {diff_label}")
                else:
                    display_options.append(name)
            
            selected_display = st.selectbox(
                "Select a Solomon instance (sorted easiest → hardest):",
                options=display_options,
                help="Datasets are sorted by difficulty. C=Clustered (easier), R=Random (medium), RC=Mixed (hardest)"
            )
            
            # Extract dataset name from display string
            selected_dataset = selected_display.split(" - ")[0] if " - " in selected_display else selected_display
            
            if st.button("Load Dataset", use_container_width=True):
                try:
                    data_service = DataService()
                    dataset_path = solomon_dir / f"{selected_dataset}.json"
                    
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        import json
                        data_dict = json.load(f)
                    
                    _update_solomon_dataset(data_dict)
                    st.session_state['solomon_dataset_name'] = selected_dataset
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

                    # CRITICAL: Apply post-GA 2-opt and TW repair (same as main.py)
                    try:
                        logger.info("Applying post-GA 2-opt and TW repair...")
                        from src.algorithms.local_search import TwoOptOptimizer
                        from src.algorithms.tw_repair import TWRepairOperator
                        from src.algorithms.decoder import RouteDecoder

                        problem = st.session_state.solomon_problem
                        tw_cfg = ga_config.get('tw_repair', {})

                        # 2-opt optimization
                        if not ga_config.get('no_local_search', False):
                            logger.info("Applying 2-opt local search to final solution...")
                            local_searcher = TwoOptOptimizer(problem)
                            optimized_routes = local_searcher.optimize_routes(best_solution.routes)
                            best_solution.routes = optimized_routes

                            # Recalculate distance after 2-opt
                            decoder = RouteDecoder(problem)
                            best_solution.chromosome = decoder.encode_routes(optimized_routes)
                            total_distance = 0.0
                            for route in optimized_routes:
                                if not route:
                                    continue
                                for i in range(len(route) - 1):
                                    total_distance += problem.get_distance(route[i], route[i + 1])
                            best_solution.total_distance = total_distance
                            logger.info(f"2-opt completed: new distance = {best_solution.total_distance:.2f}")

                        # Post-2opt TW repair with high violation_weight
                        if tw_cfg.get('enabled', True):
                            logger.info("Post-2opt TW repair: Checking for time window violations...")
                            tw_repair = TWRepairOperator(
                                problem,
                                max_iterations=tw_cfg.get('max_iterations', 500),  # Increased for better results
                                violation_weight=10000.0,  # CRITICAL: High weight to ensure 0 violations
                                max_relocations_per_route=tw_cfg.get('max_relocations_per_route', 3),
                                max_routes_to_try=len(best_solution.routes),
                                max_positions_to_try=tw_cfg.get('max_positions_to_try', None),
                                max_iterations_soft=tw_cfg.get('max_iterations_soft'),
                                max_routes_soft_limit=tw_cfg.get('max_routes_soft_limit'),
                                max_positions_soft_limit=tw_cfg.get('max_positions_soft_limit'),
                                lateness_soft_threshold=tw_cfg.get('lateness_soft_threshold'),
                                lateness_skip_threshold=tw_cfg.get('lateness_skip_threshold'),
                                # Enhanced parameters for better violation resolution
                                enable_exhaustive_search=True,
                                enable_restart=True,
                                restart_after_no_improvement=50,
                                adaptive_weight_multiplier=10.0,
                            )
                            repaired_routes = tw_repair.repair_routes(best_solution.routes)
                            best_solution.routes = repaired_routes
                            best_solution.chromosome = decoder.encode_routes(repaired_routes)
                            total_distance = 0.0
                            for route in repaired_routes:
                                if not route:
                                    continue
                                for i in range(len(route) - 1):
                                    total_distance += problem.get_distance(route[i], route[i + 1])
                            best_solution.total_distance = total_distance
                            logger.info(f"Post-2opt TW repair completed: distance = {best_solution.total_distance:.2f}")
                    except Exception as repair_err:
                        logger.warning(f"Post-GA optimization/repair failed: {repair_err}", exc_info=True)

                    # Get BKS distance and add to statistics before saving
                    bks_distance = None
                    try:
                        from src.evaluation.bks_validator import BKSValidator
                        dataset_metadata = (st.session_state.solomon_dataset or {}).get('metadata', {})
                        dataset_name = dataset_metadata.get('name', '')
                        if dataset_name:
                            bks_validator = BKSValidator()
                            bks_data = bks_validator.get_bks(dataset_name)
                            if bks_data:
                                bks_distance = bks_data.get('distance')
                                if bks_distance:
                                    # Add BKS to statistics for history saving
                                    statistics['bks_distance'] = bks_distance
                    except Exception as e:
                        logger.warning(f"Could not load BKS data for history: {str(e)}")
                    
                    st.session_state.solomon_optimization_results = {
                        'solution': best_solution,
                        'statistics': statistics,
                        'evolution_data': evolution_data,
                        'config': ga_config
                    }
                    st.session_state.solomon_optimization_running = False
                    
                    # Save to history
                    try:
                        history_service = HistoryService()
                        
                        # Get dataset name from metadata or use default
                        dataset_metadata = (st.session_state.solomon_dataset or {}).get('metadata', {})
                        dataset_name = dataset_metadata.get('name', st.session_state.get('solomon_dataset_name', 'Solomon Dataset'))
                        
                        # Try to find existing dataset or create new one
                        from app.config.database import SessionLocal
                        from app.database.crud import get_datasets, create_dataset
                        db = SessionLocal()
                        try:
                            # Find dataset by name (check all types first, then filter by type)
                            datasets_all = get_datasets(db, type=None)  # Get all datasets
                            dataset_id = None
                            
                            # Find dataset by name (prefer exact match with 'solomon' type)
                            for ds in datasets_all:
                                if ds.name == dataset_name and ds.type == 'solomon':
                                    dataset_id = ds.id
                                    logger.info(f"Found dataset {dataset_name} with type 'solomon' (ID: {dataset_id})")
                                    break
                            
                            # If not found, try searching only 'solomon' type
                            if not dataset_id:
                                datasets = get_datasets(db, type='solomon')
                                for ds in datasets:
                                    if ds.name == dataset_name:
                                        dataset_id = ds.id
                                        logger.info(f"Found dataset {dataset_name} with type 'solomon' (ID: {dataset_id})")
                                        break
                            
                            if not dataset_id:
                                # Create new dataset entry
                                dataset_json = json.dumps(st.session_state.solomon_dataset)
                                dataset = create_dataset(
                                    db,
                                    name=dataset_name,
                                    description=f"Solomon benchmark: {dataset_name}",
                                    type='solomon',
                                    data_json=dataset_json
                                )
                                dataset_id = dataset.id
                        finally:
                            db.close()
                        
                        # Save result - CRITICAL: Always save, even if best result update fails
                        logger.info(f"🔵 Calling save_result: dataset_id={dataset_id}, dataset_name={dataset_name}")
                        print(f"🔵 [SOLOMON] Calling save_result for dataset: {dataset_name}")
                        
                        run_id, is_new_best = history_service.save_result(
                            dataset_id=dataset_id,
                            dataset_name=dataset_name,
                            solution=best_solution,
                            statistics=statistics,
                            config=ga_config,
                            dataset_type="solomon"
                        )
                        
                        if run_id:
                            logger.info(f"✅ Saved optimization run to history: run_id={run_id}")
                            print(f"✅ [SOLOMON] Run saved: run_id={run_id}")
                            if is_new_best:
                                st.session_state['new_best_result'] = True
                                st.success(f"✅ Đã lưu vào history và đây là kết quả tốt nhất! (Run ID: {run_id})")
                            else:
                                st.success(f"✅ Đã lưu vào history (Run ID: {run_id})")
                        else:
                            logger.error("❌ CRITICAL: Failed to save to history: run_id is None")
                            print(f"❌ [SOLOMON] save_result returned None - check logs for errors")
                            st.error(f"❌ Không thể lưu vào history - run_id is None. Kiểm tra logs để xem lỗi.")
                    except Exception as e:
                        logger.error(f"❌ CRITICAL: Failed to save to history: {e}", exc_info=True)
                        print(f"❌ [SOLOMON] Exception saving to history: {e}")
                        import traceback
                        traceback.print_exc()
                        st.error(f"❌ Lỗi khi lưu vào history: {str(e)}")
                    
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
