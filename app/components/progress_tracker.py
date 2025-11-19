"""
Progress tracking components for optimization.
"""

import streamlit as st
from typing import Dict, Optional
import time


def render_progress_tracker(
    current_gen: int,
    max_gens: int,
    best_fitness: float,
    best_distance: float,
    is_running: bool = True
):
    """
    Render progress tracking UI.
    
    Args:
        current_gen: Current generation number
        max_gens: Maximum generations
        best_fitness: Current best fitness
        best_distance: Current best distance
        is_running: Whether optimization is running
    """
    st.subheader("Optimization Progress")
    
    # Progress bar
    progress = current_gen / max_gens if max_gens > 0 else 0
    st.progress(progress)
    st.caption(f"Generation {current_gen} of {max_gens} ({progress*100:.1f}%)")
    
    # Current metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Best Fitness", f"{best_fitness:.4f}")
    
    with col2:
        st.metric("Best Distance", f"{best_distance:.2f}")
    
    # Status
    if is_running:
        st.info("Optimization in progress...")
    else:
        st.success("Optimization completed!")


def render_optimization_controls(
    on_run: callable,
    on_stop: callable,
    is_running: bool = False,
    key_prefix: str = "opt"
):
    """
    Render optimization control buttons.
    
    Args:
        on_run: Callback for run button
        on_stop: Callback for stop button
        is_running: Whether optimization is currently running
        key_prefix: Prefix for button keys to avoid duplicates
    """
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if is_running:
            if st.button("Stop Optimization", type="secondary", use_container_width=True, key=f"{key_prefix}_stop_btn"):
                on_stop()
        else:
            if st.button("Run Optimization", type="primary", use_container_width=True, key=f"{key_prefix}_run_btn"):
                on_run()
    
    with col2:
        if is_running:
            st.warning("Optimization is running. Click 'Stop' to terminate early.")

