"""
Main Streamlit application entry point for VRP-GA Web Application.
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import from optimize codebase
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="VRP-GA Optimization System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_dataset = None
    st.session_state.current_problem = None
    st.session_state.optimization_running = False
    st.session_state.optimization_results = None

# Main app
def main():
    """Main application entry point."""
    st.title("VRP-GA Optimization System")
    st.markdown("---")
    
    st.info("""
    Welcome to the VRP-GA Optimization System!
    
    This application helps you solve Vehicle Routing Problems using Genetic Algorithms.
    
    **Features:**
    - Optimize delivery routes for Hanoi (real coordinates)
    - Benchmark with Solomon datasets (academic)
    - View interactive maps and detailed analytics
    - Save and compare optimization results
    """)
    
    st.markdown("### Get Started")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Hanoi Mode**
        - Real-world coordinates
        - Interactive Folium maps
        - Ahamove shipping cost calculation
        """)
        if st.button("Start Hanoi Optimization", use_container_width=True):
            st.switch_page("pages/hanoi_mode.py")
    
    with col2:
        st.markdown("""
        **Solomon Mode**
        - Academic benchmarks
        - BKS validation
        - Performance charts
        """)
        if st.button("Start Solomon Optimization", use_container_width=True):
            st.switch_page("pages/solomon_mode.py")
    
    st.markdown("---")
    st.markdown("### Quick Links")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("View History", use_container_width=True):
            st.switch_page("pages/history.py")
    
    with col2:
        if st.button("Manage Datasets", use_container_width=True):
            st.switch_page("pages/datasets.py")
    
    with col3:
        if st.button("Help & Documentation", use_container_width=True):
            st.switch_page("pages/help.py")

if __name__ == "__main__":
    main()

