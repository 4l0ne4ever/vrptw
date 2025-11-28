"""
Home/Landing page for VRP-GA Web Application.
"""

import streamlit as st

st.set_page_config(
    page_title="Home - VRP-GA",
    page_icon=None,
    layout="wide"
)

st.title("Welcome to VRP-GA Optimization System")

st.markdown("""
## About This Application

The VRP-GA Optimization System helps you solve **Vehicle Routing Problems (VRP)** 
using advanced **Genetic Algorithms (GA)** with the following features:

### Key Features

- **Genetic Algorithm Optimization**: Advanced GA with Split Algorithm support
- **Interactive Maps**: Real-time route visualization with Folium
- **Performance Analytics**: Comprehensive metrics and KPIs
- **Data Persistence**: Save and manage datasets and results
- **Evolution Tracking**: Monitor GA convergence and improvement
- **BKS Validation**: Compare with Best-Known Solutions (Solomon datasets)

### Quick Start

1. **Choose Your Mode**:
   - **Hanoi Mode**: Real-world delivery optimization with actual coordinates

2. **Upload or Load Data**:
   - Upload JSON dataset files
   - Load previously saved datasets
   - Use sample datasets for testing

3. **Configure Parameters**:
   - Adjust GA parameters (population, generations, etc.)
   - Choose optimization presets (Fast, Balanced, Best Quality)

4. **Run Optimization**:
   - Start GA evolution
   - Monitor progress in real-time
   - View live route updates

5. **Analyze Results**:
   - Interactive maps with route visualization
   - Detailed metrics and KPIs
   - Export results in multiple formats

### Navigation

Use the sidebar to navigate between:
- **Home**: This page
- **Hanoi Mode**: Real-world delivery optimization
- **History**: View past optimization runs
- **Datasets**: Manage your datasets
- **Help**: Documentation and guides

---

**Ready to get started?** Use the button below or navigate from the sidebar!
""")

if st.button("Start Hanoi Optimization", use_container_width=True, type="primary"):
    st.switch_page("pages/hanoi_mode.py")
