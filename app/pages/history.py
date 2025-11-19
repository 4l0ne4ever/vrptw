"""
History page to view past optimization runs.
"""

import streamlit as st

st.set_page_config(
    page_title="History - VRP-GA",
    page_icon=None,
    layout="wide"
)

st.title("Optimization History")

st.info("View and manage your past optimization runs.")

st.markdown("### Coming Soon")
st.markdown("This page will be implemented in Phase 5 of development.")

# Placeholder for future implementation
st.markdown("""
**Planned Features:**
1. List of all saved optimization runs
2. Filter and search functionality
3. View detailed results
4. Compare multiple runs
5. Delete runs
""")
