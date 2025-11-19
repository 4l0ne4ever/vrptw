"""
Help and documentation page.
"""

import streamlit as st

st.set_page_config(
    page_title="Help - VRP-GA",
    page_icon=None,
    layout="wide"
)

st.title("Help & Documentation")

st.markdown("""
## Getting Started

### Quick Start Guide

1. **Choose Your Mode**
   - Select Hanoi Mode for real-world optimization
   - Select Solomon Mode for academic benchmarking

2. **Prepare Your Data**
   - Upload JSON dataset files
   - Or use sample datasets

3. **Configure Parameters**
   - Adjust GA settings
   - Choose optimization presets

4. **Run Optimization**
   - Start the GA evolution
   - Monitor progress

5. **Analyze Results**
   - View interactive maps
   - Review metrics
   - Export results

## Documentation

- **Architecture**: See `docs/ARCHITECTURE.md`
- **User Guide**: See `docs/USER_GUIDE.md` (coming soon)
- **API Reference**: See `docs/API_REFERENCE.md` (coming soon)

## Support

For issues or questions, please refer to the documentation or contact support.

---

**Version**: 1.0.0 (Development)
""")
