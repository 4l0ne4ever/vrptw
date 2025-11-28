"""
Help and documentation page.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

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

## View Logs

View application logs directly in the browser for debugging.

**Log Location**: `logs/app_YYYYMMDD.log`

### Recent Logs
""")

# Log viewer section
log_dir = Path("logs")
today = datetime.now().strftime("%Y%m%d")
log_file = log_dir / f"app_{today}.log"

if log_file.exists():
    with st.expander("📋 View Today's Logs", expanded=False):
        # Options
        col1, col2 = st.columns(2)
        with col1:
            show_lines = st.number_input("Number of lines", min_value=10, max_value=1000, value=100, step=10)
        with col2:
            filter_text = st.text_input("Filter (e.g., TW_REPAIR)", value="", placeholder="Leave empty for all")
        
        # Read and display logs
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filter if needed
            if filter_text:
                filtered_lines = [line for line in lines if filter_text in line]
                display_lines = filtered_lines[-show_lines:] if len(filtered_lines) > show_lines else filtered_lines
                st.info(f"Showing {len(display_lines)} of {len(filtered_lines)} filtered lines")
            else:
                display_lines = lines[-show_lines:] if len(lines) > show_lines else lines
                st.info(f"Showing last {len(display_lines)} of {len(lines)} total lines")
            
            # Display logs
            log_content = "".join(display_lines)
            st.code(log_content, language=None)
            
            # Refresh button
            if st.button("🔄 Refresh Logs"):
                st.rerun()
                
        except Exception as e:
            st.error(f"Error reading log file: {e}")
else:
    st.info(f"Log file not found: {log_file}")
    st.caption("Logs will appear here after running optimizations.")

st.markdown("""
### Log Commands (Terminal)

For real-time log viewing, use these commands in a separate terminal:

```bash
# View logs in real-time
tail -f logs/app_$(date +%Y%m%d).log

# Find TW_REPAIR messages
grep 'TW_REPAIR' logs/app_$(date +%Y%m%d).log

# View last 50 lines
tail -50 logs/app_$(date +%Y%m%d).log
```

---

**Version**: 1.0.0 (Development)
""")
