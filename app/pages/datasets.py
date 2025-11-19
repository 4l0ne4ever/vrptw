"""
Datasets management page.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.database_service import DatabaseService
from app.core.exceptions import DatabaseError

st.set_page_config(
    page_title="Datasets - VRP-GA",
    page_icon=None,
    layout="wide"
)

st.title("Dataset Management")

st.info("Manage your saved datasets. View, delete, or load datasets for optimization.")

# Initialize database service
db_service = DatabaseService()

# Filter options
col1, col2 = st.columns([3, 1])

with col1:
    search_term = st.text_input("Search datasets", placeholder="Search by name...")

with col2:
    filter_type = st.selectbox(
        "Filter by type",
        options=["All", "hanoi_mockup", "solomon"],
        index=0
    )

# Get datasets
try:
    dataset_type = None if filter_type == "All" else filter_type
    all_datasets = db_service.get_all_datasets(dataset_type=dataset_type)
    
    # Filter by search term
    if search_term:
        all_datasets = [
            d for d in all_datasets
            if search_term.lower() in d['name'].lower()
        ]
    
    if not all_datasets:
        st.info("No datasets found. Upload a dataset from Hanoi Mode or Solomon Mode to get started.")
    else:
        st.markdown(f"### Found {len(all_datasets)} dataset(s)")
        
        # Display datasets in cards
        for dataset in all_datasets:
            with st.expander(f"**{dataset['name']}** - {dataset['num_customers']} customers ({dataset['type']})"):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**Description:** {dataset['description'] or 'No description'}")
                    if dataset['created_at']:
                        created_date = datetime.fromisoformat(dataset['created_at']).strftime('%Y-%m-%d %H:%M:%S')
                        st.write(f"**Created:** {created_date}")
                
                with col2:
                    if st.button("Load", key=f"load_{dataset['id']}", use_container_width=True):
                        # Store in session state for use in other pages
                        try:
                            data_dict = db_service.load_dataset(dataset['id'])
                            st.session_state.hanoi_dataset = data_dict
                            st.session_state.hanoi_dataset_id = dataset['id']
                            st.success(f"Dataset '{dataset['name']}' loaded!")
                            st.info("Navigate to Hanoi Mode or Solomon Mode to use this dataset.")
                        except Exception as e:
                            st.error(f"Error loading dataset: {str(e)}")
                
                with col3:
                    if st.button("Delete", key=f"delete_{dataset['id']}", use_container_width=True):
                        try:
                            db_service.delete_dataset(dataset['id'])
                            st.success(f"Dataset '{dataset['name']}' deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting dataset: {str(e)}")
        
        # Summary statistics
        st.markdown("---")
        st.markdown("### Summary")
        col1, col2, col3 = st.columns(3)
        
        hanoi_count = sum(1 for d in all_datasets if d['type'] == 'hanoi_mockup')
        solomon_count = sum(1 for d in all_datasets if d['type'] == 'solomon')
        total_customers = sum(d['num_customers'] for d in all_datasets)
        
        with col1:
            st.metric("Total Datasets", len(all_datasets))
        
        with col2:
            st.metric("Hanoi Datasets", hanoi_count)
        
        with col3:
            st.metric("Solomon Datasets", solomon_count)

except DatabaseError as e:
    st.error(f"Database error: {str(e)}")
except Exception as e:
    st.error(f"Error loading datasets: {str(e)}")
