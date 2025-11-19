"""
Data input components for Streamlit.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Tuple
from app.services.data_service import DataService
from app.services.database_service import DatabaseService
from app.core.exceptions import ValidationError, DatasetError


def render_file_uploader(dataset_type: str = "hanoi_mockup") -> Optional[Dict]:
    """
    Render file uploader component.
    
    Args:
        dataset_type: Type of dataset expected
        
    Returns:
        Data dictionary if file uploaded and valid, None otherwise
    """
    st.subheader("Upload Dataset File")
    
    # Show file format help
    with st.expander("📋 File Format Help"):
        st.markdown("""
        **Supported formats:** JSON, CSV, Excel (.xlsx, .xls)
        
        **Required columns:**
        - `id`: Customer/Depot ID (0 for depot, >0 for customers)
        - `x`: Longitude (for Hanoi: 105.3 - 106.0)
        - `y`: Latitude (for Hanoi: 20.7 - 21.4)
        - `demand`: Customer demand (must be positive)
        
        **Optional columns:**
        - `ready_time`: Ready time (default: 0)
        - `due_date`: Due date (default: 1000)
        - `service_time`: Service time (default: 10)
        
        **Example CSV:**
        ```
        id,x,y,demand
        0,105.8542,21.0285,0
        1,105.8400,21.0200,10
        2,105.8500,21.0300,15
        ```
        """)
    
    uploaded_file = st.file_uploader(
        "Choose a file (JSON, CSV, or Excel)",
        type=['json', 'csv', 'xlsx', 'xls'],
        help="Upload a VRP dataset in JSON, CSV, or Excel format"
    )
    
    if uploaded_file is not None:
        data_service = DataService()
        
        # Parse and validate
        success, error_msg, data_dict = data_service.parse_uploaded_file(
            uploaded_file, dataset_type
        )
        
        if success:
            st.success("File uploaded and validated successfully!")
            return data_dict
        else:
            st.error(f"Validation error: {error_msg}")
            return None
    
    return None


def render_dataset_preview(data_dict: Dict, dataset_type: str = "hanoi_mockup"):
    """
    Render dataset preview with statistics and map.
    
    Args:
        data_dict: Validated data dictionary
        dataset_type: Type of dataset
    """
    data_service = DataService()
    stats = data_service.generate_preview_statistics(data_dict)
    
    st.subheader("Dataset Preview")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customers", stats['num_customers'])
    
    with col2:
        st.metric("Total Demand", f"{stats['total_demand']:.1f}")
    
    with col3:
        st.metric("Vehicle Capacity", stats['vehicle_capacity'])
    
    with col4:
        st.metric("Vehicles", stats['num_vehicles'])
    
    # Map preview (for Hanoi datasets)
    if dataset_type == "hanoi_mockup":
        try:
            import folium
            from streamlit_folium import st_folium
            
            # Create map centered on depot
            depot = data_dict['depot']
            m = folium.Map(
                location=[depot['y'], depot['x']],  # [lat, lon]
                zoom_start=12
            )
            
            # Add depot marker with unique key
            depot_marker = folium.Marker(
                [depot['y'], depot['x']],
                popup=f"Depot<br>Capacity: {stats['vehicle_capacity']}",
                tooltip="Depot",
                icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
            )
            depot_marker.add_to(m)
            
            # Add customer markers with unique keys
            for idx, customer in enumerate(data_dict['customers']):
                customer_marker = folium.Marker(
                    [customer['y'], customer['x']],
                    popup=f"Customer {customer['id']}<br>Demand: {customer['demand']}",
                    tooltip=f"Customer {customer['id']}",
                    icon=folium.Icon(color='blue', icon='user', prefix='fa')
                )
                customer_marker.add_to(m)
            
            # Use unique key for map to avoid duplicate key errors
            import hashlib
            map_key = hashlib.md5(str(data_dict.get('metadata', {}).get('name', 'preview')).encode()).hexdigest()[:16]
            st_folium(m, width=700, height=400, key=f"preview_map_{map_key}")
            
        except ImportError:
            st.info("Folium not available for map preview")
        except Exception as e:
            st.warning(f"Could not create map preview: {e}")
    
    # Data table
    st.subheader("Customer Details")
    customers_data = []
    for customer in data_dict['customers'][:20]:  # Show first 20
        customers_data.append({
            'ID': customer['id'],
            'X (Lon)': f"{customer['x']:.6f}",
            'Y (Lat)': f"{customer['y']:.6f}",
            'Demand': customer['demand']
        })
    
    if customers_data:
        df = pd.DataFrame(customers_data)
        st.dataframe(df, use_container_width=True)
        
        if len(data_dict['customers']) > 20:
            st.info(f"Showing first 20 of {len(data_dict['customers'])} customers")


def render_save_dataset_form(data_dict: Dict, dataset_type: str) -> Optional[int]:
    """
    Render form to save dataset to database.
    
    Args:
        data_dict: Validated data dictionary
        dataset_type: Type of dataset
        
    Returns:
        Dataset ID if saved, None otherwise
    """
    st.subheader("Save Dataset")
    
    with st.form("save_dataset_form"):
        name = st.text_input("Dataset Name", value=data_dict.get('metadata', {}).get('name', ''))
        description = st.text_area("Description", value=data_dict.get('metadata', {}).get('description', ''))
        
        submitted = st.form_submit_button("Save to Database", use_container_width=True)
        
        if submitted:
            if not name:
                st.error("Dataset name is required")
                return None
            
            try:
                db_service = DatabaseService()
                dataset_id = db_service.save_dataset(
                    name=name,
                    description=description,
                    dataset_type=dataset_type,
                    data_dict=data_dict,
                    metadata=data_dict.get('metadata')
                )
                
                st.success(f"Dataset saved successfully! (ID: {dataset_id})")
                return dataset_id
                
            except Exception as e:
                st.error(f"Error saving dataset: {str(e)}")
                return None
    
    return None


def render_load_saved_datasets(dataset_type: Optional[str] = None) -> Optional[Dict]:
    """
    Render component to load saved datasets.
    
    Args:
        dataset_type: Optional filter by type
        
    Returns:
        Selected dataset data dictionary or None
    """
    st.subheader("Load Saved Dataset")
    
    try:
        db_service = DatabaseService()
        datasets = db_service.get_all_datasets(dataset_type=dataset_type)
        
        if not datasets:
            st.info("No saved datasets found. Upload a dataset first.")
            return None
        
        # Create selection options
        dataset_options = {
            f"{d['name']} ({d['num_customers']} customers)": d['id']
            for d in datasets
        }
        
        selected = st.selectbox(
            "Select a dataset",
            options=list(dataset_options.keys()),
            help="Choose a previously saved dataset"
        )
        
        if selected and st.button("Load Dataset", use_container_width=True):
            dataset_id = dataset_options[selected]
            data_dict = db_service.load_dataset(dataset_id)
            
            if data_dict:
                st.success(f"Dataset loaded successfully!")
                return data_dict
            else:
                st.error("Failed to load dataset")
                return None
        
        return None
        
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return None


def render_sample_data_button(dataset_type: str = "hanoi_mockup") -> Optional[Dict]:
    """
    Render button to use sample data.
    
    Args:
        dataset_type: Type of dataset
        
    Returns:
        Sample data dictionary if button clicked
    """
    st.subheader("Quick Start")
    
    # Use session state to track if button was clicked
    button_key = f"sample_data_clicked_{dataset_type}"
    if button_key not in st.session_state:
        st.session_state[button_key] = False
    
    if st.button("Use Sample Data", use_container_width=True, type="primary", key=f"sample_btn_{dataset_type}"):
        st.session_state[button_key] = True
        st.rerun()
    
    # Generate sample data if button was clicked
    if st.session_state.get(button_key, False):
        data_service = DataService()
        
        try:
            if dataset_type == "hanoi_mockup":
                sample_data = data_service.create_sample_hanoi_dataset(n_customers=10)
                st.success("Sample dataset generated!")
                # Reset button state
                st.session_state[button_key] = False
                return sample_data
            else:
                st.info("Sample data for Solomon mode coming soon")
                st.session_state[button_key] = False
                return None
        except Exception as e:
            st.error(f"Error generating sample data: {str(e)}")
            st.session_state[button_key] = False
            return None
    
    return None

