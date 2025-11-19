"""
Manual data entry components for Streamlit.
"""

import streamlit as st
from typing import Optional, Dict, List
import pandas as pd
from app.services.data_service import DataService
from app.utils.validators import validate_coordinates, validate_demand, validate_capacity, validate_num_vehicles
from app.utils.constants import HANOI_BOUNDS


def render_manual_entry_form(dataset_type: str = "hanoi_mockup") -> Optional[Dict]:
    """
    Render manual data entry form.
    
    Args:
        dataset_type: Type of dataset
        
    Returns:
        Data dictionary if form submitted and valid, None otherwise
    """
    st.subheader("Manual Data Entry")
    
    # Initialize session state for customers
    if 'manual_customers' not in st.session_state:
        st.session_state.manual_customers = []
    if 'manual_depot' not in st.session_state:
        st.session_state.manual_depot = None
    if 'manual_vehicle_capacity' not in st.session_state:
        st.session_state.manual_vehicle_capacity = 200
    if 'manual_num_vehicles' not in st.session_state:
        st.session_state.manual_num_vehicles = 5
    
    # Depot information
    st.markdown("### Depot Information")
    col1, col2 = st.columns(2)
    
    with col1:
        depot_lat = st.number_input(
            "Depot Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=21.0285,
            step=0.0001,
            format="%.6f",
            help=f"Recommended between {HANOI_BOUNDS['min_lat']} and {HANOI_BOUNDS['max_lat']}",
            key="depot_lat_input"
        )
    
    with col2:
        depot_lon = st.number_input(
            "Depot Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=105.8542,
            step=0.0001,
            format="%.6f",
            help=f"Recommended between {HANOI_BOUNDS['min_lon']} and {HANOI_BOUNDS['max_lon']}",
            key="depot_lon_input"
        )
    
    # Validate depot coordinates
    is_valid, error = validate_coordinates(depot_lat, depot_lon)
    if not is_valid:
        st.error(f"Depot coordinates: {error}")
    else:
        st.session_state.manual_depot = {
            'lat': depot_lat,
            'lon': depot_lon
        }
    
    # Vehicle configuration
    st.markdown("### Vehicle Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        vehicle_capacity = st.number_input(
            "Vehicle Capacity",
            min_value=1.0,
            value=float(st.session_state.manual_vehicle_capacity),
            step=10.0,
            format="%.1f"
        )
        st.session_state.manual_vehicle_capacity = vehicle_capacity
    
    with col2:
        num_vehicles = st.number_input(
            "Number of Vehicles",
            min_value=1,
            value=int(st.session_state.manual_num_vehicles),
            step=1
        )
        st.session_state.manual_num_vehicles = num_vehicles
    
    # Customer entry
    st.markdown("### Customer Entry")
    
    # Add customer button
    if st.button("Add Customer", type="primary"):
        # Add new customer to list
        new_customer = {
            'id': len(st.session_state.manual_customers) + 1,
            'lat': 21.0285,
            'lon': 105.8542,
            'demand': 10.0
        }
        st.session_state.manual_customers.append(new_customer)
        st.rerun()
    
    # Display and edit customers
    if st.session_state.manual_customers:
        st.markdown(f"**{len(st.session_state.manual_customers)} customer(s) added**")
        
        for i, customer in enumerate(st.session_state.manual_customers):
            with st.expander(f"Customer {customer['id']}", expanded=False):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    lat = st.number_input(
                        "Latitude",
                        key=f"lat_{i}",
                        min_value=-90.0,
                        max_value=90.0,
                        value=float(customer['lat']),
                        step=0.0001,
                        format="%.6f"
                    )
                    customer['lat'] = lat
                
                with col2:
                    lon = st.number_input(
                        "Longitude",
                        key=f"lon_{i}",
                        min_value=-180.0,
                        max_value=180.0,
                        value=float(customer['lon']),
                        step=0.0001,
                        format="%.6f"
                    )
                    customer['lon'] = lon
                
                with col3:
                    demand = st.number_input(
                        "Demand",
                        key=f"demand_{i}",
                        min_value=0.1,
                        value=float(customer['demand']),
                        step=1.0,
                        format="%.1f"
                    )
                    customer['demand'] = demand
                
                with col4:
                    st.write("")  # Spacer
                    if st.button("Delete", key=f"delete_{i}", type="secondary"):
                        st.session_state.manual_customers.pop(i)
                        st.rerun()
                
                # Validate customer coordinates
                is_valid, error = validate_coordinates(lat, lon)
                if not is_valid:
                    st.warning(f"Coordinates: {error}")
                
                is_valid, error = validate_demand(demand)
                if not is_valid:
                    st.warning(f"Demand: {error}")
        
        # Coverage settings
        radius_limit_km = st.slider(
            "Maximum Distance from Depot (km)",
            min_value=25,
            max_value=50,
            value=35,
            step=5,
            help="Customers must be within this distance from depot",
            key="radius_limit_slider"
        )
        
        # Preview map with click functionality
        if st.session_state.manual_depot:
            try:
                import folium
                from streamlit_folium import st_folium
                import math
                
                # Create map
                m = folium.Map(
                    location=[st.session_state.manual_depot['lat'], st.session_state.manual_depot['lon']],
                    zoom_start=11
                )
                
                # Add depot
                folium.Marker(
                    [st.session_state.manual_depot['lat'], st.session_state.manual_depot['lon']],
                    popup="Depot",
                    tooltip="Depot",
                    icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
                ).add_to(m)
                
                # Add radius circle
                folium.Circle(
                    location=[st.session_state.manual_depot['lat'], st.session_state.manual_depot['lon']],
                    radius=radius_limit_km * 1000,  # Convert km to meters
                    popup=f"Service Area ({radius_limit_km}km)",
                    color='green',
                    fill=False,
                    weight=2,
                    opacity=0.7,
                    interactive=False
                ).add_to(m)
                
                # Add customers
                for customer in st.session_state.manual_customers:
                    # Calculate distance from depot
                    def haversine_distance(lat1, lon1, lat2, lon2):
                        """Calculate distance in km between two GPS coordinates."""
                        R = 6371.0  # Earth radius in km
                        dlat = math.radians(lat2 - lat1)
                        dlon = math.radians(lon2 - lon1)
                        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
                        c = 2 * math.asin(math.sqrt(a))
                        return R * c
                    
                    distance = haversine_distance(
                        st.session_state.manual_depot['lat'],
                        st.session_state.manual_depot['lon'],
                        customer['lat'],
                        customer['lon']
                    )
                    
                    # Color based on distance
                    if distance > radius_limit_km:
                        color = 'red'
                        icon = 'exclamation-triangle'
                    else:
                        color = 'blue'
                        icon = 'user'
                    
                    folium.Marker(
                        [customer['lat'], customer['lon']],
                        popup=f"Customer {customer['id']}<br>Demand: {customer['demand']}<br>Distance: {distance:.2f}km",
                        tooltip=f"Customer {customer['id']} ({distance:.1f}km)",
                        icon=folium.Icon(color=color, icon=icon, prefix='fa')
                    ).add_to(m)
                    
                    # Show warning if outside radius
                    if distance > radius_limit_km:
                        st.warning(f"Customer {customer['id']} is {distance:.2f}km from depot (limit: {radius_limit_km}km)")
                
                # Display map and handle clicks
                st.markdown("**Click on map to add customer at that location**")
                
                # Use unique key for map to avoid duplicate key errors
                map_key = f"manual_entry_map_{len(st.session_state.manual_customers)}"
                map_data = st_folium(m, width=700, height=500, key=map_key, returned_objects=["last_clicked"])
                
                # Handle map click - check if there's a new click
                click_key = f"last_map_click_{len(st.session_state.manual_customers)}"
                if click_key not in st.session_state:
                    st.session_state[click_key] = None
                
                if map_data and map_data.get("last_clicked"):
                    clicked_lat = map_data["last_clicked"].get("lat")
                    clicked_lon = map_data["last_clicked"].get("lng")
                    
                    # Check if this is a new click (different from last one)
                    last_click = st.session_state.get(click_key)
                    if last_click is None or (last_click.get("lat") != clicked_lat or last_click.get("lng") != clicked_lon):
                        # Calculate distance from depot
                        def haversine_distance(lat1, lon1, lat2, lon2):
                            R = 6371.0
                            dlat = math.radians(lat2 - lat1)
                            dlon = math.radians(lon2 - lon1)
                            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
                            c = 2 * math.asin(math.sqrt(a))
                            return R * c
                        
                        distance = haversine_distance(
                            st.session_state.manual_depot['lat'],
                            st.session_state.manual_depot['lon'],
                            clicked_lat,
                            clicked_lon
                        )
                        
                        if distance > radius_limit_km:
                            st.error(f"Location is {distance:.2f}km from depot (limit: {radius_limit_km}km). Please choose a location closer to depot.")
                        else:
                            # Add new customer at clicked location
                            new_customer = {
                                'id': len(st.session_state.manual_customers) + 1,
                                'lat': clicked_lat,
                                'lon': clicked_lon,
                                'demand': 10.0
                            }
                            st.session_state.manual_customers.append(new_customer)
                            st.session_state[click_key] = {"lat": clicked_lat, "lng": clicked_lon}
                            st.success(f"Customer {new_customer['id']} added at clicked location ({distance:.2f}km from depot)")
                            st.rerun()
                
            except ImportError:
                st.info("Folium not available for map preview")
            except Exception as e:
                st.warning(f"Could not create map preview: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
        
        # Create dataset button
        if st.button("Create Dataset", type="primary", use_container_width=True):
            # Validate all inputs
            errors = []
            
            if not st.session_state.manual_depot:
                errors.append("Depot coordinates required")
            
            if not st.session_state.manual_customers:
                errors.append("At least one customer required")
            
            # Validate depot
            if st.session_state.manual_depot:
                is_valid, error = validate_coordinates(
                    st.session_state.manual_depot['lat'],
                    st.session_state.manual_depot['lon']
                )
                if not is_valid:
                    errors.append(f"Depot: {error}")
            
            # Validate customers
            for i, customer in enumerate(st.session_state.manual_customers):
                is_valid, error = validate_coordinates(customer['lat'], customer['lon'])
                if not is_valid:
                    errors.append(f"Customer {customer['id']} coordinates: {error}")
                
                is_valid, error = validate_demand(customer['demand'])
                if not is_valid:
                    errors.append(f"Customer {customer['id']} demand: {error}")
            
            # Validate vehicle config
            is_valid, error = validate_capacity(vehicle_capacity)
            if not is_valid:
                errors.append(f"Vehicle capacity: {error}")
            
            is_valid, error = validate_num_vehicles(num_vehicles)
            if not is_valid:
                errors.append(f"Number of vehicles: {error}")
            
            if errors:
                for error in errors:
                    st.error(error)
                return None
            
            # Create data dictionary
            depot = {
                'id': 0,
                'x': st.session_state.manual_depot['lon'],
                'y': st.session_state.manual_depot['lat'],
                'demand': 0,
                'ready_time': 0.0,
                'due_date': 1000.0,
                'service_time': 0.0
            }
            
            customers = []
            for customer in st.session_state.manual_customers:
                customers.append({
                    'id': customer['id'],
                    'x': customer['lon'],
                    'y': customer['lat'],
                    'demand': customer['demand'],
                    'ready_time': 0.0,
                    'due_date': 1000.0,
                    'service_time': 10.0
                })
            
            data_dict = {
                'depot': depot,
                'customers': customers,
                'vehicle_capacity': vehicle_capacity,
                'num_vehicles': num_vehicles,
                'metadata': {
                    'name': 'Manually Entered Dataset',
                    'source': 'manual_entry',
                    'format': dataset_type,
                    'num_customers': len(customers)
                },
                'problem_config': {
                    'vehicle_capacity': vehicle_capacity,
                    'num_vehicles': num_vehicles,
                    'traffic_factor': 1.0
                }
            }
            
            st.success("Dataset created successfully!")
            return data_dict
    
    else:
        st.info("Click 'Add Customer' to start entering customer data.")
    
    return None


def clear_manual_entry():
    """Clear manual entry form data."""
    if 'manual_customers' in st.session_state:
        del st.session_state.manual_customers
    if 'manual_depot' in st.session_state:
        del st.session_state.manual_depot
    if 'manual_vehicle_capacity' in st.session_state:
        del st.session_state.manual_vehicle_capacity
    if 'manual_num_vehicles' in st.session_state:
        del st.session_state.manual_num_vehicles

