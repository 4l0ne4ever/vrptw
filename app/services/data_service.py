"""
Data service for parsing, validating, and managing VRP datasets.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import io

# Add parent directory to path to import from optimize codebase
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.vrp_model import create_vrp_problem_from_dict, VRPProblem
from src.data_processing.distance import DistanceCalculator
from src.core.pipeline_profiler import pipeline_profiler
from app.utils.validators import validate_json_dataset
from app.core.exceptions import DatasetError
from app.core.logger import setup_app_logger
from config import VRP_CONFIG

logger = setup_app_logger()


class DataService:
    """Service for handling VRP dataset operations."""
    
    def __init__(self):
        """Initialize data service."""
        use_adaptive = VRP_CONFIG.get('use_adaptive_traffic', False)
        traffic_factor = VRP_CONFIG.get('traffic_factor', 1.0)
        # Will set dataset_type when creating problem
        self.distance_calculator = None  # Will be created per dataset type
        self.use_adaptive = use_adaptive
        self.traffic_factor = traffic_factor
    
    def parse_uploaded_file(self, uploaded_file, dataset_type: str = "hanoi_mockup") -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Parse and validate uploaded file (JSON, CSV, or Excel).
        
        Args:
            uploaded_file: Streamlit uploaded file object
            dataset_type: Type of dataset ("hanoi_mockup" or "solomon")
            
        Returns:
            Tuple of (success, error_message, data_dict)
        """
        try:
            # Get file extension
            file_name = uploaded_file.name.lower()
            
            if file_name.endswith('.json'):
                return self._parse_json_file(uploaded_file, dataset_type)
            elif file_name.endswith('.csv'):
                return self._parse_csv_file(uploaded_file, dataset_type)
            elif file_name.endswith(('.xlsx', '.xls')):
                return self._parse_excel_file(uploaded_file, dataset_type)
            else:
                return False, f"Unsupported file format. Supported: JSON, CSV, Excel (.xlsx, .xls)", None
            
        except Exception as e:
            logger.error(f"Error parsing uploaded file: {e}")
            return False, f"Error parsing file: {str(e)}", None
    
    def _parse_json_file(self, uploaded_file, dataset_type: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Parse JSON file."""
        try:
            file_content = uploaded_file.read()
            json_data = json.loads(file_content.decode('utf-8'))
            
            # Validate dataset
            is_valid, error_msg, validated_data = validate_json_dataset(json_data, dataset_type)
            
            if not is_valid:
                return False, error_msg, None
            
            return True, None, validated_data
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", None
        except Exception as e:
            return False, f"Error parsing JSON: {str(e)}", None
    
    def _parse_csv_file(self, uploaded_file, dataset_type: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Parse CSV file and convert to VRP format."""
        try:
            file_content = uploaded_file.read()
            
            # Try different encodings
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
            except:
                df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1')
            
            # Convert CSV to VRP format
            data_dict = self._csv_to_vrp_format(df, dataset_type)
            
            if data_dict is None:
                return False, "CSV format not recognized. Expected columns: id, x, y, demand (and optional: ready_time, due_date, service_time)", None
            
            # Validate dataset
            is_valid, error_msg, validated_data = validate_json_dataset(data_dict, dataset_type)
            
            if not is_valid:
                return False, error_msg, None
            
            return True, None, validated_data
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            return False, f"Error parsing CSV file: {str(e)}", None
    
    def _parse_excel_file(self, uploaded_file, dataset_type: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Parse Excel file and convert to VRP format."""
        try:
            file_content = uploaded_file.read()
            df = pd.read_excel(io.BytesIO(file_content))
            
            # Convert Excel to VRP format
            data_dict = self._csv_to_vrp_format(df, dataset_type)
            
            if data_dict is None:
                return False, "Excel format not recognized. Expected columns: id, x, y, demand (and optional: ready_time, due_date, service_time)", None
            
            # Validate dataset
            is_valid, error_msg, validated_data = validate_json_dataset(data_dict, dataset_type)
            
            if not is_valid:
                return False, error_msg, None
            
            return True, None, validated_data
            
        except Exception as e:
            logger.error(f"Error parsing Excel: {e}")
            return False, f"Error parsing Excel file: {str(e)}", None
    
    def _csv_to_vrp_format(self, df: pd.DataFrame, dataset_type: str) -> Optional[Dict]:
        """
        Convert CSV/Excel DataFrame to VRP format.
        
        Expected format:
        - id, x, y, demand (required)
        - ready_time, due_date, service_time (optional)
        - First row with id=0 is depot, others are customers
        """
        try:
            # Normalize column names (case insensitive, strip whitespace)
            df.columns = df.columns.str.strip().str.lower()
            
            # Check required columns
            required_cols = ['id', 'x', 'y', 'demand']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None
            
            # Separate depot and customers
            depot_row = df[df['id'] == 0]
            customer_rows = df[df['id'] != 0]
            
            if depot_row.empty:
                return None
            
            if customer_rows.empty:
                return None
            
            # Check for duplicate IDs
            customer_ids = customer_rows['id'].tolist()
            if len(customer_ids) != len(set(customer_ids)):
                logger.error("Duplicate customer IDs found")
                return None
            
            # Check for duplicate depot (should only be one)
            if len(depot_row) > 1:
                logger.error("Multiple depot entries found")
                return None
            
            # Extract depot
            depot = depot_row.iloc[0]
            depot_dict = {
                'id': int(depot['id']),
                'x': float(depot['x']),
                'y': float(depot['y']),
                'demand': float(depot.get('demand', 0)),
                'ready_time': float(depot.get('ready_time', 0.0)),
                'due_date': float(depot.get('due_date', 1000.0)),
                'service_time': float(depot.get('service_time', 0.0))
            }
            
            # Extract customers
            customers = []
            seen_coords = set()  # Track coordinates for duplicate detection
            
            for _, row in customer_rows.iterrows():
                customer_id = int(row['id'])
                x = float(row['x'])
                y = float(row['y'])
                demand = float(row['demand'])
                
                # Check for duplicate coordinates
                coord_key = (round(x, 6), round(y, 6))  # Round to avoid floating point issues
                if coord_key in seen_coords:
                    logger.warning(f"Duplicate coordinates detected for customer {customer_id}: ({x}, {y})")
                seen_coords.add(coord_key)
                
                # Check for unreasonably large demand (warn but allow)
                if demand > 10000:
                    logger.warning(f"Very large demand detected for customer {customer_id}: {demand}")
                
                customer_dict = {
                    'id': customer_id,
                    'x': x,
                    'y': y,
                    'demand': demand,
                    'ready_time': float(row.get('ready_time', 0.0)),
                    'due_date': float(row.get('due_date', 1000.0)),
                    'service_time': float(row.get('service_time', 10.0))
                }
                customers.append(customer_dict)
            
            # Get vehicle capacity and num_vehicles
            # Check if they're in a separate row or use defaults
            vehicle_capacity = 200.0
            num_vehicles = 5
            
            # Try to get from dataframe if exists as columns
            if 'vehicle_capacity' in df.columns:
                # Get first non-zero value or use default
                cap_values = df[df['vehicle_capacity'] > 0]['vehicle_capacity']
                if not cap_values.empty:
                    vehicle_capacity = float(cap_values.iloc[0])
            
            if 'num_vehicles' in df.columns:
                # Get first non-zero value or use default
                veh_values = df[df['num_vehicles'] > 0]['num_vehicles']
                if not veh_values.empty:
                    num_vehicles = int(veh_values.iloc[0])
            
            # If not in dataframe, calculate from total demand
            if 'vehicle_capacity' not in df.columns:
                total_demand = sum(c['demand'] for c in customers)
                num_vehicles = max(1, int(np.ceil(total_demand / vehicle_capacity)) + 2)
            
            return {
                'depot': depot_dict,
                'customers': customers,
                'vehicle_capacity': vehicle_capacity,
                'num_vehicles': num_vehicles,
                'metadata': {
                    'name': 'Imported Dataset',
                    'source': 'csv/excel',
                    'format': dataset_type,
                    'num_customers': len(customers)
                },
                'problem_config': {
                    'vehicle_capacity': vehicle_capacity,
                    'num_vehicles': num_vehicles,
                    'traffic_factor': 1.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error converting CSV/Excel to VRP format: {e}")
            return None
    
    def create_vrp_problem(self, data_dict: Dict, calculate_distance: bool = True, dataset_type: str = "hanoi") -> VRPProblem:
        """
        Create VRPProblem object from data dictionary.
        
        Args:
            data_dict: Validated data dictionary
            calculate_distance: Whether to calculate distance matrix
            dataset_type: Type of dataset ("hanoi" or "solomon") - affects routing method
            
        Returns:
            VRPProblem instance
        """
        try:
            with pipeline_profiler.profile(
                "data.create_vrp_problem",
                metadata={'calculate_distance': calculate_distance}
            ):
                # Ensure vehicle_capacity and num_vehicles are in data_dict
                # They might be in problem_config
                # For hanoi mode, always use 200 as specified
                if dataset_type and 'hanoi' in dataset_type.lower():
                    from config import VRP_CONFIG
                    data_dict['vehicle_capacity'] = VRP_CONFIG.get('vehicle_capacity', 200)
                elif 'vehicle_capacity' not in data_dict:
                    if 'problem_config' in data_dict and 'vehicle_capacity' in data_dict['problem_config']:
                        data_dict['vehicle_capacity'] = data_dict['problem_config']['vehicle_capacity']
                    else:
                        raise ValueError("vehicle_capacity not found in data_dict or problem_config")
                
                if 'num_vehicles' not in data_dict:
                    if 'problem_config' in data_dict and 'num_vehicles' in data_dict['problem_config']:
                        data_dict['num_vehicles'] = data_dict['problem_config']['num_vehicles']
                    else:
                        # Calculate using formula: ⌈n/8⌉ where n = number of customers
                        num_customers = len(data_dict.get('customers', []))
                        num_vehicles_formula = VRP_CONFIG.get('num_vehicles_formula', 'ceil(n/8)')
                        if num_vehicles_formula == 'ceil(n/8)':
                            data_dict['num_vehicles'] = max(1, int(np.ceil(num_customers / 8)))
                        else:
                            # Fallback to old formula
                            total_demand = sum(c.get('demand', 0) for c in data_dict.get('customers', []))
                            data_dict['num_vehicles'] = max(1, int(np.ceil(total_demand / data_dict['vehicle_capacity'])) + 2)
                
                distance_matrix = None
                
                if calculate_distance:
                    # Create DistanceCalculator with correct dataset_type for real routes
                    # Determine dataset type from metadata or parameter
                    metadata = data_dict.get('metadata', {}) or {}
                    if 'dataset_type' in metadata:
                        actual_dataset_type = metadata['dataset_type']
                    elif 'hanoi' in dataset_type.lower():
                        actual_dataset_type = "hanoi"
                    else:
                        actual_dataset_type = "solomon"
                    dataset_name = metadata.get('name')
                    pipeline_profiler.set_context(dataset_type=actual_dataset_type)
                    
                    # Create distance calculator with real routes for Hanoi
                    self.distance_calculator = DistanceCalculator(
                        traffic_factor=self.traffic_factor,
                        use_adaptive=self.use_adaptive,
                        dataset_type=actual_dataset_type,  # This enables real routes for Hanoi
                        dataset_name=dataset_name
                    )
                    
                    # Calculate distance matrix (real road routes for Hanoi, haversine for others)
                    coordinates = [(data_dict['depot']['x'], data_dict['depot']['y'])]
                    coordinates.extend([(c['x'], c['y']) for c in data_dict['customers']])
                    with pipeline_profiler.profile(
                        "distance.matrix.build",
                        metadata={'n_points': len(coordinates), 'dataset_type': actual_dataset_type}
                    ):
                        distance_matrix = self.distance_calculator.calculate_distance_matrix(coordinates)
                
                # Ensure dataset_type is set in metadata for downstream components
                if 'metadata' not in data_dict:
                    data_dict['metadata'] = {}
                data_dict['metadata']['dataset_type'] = actual_dataset_type
                
                # CORRECTNESS FIX: For Solomon datasets, disable adaptive traffic
                # Solomon should use pure Euclidean distance without traffic factors
                use_adaptive = VRP_CONFIG.get('use_adaptive_traffic', False)
                if actual_dataset_type.lower() == "solomon":
                    use_adaptive = False  # Force disable for Solomon
                
                with pipeline_profiler.profile("data.create_vrp_problem.model_build"):
                    problem = create_vrp_problem_from_dict(
                        data_dict, 
                        distance_matrix,
                        use_adaptive_traffic=use_adaptive
                    )
                
                # Set distance calculator reference for adaptive traffic (Hanoi only)
                if use_adaptive and self.distance_calculator:
                    problem.set_distance_calculator(self.distance_calculator)
                
                return problem
            
        except Exception as e:
            logger.error(f"Error creating VRP problem: {e}")
            raise DatasetError(f"Error creating VRP problem: {str(e)}")
        finally:
            pipeline_profiler.clear_context('dataset_type', 'num_customers')
    
    def generate_preview_statistics(self, data_dict: Dict) -> Dict:
        """
        Generate preview statistics for dataset.
        
        Args:
            data_dict: Validated data dictionary
            
        Returns:
            Dictionary with statistics
        """
        customers = data_dict.get('customers', [])
        depot = data_dict.get('depot', {})
        
        total_demand = sum(c.get('demand', 0) for c in customers)
        min_vehicles_needed = max(1, int(np.ceil(total_demand / data_dict.get('vehicle_capacity', 1))))
        
        # Calculate bounds
        if customers:
            x_coords = [c.get('x', 0) for c in customers] + [depot.get('x', 0)]
            y_coords = [c.get('y', 0) for c in customers] + [depot.get('y', 0)]
            
            bounds = {
                'min_x': min(x_coords),
                'max_x': max(x_coords),
                'min_y': min(y_coords),
                'max_y': max(y_coords)
            }
        else:
            bounds = None
        
        return {
            'num_customers': len(customers),
            'total_demand': total_demand,
            'vehicle_capacity': data_dict.get('vehicle_capacity', 0),
            'num_vehicles': data_dict.get('num_vehicles', 0),
            'min_vehicles_needed': min_vehicles_needed,
            'depot_location': {
                'x': depot.get('x', 0),
                'y': depot.get('y', 0)
            },
            'bounds': bounds
        }
    
    def create_sample_hanoi_dataset(self, n_customers: int = 10) -> Dict:
        """
        Create a sample Hanoi dataset for testing.
        
        Args:
            n_customers: Number of customers
            
        Returns:
            Data dictionary
        """
        import random
        
        # Hanoi center (Hoan Kiem Lake)
        base_lat, base_lon = 21.0285, 105.8542
        
        # Generate depot at center
        depot = {
            'id': 0,
            'x': base_lon,
            'y': base_lat,
            'demand': 0,
            'ready_time': 0.0,
            'due_date': 1000.0,
            'service_time': 0.0
        }
        
        # Generate customers around Hanoi
        customers = []
        random.seed(42)  # For reproducibility
        
        for i in range(1, n_customers + 1):
            # Generate coordinates within Hanoi bounds
            lat_offset = (random.random() - 0.5) * 0.15  # ~15km radius
            lon_offset = (random.random() - 0.5) * 0.15
            
            customers.append({
                'id': i,
                'x': base_lon + lon_offset,
                'y': base_lat + lat_offset,
                'demand': random.randint(5, 30),
                'ready_time': 0.0,
                'due_date': 1000.0,
                'service_time': 10.0
            })
        
        total_demand = sum(c['demand'] for c in customers)
        vehicle_capacity = 200
        min_vehicles = max(1, int(np.ceil(total_demand / vehicle_capacity)))
        
        return {
            'depot': depot,
            'customers': customers,
            'vehicle_capacity': vehicle_capacity,
            'num_vehicles': min_vehicles + 2,  # Add buffer
            'metadata': {
                'name': f'Sample Hanoi Dataset ({n_customers} customers)',
                'source': 'generated',
                'format': 'mockup',
                'description': f'Sample dataset for testing with {n_customers} customers',
                'num_customers': n_customers
            },
            'problem_config': {
                'vehicle_capacity': vehicle_capacity,
                'num_vehicles': min_vehicles + 2,
                'traffic_factor': 1.0
            }
        }

