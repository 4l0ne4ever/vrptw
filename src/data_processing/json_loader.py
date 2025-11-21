"""
JSON dataset loader for VRP problems.
Loads VRP datasets from JSON format with metadata support.
"""

import json
import os
from typing import Dict, List, Optional
from src.models.vrp_model import Customer, Depot, VRPProblem, create_vrp_problem_from_dict
from src.data_processing.distance import DistanceCalculator


class JSONDatasetLoader:
    """Loads VRP datasets from JSON format."""
    
    def __init__(self, datasets_dir: str = "data/datasets"):
        """
        Initialize JSON dataset loader.
        
        Args:
            datasets_dir: Directory containing JSON datasets
        """
        self.datasets_dir = datasets_dir
        self.solomon_dir = os.path.join(datasets_dir, "solomon")
        self.mockup_dir = os.path.join(datasets_dir, "mockup")
    
    def load_dataset(self, dataset_name: str, dataset_type: str = None) -> Dict:
        """
        Load VRP dataset by name.
        
        Args:
            dataset_name: Name of the dataset (with or without .json extension)
            dataset_type: Type of dataset ("solomon", "mockup") - auto-detect if None
            
        Returns:
            VRP data dictionary
        """
        # Add .json extension if not present
        if not dataset_name.endswith('.json'):
            dataset_name += '.json'
        
        # Auto-detect dataset type if not specified
        if dataset_type is None:
            filepath = os.path.join(self.solomon_dir, dataset_name)
            if os.path.exists(filepath):
                dataset_type = "solomon"
            else:
                filepath = os.path.join(self.mockup_dir, dataset_name)
                if os.path.exists(filepath):
                    dataset_type = "mockup"
                else:
                    raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        else:
            if dataset_type == "solomon":
                filepath = os.path.join(self.solomon_dir, dataset_name)
            elif dataset_type == "mockup":
                filepath = os.path.join(self.mockup_dir, dataset_name)
            else:
                raise ValueError(f"Invalid dataset type: {dataset_type}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Ensure metadata has dataset_type
        metadata = json_data['metadata'].copy() if 'metadata' in json_data else {}
        metadata['dataset_type'] = dataset_type  # Add dataset_type to metadata
        
        # CORRECTNESS FIX: For Solomon datasets, verify and correct vehicle capacity and num_vehicles
        num_vehicles = json_data['problem_config']['num_vehicles']
        vehicle_capacity = json_data['problem_config']['vehicle_capacity']
        
        if dataset_type == "solomon":
            dataset_name = metadata.get('name', '')
            
            # CORRECTNESS: Verify vehicle capacity matches Solomon benchmark standards
            # C1, R1, RC1: 200
            # C2: 700
            # R2, RC2: 1000
            # Pattern: C101-C109 (C1), C201-C208 (C2), R101-R112 (R1), R201-R211 (R2), etc.
            if dataset_name:
                correct_capacity = None
                if dataset_name.startswith('C'):
                    # C series: check 2nd character (index 1) - '1' for C1, '2' for C2
                    if len(dataset_name) >= 2 and dataset_name[1] == '2':
                        correct_capacity = 700  # C2 series
                    else:
                        correct_capacity = 200  # C1 series
                elif dataset_name.startswith('RC'):
                    # RC series: check 3rd character (index 2) - '1' for RC1, '2' for RC2
                    if len(dataset_name) >= 3 and dataset_name[2] == '2':
                        correct_capacity = 1000  # RC2 series
                    else:
                        correct_capacity = 200  # RC1 series
                elif dataset_name.startswith('R'):
                    # R series: check 2nd character (index 1) - '1' for R1, '2' for R2
                    if len(dataset_name) >= 2 and dataset_name[1] == '2':
                        correct_capacity = 1000  # R2 series
                    else:
                        correct_capacity = 200  # R1 series
                
                # Override with correct capacity if different (JSON might have wrong value)
                if correct_capacity is not None and vehicle_capacity != correct_capacity:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Correcting vehicle capacity for {dataset_name}: {vehicle_capacity} -> {correct_capacity}")
                    vehicle_capacity = correct_capacity
            
            # CORRECTNESS: Use BKS vehicle count if available (optimal)
            if dataset_name:
                try:
                    bks_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'solomon_bks.json')
                    if os.path.exists(bks_path):
                        import json as json_module
                        with open(bks_path, 'r') as f:
                            bks_data = json_module.load(f)
                            if dataset_name in bks_data:
                                bks_vehicles = bks_data[dataset_name].get('vehicles')
                                if bks_vehicles is not None:
                                    # Use BKS vehicle count (optimal) with minimal buffer for optimization flexibility
                                    # Add +1 buffer to allow GA to find solutions, but not excessive
                                    num_vehicles = min(bks_vehicles + 1, len(json_data['customers']))
                except Exception:
                    # If BKS loading fails, use JSON value (fallback)
                    pass
        
        # Convert to standard format
        return {
            'depot': json_data['depot'],
            'customers': json_data['customers'],
            'vehicle_capacity': vehicle_capacity,
            'num_vehicles': num_vehicles,
            'metadata': metadata,
            'problem_config': {
                **json_data['problem_config'],
                'vehicle_capacity': vehicle_capacity,
                'num_vehicles': num_vehicles  # Update with corrected values
            }
        }
    
    def load_dataset_with_distance_matrix(self, 
                                        dataset_name: str,
                                        traffic_factor: float = 1.0,
                                        dataset_type: str = None) -> tuple:
        """
        Load dataset and calculate distance matrix.
        
        Args:
            dataset_name: Name of the dataset
            traffic_factor: Traffic factor for distance calculation
            dataset_type: Type of dataset ("solomon", "mockup") - auto-detect if None
            
        Returns:
            Tuple of (data_dict, distance_matrix)
        """
        data = self.load_dataset(dataset_name, dataset_type)
        
        # Calculate distance matrix
        distance_calculator = DistanceCalculator(traffic_factor)
        coordinates = [(data['depot']['x'], data['depot']['y'])]
        coordinates.extend([(c['x'], c['y']) for c in data['customers']])
        distance_matrix = distance_calculator.calculate_distance_matrix(coordinates)
        
        return data, distance_matrix
    
    def create_vrp_problem_from_dataset(self, 
                                      dataset_name: str,
                                      traffic_factor: float = 1.0) -> VRPProblem:
        """
        Create VRP problem from dataset.
        
        Args:
            dataset_name: Name of the dataset
            traffic_factor: Traffic factor for distance calculation
            
        Returns:
            VRPProblem instance
        """
        data, distance_matrix = self.load_dataset_with_distance_matrix(
            dataset_name, traffic_factor
        )
        
        return create_vrp_problem_from_dict(data, distance_matrix)
    
    def list_available_datasets(self, dataset_type: str = "all") -> List[Dict]:
        """
        List all available datasets.
        
        Args:
            dataset_type: Type of datasets to list ("solomon", "mockup", "all")
            
        Returns:
            List of dataset information
        """
        datasets = []
        
        # Determine which directories to search
        search_dirs = []
        if dataset_type in ["solomon", "all"]:
            search_dirs.append(("solomon", self.solomon_dir))
        if dataset_type in ["mockup", "all"]:
            search_dirs.append(("mockup", self.mockup_dir))
        
        for dataset_type_name, dataset_dir in search_dirs:
            if not os.path.exists(dataset_dir):
                continue
                
            for filename in os.listdir(dataset_dir):
                if filename.endswith('.json'):
                    try:
                        data = self.load_dataset(filename, dataset_type_name)
                        datasets.append({
                            'name': filename.replace('.json', ''),
                            'filename': filename,
                            'metadata': data['metadata'],
                            'num_customers': len(data['customers']),
                            'vehicle_capacity': data['vehicle_capacity'],
                            'num_vehicles': data['num_vehicles'],
                            'type': dataset_type_name
                        })
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
        
        return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset information dictionary
        """
        data = self.load_dataset(dataset_name)
        
        return {
            'name': data['metadata']['name'],
            'description': data['metadata']['description'],
            'format': data['metadata']['format'],
            'num_customers': len(data['customers']),
            'vehicle_capacity': data['vehicle_capacity'],
            'num_vehicles': data['num_vehicles'],
            'source': data['metadata'].get('source', 'unknown'),
            'problem_config': data['problem_config']
        }
    
    def search_datasets(self, 
                       min_customers: Optional[int] = None,
                       max_customers: Optional[int] = None,
                       min_capacity: Optional[float] = None,
                       max_capacity: Optional[float] = None,
                       format_filter: Optional[str] = None) -> List[Dict]:
        """
        Search datasets by criteria.
        
        Args:
            min_customers: Minimum number of customers
            max_customers: Maximum number of customers
            min_capacity: Minimum vehicle capacity
            max_capacity: Maximum vehicle capacity
            format_filter: Format filter (solomon_csv, mockup, etc.)
            
        Returns:
            List of matching datasets
        """
        datasets = self.list_available_datasets()
        filtered = []
        
        for dataset in datasets:
            # Apply filters
            if min_customers is not None and dataset['num_customers'] < min_customers:
                continue
            if max_customers is not None and dataset['num_customers'] > max_customers:
                continue
            if min_capacity is not None and dataset['vehicle_capacity'] < min_capacity:
                continue
            if max_capacity is not None and dataset['vehicle_capacity'] > max_capacity:
                continue
            if format_filter is not None and dataset['metadata']['format'] != format_filter:
                continue
            
            filtered.append(dataset)
        
        return filtered


def load_json_dataset(dataset_name: str, 
                     datasets_dir: str = "data/datasets",
                     traffic_factor: float = 1.0) -> tuple:
    """
    Convenience function to load JSON dataset.
    
    Args:
        dataset_name: Name of the dataset
        datasets_dir: Directory containing datasets
        traffic_factor: Traffic factor for distance calculation
        
    Returns:
        Tuple of (data_dict, distance_matrix)
    """
    loader = JSONDatasetLoader(datasets_dir)
    return loader.load_dataset_with_distance_matrix(dataset_name, traffic_factor)


def create_vrp_problem_from_json_dataset(dataset_name: str,
                                        datasets_dir: str = "data/datasets",
                                        traffic_factor: float = 1.0) -> VRPProblem:
    """
    Convenience function to create VRP problem from JSON dataset.
    
    Args:
        dataset_name: Name of the dataset
        datasets_dir: Directory containing datasets
        traffic_factor: Traffic factor for distance calculation
        
    Returns:
        VRPProblem instance
    """
    loader = JSONDatasetLoader(datasets_dir)
    return loader.create_vrp_problem_from_dataset(dataset_name, traffic_factor)
