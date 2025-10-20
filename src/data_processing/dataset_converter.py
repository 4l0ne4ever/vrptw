"""
Dataset converter for VRP problems.
Converts Solomon CSV files to JSON format and manages datasets.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src.data_processing.loader import load_solomon_dataset


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class DatasetConverter:
    """Converts and manages VRP datasets in JSON format."""
    
    def __init__(self, datasets_dir: str = "data/datasets"):
        """
        Initialize dataset converter.
        
        Args:
            datasets_dir: Directory to store JSON datasets
        """
        self.datasets_dir = datasets_dir
        self.solomon_dir = os.path.join(datasets_dir, "solomon")
        self.mockup_dir = os.path.join(datasets_dir, "mockup")
        
        os.makedirs(self.solomon_dir, exist_ok=True)
        os.makedirs(self.mockup_dir, exist_ok=True)
    
    def convert_solomon_to_json(self, csv_file: str, output_file: str = None):
        """
        Convert Solomon CSV file to JSON format.
        
        Args:
            csv_file: Path to Solomon CSV file
            output_file: Output JSON file path (auto-generated if None)
        """
        # Load Solomon data
        data = load_solomon_dataset(csv_file)
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_file = os.path.join(self.solomon_dir, f"{base_name}.json")
        
        # Convert to JSON format
        json_data = self._convert_to_json_format(data, csv_file)
        
        # Save JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"Converted {csv_file} to {output_file}")
        return output_file
    
    def _convert_to_json_format(self, data: Dict, source_file: str) -> Dict:
        """
        Convert VRP data to standardized JSON format.
        
        Args:
            data: VRP data dictionary
            source_file: Source file path
            
        Returns:
            Standardized JSON data
        """
        return {
            "metadata": {
                "name": os.path.splitext(os.path.basename(source_file))[0],
                "source": source_file,
                "format": "solomon_csv",
                "description": f"Solomon VRP dataset converted from {os.path.basename(source_file)}",
                "num_customers": len(data['customers']),
                "vehicle_capacity": data['vehicle_capacity'],
                "num_vehicles": data['num_vehicles']
            },
            "depot": data['depot'],
            "customers": data['customers'],
            "problem_config": {
                "vehicle_capacity": data['vehicle_capacity'],
                "num_vehicles": data['num_vehicles'],
                "traffic_factor": 1.0,
                "penalty_weight": 1000
            }
        }
    
    def create_mockup_dataset(self, 
                            n_customers: int,
                            vehicle_capacity: float,
                            clustering: str = 'kmeans',
                            name: str = None,
                            output_file: str = None) -> str:
        """
        Create a mockup dataset in JSON format.
        
        Args:
            n_customers: Number of customers
            vehicle_capacity: Vehicle capacity
            clustering: Clustering method
            name: Dataset name
            output_file: Output file path
            
        Returns:
            Path to created JSON file
        """
        from src.data_processing.generator import generate_mockup_data
        
        # Generate mockup data
        data = generate_mockup_data(
            n_customers=n_customers,
            vehicle_capacity=vehicle_capacity,
            clustering=clustering
        )
        
        # Generate name and output file
        if name is None:
            name = f"mockup_{n_customers}c_{int(vehicle_capacity)}cap_{clustering}"
        
        if output_file is None:
            output_file = os.path.join(self.mockup_dir, f"{name}.json")
        
        # Create JSON data
        json_data = {
            "metadata": {
                "name": name,
                "source": "generated",
                "format": "mockup",
                "description": f"Generated mockup VRP dataset with {n_customers} customers",
                "num_customers": n_customers,
                "vehicle_capacity": vehicle_capacity,
                "num_vehicles": data['num_vehicles'],
                "clustering": clustering
            },
            "depot": data['depot'],
            "customers": data['customers'],
            "problem_config": {
                "vehicle_capacity": vehicle_capacity,
                "num_vehicles": data['num_vehicles'],
                "traffic_factor": 1.0,
                "penalty_weight": 1000,
                "clustering": clustering
            }
        }
        
        # Save JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"Created mockup dataset: {output_file}")
        return output_file
    
    def load_json_dataset(self, json_file: str) -> Dict:
        """
        Load VRP dataset from JSON file.
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            VRP data dictionary
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Convert to standard format
        return {
            'depot': json_data['depot'],
            'customers': json_data['customers'],
            'vehicle_capacity': json_data['problem_config']['vehicle_capacity'],
            'num_vehicles': json_data['problem_config']['num_vehicles'],
            'metadata': json_data['metadata']
        }
    
    def list_datasets(self, dataset_type: str = "all") -> List[Dict]:
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
                    filepath = os.path.join(dataset_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        datasets.append({
                            'filename': filename,
                            'filepath': filepath,
                            'name': data['metadata']['name'],
                            'description': data['metadata']['description'],
                            'num_customers': data['metadata']['num_customers'],
                            'vehicle_capacity': data['metadata']['vehicle_capacity'],
                            'format': data['metadata']['format'],
                            'type': dataset_type_name
                        })
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
        
        return datasets
    
    def convert_all_solomon_datasets(self, solomon_dir: str = "data/solomon_dataset"):
        """
        Convert all Solomon datasets to JSON format.
        
        Args:
            solomon_dir: Directory containing Solomon CSV files
        """
        converted_count = 0
        
        for root, dirs, files in os.walk(solomon_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    try:
                        self.convert_solomon_to_json(csv_path)
                        converted_count += 1
                    except Exception as e:
                        print(f"Error converting {csv_path}: {e}")
        
        print(f"Converted {converted_count} Solomon datasets to JSON format")
    
    def create_dataset_catalog(self) -> str:
        """
        Create a catalog of all available datasets.
        
        Returns:
            Path to catalog file
        """
        datasets = self.list_datasets()
        
        catalog = {
            "catalog_info": {
                "created": "2024-01-01",
                "total_datasets": len(datasets),
                "description": "VRP-GA System Dataset Catalog"
            },
            "datasets": datasets
        }
        
        catalog_file = os.path.join(self.datasets_dir, "catalog.json")
        with open(catalog_file, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"Created dataset catalog: {catalog_file}")
        return catalog_file


def convert_solomon_datasets():
    """Convert all Solomon datasets to JSON format."""
    converter = DatasetConverter()
    converter.convert_all_solomon_datasets()
    converter.create_dataset_catalog()


def create_sample_datasets():
    """Create sample mockup datasets."""
    converter = DatasetConverter()
    
    # Create various sample datasets
    datasets = [
        (10, 50, 'random', 'small_random'),
        (20, 100, 'kmeans', 'medium_kmeans'),
        (30, 150, 'radial', 'medium_radial'),
        (50, 200, 'kmeans', 'large_kmeans'),
        (100, 300, 'kmeans', 'xlarge_kmeans')
    ]
    
    for n_customers, capacity, clustering, name in datasets:
        converter.create_mockup_dataset(
            n_customers=n_customers,
            vehicle_capacity=capacity,
            clustering=clustering,
            name=name
        )
    
    converter.create_dataset_catalog()


if __name__ == "__main__":
    # Convert Solomon datasets
    convert_solomon_datasets()
    
    # Create sample datasets
    create_sample_datasets()
