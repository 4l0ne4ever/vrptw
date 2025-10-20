"""
Data loader for Solomon benchmark dataset.
Handles CSV parsing and data validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


class SolomonLoader:
    """Loads and parses Solomon VRP benchmark dataset."""
    
    def __init__(self):
        self.customers = None
        self.depot = None
        self.vehicle_capacity = None
        self.num_vehicles = None
        
    def load_from_file(self, file_path: str) -> Dict:
        """
        Load VRP data from Solomon CSV file.
        
        Args:
            file_path: Path to Solomon CSV file
            
        Returns:
            Dictionary containing customers, depot, and problem parameters
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 
                           'READY TIME', 'DUE DATE', 'SERVICE TIME']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Extract depot (customer with demand 0, typically customer 1)
        depot_row = df[df['DEMAND'] == 0].iloc[0]
        self.depot = {
            'id': 0,  # Always use 0 as depot ID
            'x': depot_row['XCOORD.'],
            'y': depot_row['YCOORD.'],
            'demand': depot_row['DEMAND'],
            'ready_time': depot_row['READY TIME'],
            'due_date': depot_row['DUE DATE'],
            'service_time': depot_row['SERVICE TIME']
        }
        
        # Extract customers (excluding depot - customer with demand 0)
        customers_df = df[df['DEMAND'] > 0].copy()
        self.customers = []
        
        for _, row in customers_df.iterrows():
            customer = {
                'id': int(row['CUST NO.']),
                'x': float(row['XCOORD.']),
                'y': float(row['YCOORD.']),
                'demand': float(row['DEMAND']),
                'ready_time': float(row['READY TIME']),
                'due_date': float(row['DUE DATE']),
                'service_time': float(row['SERVICE TIME'])
            }
            self.customers.append(customer)
        
        # Extract problem parameters from filename or data
        self._extract_problem_parameters(file_path)
        
        return self._to_dict()
    
    def _extract_problem_parameters(self, file_path: str):
        """Extract vehicle capacity and other parameters from filename or data."""
        filename = os.path.basename(file_path)
        
        # Determine vehicle capacity from filename pattern
        if filename.startswith('C'):
            self.vehicle_capacity = 200
        elif filename.startswith('R'):
            self.vehicle_capacity = 200
        elif filename.startswith('RC'):
            self.vehicle_capacity = 200
        else:
            # Default capacity
            self.vehicle_capacity = 200
        
        # Estimate number of vehicles needed
        total_demand = sum(c['demand'] for c in self.customers)
        self.num_vehicles = max(1, int(np.ceil(total_demand / self.vehicle_capacity)))
        
        # Add some buffer for optimization
        self.num_vehicles = min(self.num_vehicles + 2, len(self.customers))
    
    def _to_dict(self) -> Dict:
        """Convert loaded data to dictionary format."""
        return {
            'customers': self.customers,
            'depot': self.depot,
            'vehicle_capacity': self.vehicle_capacity,
            'num_vehicles': self.num_vehicles,
            'num_customers': len(self.customers)
        }
    
    def validate_data(self) -> bool:
        """
        Validate loaded data for consistency.
        
        Returns:
            True if data is valid, False otherwise
        """
        if not self.customers or not self.depot:
            return False
        
        # Check depot coordinates
        if self.depot['x'] is None or self.depot['y'] is None:
            return False
        
        # Check customer data
        for customer in self.customers:
            if (customer['x'] is None or customer['y'] is None or 
                customer['demand'] < 0 or customer['demand'] > self.vehicle_capacity):
                return False
        
        # Check capacity constraints
        max_demand = max(c['demand'] for c in self.customers)
        if max_demand > self.vehicle_capacity:
            return False
        
        return True
    
    def get_customer_coordinates(self) -> List[Tuple[float, float]]:
        """Get list of customer coordinates (x, y)."""
        return [(c['x'], c['y']) for c in self.customers]
    
    def get_depot_coordinates(self) -> Tuple[float, float]:
        """Get depot coordinates (x, y)."""
        return (self.depot['x'], self.depot['y'])
    
    def get_demands(self) -> List[float]:
        """Get list of customer demands."""
        return [c['demand'] for c in self.customers]
    
    def get_time_windows(self) -> List[Tuple[float, float]]:
        """Get list of time windows (ready_time, due_date)."""
        return [(c['ready_time'], c['due_date']) for c in self.customers]
    
    def get_service_times(self) -> List[float]:
        """Get list of service times."""
        return [c['service_time'] for c in self.customers]


def load_solomon_dataset(file_path: str) -> Dict:
    """
    Convenience function to load Solomon dataset.
    
    Args:
        file_path: Path to Solomon CSV file
        
    Returns:
        Dictionary containing VRP problem data
    """
    loader = SolomonLoader()
    data = loader.load_from_file(file_path)
    
    if not loader.validate_data():
        raise ValueError("Invalid data loaded from file")
    
    return data
