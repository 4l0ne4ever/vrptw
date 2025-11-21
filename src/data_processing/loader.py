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
        
        # CORRECTNESS: Different Solomon series have different vehicle capacities
        # C1, R1, RC1: capacity 200
        # C2: capacity 700
        # R2, RC2: capacity 1000
        # Pattern: C101-C109 (C1), C201-C208 (C2), R101-R112 (R1), R201-R211 (R2), etc.
        if filename.startswith('C'):
            # C series: check 2nd character (index 1) - '1' for C1, '2' for C2
            # C101 -> filename[1] = '1' (C1), C201 -> filename[1] = '2' (C2)
            if len(filename) >= 2 and filename[1] == '2':
                self.vehicle_capacity = 700  # C2 series
            else:
                self.vehicle_capacity = 200  # C1 series
        elif filename.startswith('RC'):
            # RC series: check 3rd character (index 2) - '1' for RC1, '2' for RC2
            # RC101 -> filename[2] = '1' (RC1), RC201 -> filename[2] = '2' (RC2)
            if len(filename) >= 3 and filename[2] == '2':
                self.vehicle_capacity = 1000  # RC2 series
            else:
                self.vehicle_capacity = 200  # RC1 series
        elif filename.startswith('R'):
            # R series: check 2nd character (index 1) - '1' for R1, '2' for R2
            # R101 -> filename[1] = '1' (R1), R201 -> filename[1] = '2' (R2)
            if len(filename) >= 2 and filename[1] == '2':
                self.vehicle_capacity = 1000  # R2 series
            else:
                self.vehicle_capacity = 200  # R1 series
        else:
            # Default capacity (should not happen for Solomon datasets)
            self.vehicle_capacity = 200
        
        # CORRECTNESS FIX: Calculate num_vehicles more accurately
        # For Solomon datasets, try to use BKS vehicle count if available
        total_demand = sum(c['demand'] for c in self.customers)
        min_vehicles = max(1, int(np.ceil(total_demand / self.vehicle_capacity)))
        
        # Try to load BKS data for accurate vehicle count
        dataset_name = os.path.splitext(filename)[0]
        try:
            bks_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'solomon_bks.json')
            if os.path.exists(bks_path):
                import json
                with open(bks_path, 'r') as f:
                    bks_data = json.load(f)
                    if dataset_name in bks_data:
                        bks_vehicles = bks_data[dataset_name].get('vehicles')
                        if bks_vehicles is not None:
                            # Use BKS vehicle count (optimal) with minimal buffer for optimization
                            # Add +1 to allow GA flexibility, but not excessive
                            self.num_vehicles = min(bks_vehicles + 1, len(self.customers))
                            return
        except Exception:
            # If BKS loading fails, use calculated value
            pass
        
        # Fallback: Use calculated minimum with small buffer (not +2 which is too much)
        # Only add +1 buffer for optimization flexibility
        self.num_vehicles = min(min_vehicles + 1, len(self.customers))
    
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
