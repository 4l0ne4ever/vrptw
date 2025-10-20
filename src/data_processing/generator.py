"""
Mockup data generator for VRP problems.
Creates synthetic customer data with various clustering patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import os
from config import MOCKUP_CONFIG
from .enhanced_hanoi_coordinates import EnhancedHanoiCoordinateGenerator

# Optional import for KMeans
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MockupDataGenerator:
    """Generates synthetic VRP problem instances."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize generator with configuration.
        
        Args:
            config: Configuration dictionary, uses default if None
        """
        self.config = config or MOCKUP_CONFIG.copy()
        self.customers = []
        self.depot = None
        self.vehicle_capacity = None
        self.num_vehicles = None
        
        # Initialize enhanced Hanoi coordinate generator
        self.hanoi_generator = EnhancedHanoiCoordinateGenerator()
        
        # Set random seed for reproducibility
        np.random.seed(self.config['seed'])
    
    def generate_customers(self, n_customers: Optional[int] = None) -> List[Dict]:
        """
        Generate customer data with specified clustering pattern using Hanoi coordinates.
        
        Args:
            n_customers: Number of customers to generate
            
        Returns:
            List of customer dictionaries
        """
        n_customers = n_customers or self.config['n_customers']
        
        # Generate Hanoi coordinates
        customer_coords, depot_coords = self.hanoi_generator.generate_coordinates(
            n_customers, 
            self.config['clustering'],
            'hoan_kiem_lake'  # Default depot location
        )
        
        # Generate demands using Poisson distribution
        demands = self._generate_demands(n_customers)
        
        # Generate time windows
        time_windows = self._generate_time_windows(n_customers)
        
        # Generate service times
        service_times = self._generate_service_times(n_customers)
        
        # Create customer dictionaries
        self.customers = []
        for i in range(n_customers):
            lat, lon = customer_coords[i]
            customer = {
                'id': i + 1,  # Start from 1, depot will be 0
                'x': lon,     # Longitude (x-coordinate)
                'y': lat,     # Latitude (y-coordinate)
                'demand': demands[i],
                'ready_time': time_windows[i][0],
                'due_date': time_windows[i][1],
                'service_time': service_times[i]
            }
            self.customers.append(customer)
        
        # Store depot coordinates for later use
        self._depot_coords = depot_coords
        
        return self.customers
    
    def generate_depot(self, x: Optional[float] = None, y: Optional[float] = None) -> Dict:
        """
        Generate depot location using Hanoi coordinates.
        
        Args:
            x: Depot longitude (uses generated if None)
            y: Depot latitude (uses generated if None)
            
        Returns:
            Depot dictionary
        """
        if hasattr(self, '_depot_coords'):
            # Use generated depot coordinates
            lat, lon = self._depot_coords
            depot_x = lon  # Longitude
            depot_y = lat  # Latitude
        else:
            # Fallback to Hoan Kiem Lake coordinates
            depot_x = 105.8542  # Longitude
            depot_y = 21.0285    # Latitude
        
        # Override with provided coordinates if any
        if x is not None:
            depot_x = x
        if y is not None:
            depot_y = y
        
        self.depot = {
            'id': 0,
            'x': depot_x,
            'y': depot_y,
            'demand': 0,
            'ready_time': 0,
            'due_date': 1000,  # Large time window for depot
            'service_time': 0
        }
        
        return self.depot
    
    def _generate_random_coordinates(self, n_customers: int) -> List[Tuple[float, float]]:
        """Generate random coordinates within area bounds."""
        x_min, x_max = self.config['area_bounds']
        y_min, y_max = self.config['area_bounds']
        
        x_coords = np.random.uniform(x_min, x_max, n_customers)
        y_coords = np.random.uniform(y_min, y_max, n_customers)
        
        return list(zip(x_coords, y_coords))
    
    def _generate_kmeans_coordinates(self, n_customers: int) -> List[Tuple[float, float]]:
        """Generate coordinates using K-means clustering."""
        n_clusters = min(self.config['n_clusters'], n_customers)
        
        # Generate cluster centers
        x_min, x_max = self.config['area_bounds']
        y_min, y_max = self.config['area_bounds']
        
        cluster_centers = np.random.uniform(
            [x_min, y_min], [x_max, y_max], 
            size=(n_clusters, 2)
        )
        
        # Generate points around cluster centers
        coordinates = []
        customers_per_cluster = n_customers // n_clusters
        remaining_customers = n_customers % n_clusters
        
        for i, center in enumerate(cluster_centers):
            cluster_size = customers_per_cluster
            if i < remaining_customers:
                cluster_size += 1
            
            # Generate points with Gaussian distribution around center
            cluster_coords = np.random.normal(
                center, 
                scale=(x_max - x_min) * 0.1,  # 10% of area width as std
                size=(cluster_size, 2)
            )
            
            # Clip to area bounds
            cluster_coords[:, 0] = np.clip(cluster_coords[:, 0], x_min, x_max)
            cluster_coords[:, 1] = np.clip(cluster_coords[:, 1], y_min, y_max)
            
            coordinates.extend([(x, y) for x, y in cluster_coords])
        
        return coordinates
    
    def _generate_radial_coordinates(self, n_customers: int) -> List[Tuple[float, float]]:
        """Generate coordinates in radial pattern around depot."""
        x_min, x_max = self.config['area_bounds']
        y_min, y_max = self.config['area_bounds']
        
        # Depot at center
        depot_x = (x_min + x_max) / 2
        depot_y = (y_min + y_max) / 2
        
        coordinates = []
        for i in range(n_customers):
            # Random angle
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Random radius (closer to depot more likely)
            max_radius = min(x_max - depot_x, y_max - depot_y) * 0.8
            radius = np.random.exponential(max_radius / 3)
            radius = min(radius, max_radius)
            
            x = depot_x + radius * np.cos(angle)
            y = depot_y + radius * np.sin(angle)
            
            # Clip to bounds
            x = np.clip(x, x_min, x_max)
            y = np.clip(y, y_min, y_max)
            
            coordinates.append((x, y))
        
        return coordinates
    
    def _generate_demands(self, n_customers: int) -> List[float]:
        """Generate customer demands using Poisson distribution."""
        lambda_param = self.config['demand_lambda']
        demands = np.random.poisson(lambda_param, n_customers)
        
        # Clip to min/max bounds
        demands = np.clip(
            demands, 
            self.config['demand_min'], 
            self.config['demand_max']
        )
        
        return demands.tolist()
    
    def _generate_time_windows(self, n_customers: int) -> List[Tuple[float, float]]:
        """Generate time windows for customers."""
        time_windows = []
        window_width = self.config['time_window_width']
        
        for i in range(n_customers):
            # Random ready time
            ready_time = np.random.uniform(0, 500)
            due_date = ready_time + window_width
            
            time_windows.append((ready_time, due_date))
        
        return time_windows
    
    def _generate_service_times(self, n_customers: int) -> List[float]:
        """Generate service times for customers."""
        service_time = self.config['service_time']
        # Add some variation
        service_times = np.random.normal(service_time, service_time * 0.2, n_customers)
        service_times = np.clip(service_times, service_time * 0.5, service_time * 1.5)
        
        return service_times.tolist()
    
    def set_vehicle_capacity(self, capacity: float):
        """Set vehicle capacity."""
        self.vehicle_capacity = capacity
    
    def estimate_vehicles_needed(self) -> int:
        """Estimate number of vehicles needed based on total demand."""
        if not self.customers or not self.vehicle_capacity:
            return 1
        
        total_demand = sum(c['demand'] for c in self.customers)
        num_vehicles = max(1, int(np.ceil(total_demand / self.vehicle_capacity)))
        
        # Add buffer for optimization
        return min(num_vehicles + 2, len(self.customers))
    
    def export_to_csv(self, file_path: str):
        """
        Export generated data to CSV file in Solomon format.
        
        Args:
            file_path: Output file path
        """
        if not self.customers or not self.depot:
            raise ValueError("No data to export. Generate customers and depot first.")
        
        # Create DataFrame
        data = []
        
        # Add depot
        depot_row = [
            self.depot['id'],
            self.depot['x'],
            self.depot['y'],
            self.depot['demand'],
            self.depot['ready_time'],
            self.depot['due_date'],
            self.depot['service_time']
        ]
        data.append(depot_row)
        
        # Add customers
        for customer in self.customers:
            customer_row = [
                customer['id'],
                customer['x'],
                customer['y'],
                customer['demand'],
                customer['ready_time'],
                customer['due_date'],
                customer['service_time']
            ]
            data.append(customer_row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[
            'CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND',
            'READY TIME', 'DUE DATE', 'SERVICE TIME'
        ])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
    
    def to_dict(self) -> Dict:
        """Convert generated data to dictionary format."""
        return {
            'customers': self.customers,
            'depot': self.depot,
            'vehicle_capacity': self.vehicle_capacity,
            'num_vehicles': self.num_vehicles or self.estimate_vehicles_needed(),
            'num_customers': len(self.customers)
        }


def generate_mockup_data(n_customers: int, 
                        vehicle_capacity: float = 200,
                        clustering: str = 'kmeans',
                        output_file: Optional[str] = None) -> Dict:
    """
    Convenience function to generate mockup data.
    
    Args:
        n_customers: Number of customers to generate
        vehicle_capacity: Vehicle capacity
        clustering: Clustering method ('random', 'kmeans', 'radial')
        output_file: Optional output CSV file path
        
    Returns:
        Dictionary containing generated VRP data
    """
    config = MOCKUP_CONFIG.copy()
    config['n_customers'] = n_customers
    config['clustering'] = clustering
    
    generator = MockupDataGenerator(config)
    generator.generate_customers()
    generator.generate_depot()
    generator.set_vehicle_capacity(vehicle_capacity)
    generator.num_vehicles = generator.estimate_vehicles_needed()
    
    if output_file:
        generator.export_to_csv(output_file)
    
    return generator.to_dict()
