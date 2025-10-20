"""
Hanoi coordinate generator for mockup VRP datasets.
Generates realistic coordinates within Hanoi city boundaries.
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
import random


class HanoiCoordinateGenerator:
    """Generates realistic coordinates within Hanoi city."""
    
    def __init__(self):
        """Initialize Hanoi coordinate generator."""
        # Hanoi city boundaries (approximate)
        self.hanoi_bounds = {
            'min_lat': 20.7,   # Southern boundary
            'max_lat': 21.4,   # Northern boundary  
            'min_lon': 105.3,  # Western boundary
            'max_lon': 106.0   # Eastern boundary
        }
        
        # Major districts in Hanoi (approximate centers)
        self.districts = {
            'hoan_kiem': {'lat': 21.0285, 'lon': 105.8542, 'radius': 0.05},
            'ba_dinh': {'lat': 21.0333, 'lon': 105.8333, 'radius': 0.08},
            'dong_da': {'lat': 21.0167, 'lon': 105.8333, 'radius': 0.08},
            'hai_ba_trung': {'lat': 21.0167, 'lon': 105.8500, 'radius': 0.08},
            'cau_giay': {'lat': 21.0333, 'lon': 105.8000, 'radius': 0.1},
            'dong_anh': {'lat': 21.1667, 'lon': 105.7500, 'radius': 0.15},
            'long_bien': {'lat': 21.0500, 'lon': 105.9000, 'radius': 0.1},
            'thanh_xuan': {'lat': 21.0000, 'lon': 105.8000, 'radius': 0.08},
            'tu_liem': {'lat': 21.1000, 'lon': 105.7000, 'radius': 0.12},
            'gia_lam': {'lat': 21.0167, 'lon': 105.9167, 'radius': 0.1}
        }
        
        # Major landmarks for depot placement
        self.landmarks = {
            'hoan_kiem_lake': {'lat': 21.0285, 'lon': 105.8542},
            'west_lake': {'lat': 21.0500, 'lon': 105.8167},
            'noi_bai_airport': {'lat': 21.2167, 'lon': 105.8000},
            'long_bien_bridge': {'lat': 21.0500, 'lon': 105.8667},
            'temple_of_literature': {'lat': 21.0267, 'lon': 105.8356}
        }
    
    def generate_coordinates(self, n_customers: int, 
                           clustering: str = 'random',
                           depot_location: str = 'hoan_kiem_lake') -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
        """
        Generate customer coordinates and depot location.
        
        Args:
            n_customers: Number of customers
            clustering: Clustering method ('random', 'kmeans', 'radial', 'district')
            depot_location: Depot landmark name
            
        Returns:
            Tuple of (customer_coordinates, depot_coordinates)
        """
        # Generate depot coordinates
        depot_coords = self._generate_depot_coordinates(depot_location)
        
        # Generate customer coordinates based on clustering method
        if clustering == 'random':
            customer_coords = self._generate_random_coordinates(n_customers)
        elif clustering == 'kmeans':
            customer_coords = self._generate_kmeans_coordinates(n_customers, depot_coords)
        elif clustering == 'radial':
            customer_coords = self._generate_radial_coordinates(n_customers, depot_coords)
        elif clustering == 'district':
            customer_coords = self._generate_district_coordinates(n_customers)
        else:
            customer_coords = self._generate_random_coordinates(n_customers)
        
        return customer_coords, depot_coords
    
    def _generate_depot_coordinates(self, location: str) -> Tuple[float, float]:
        """Generate depot coordinates at specified landmark."""
        if location in self.landmarks:
            landmark = self.landmarks[location]
            # Add small random offset to avoid exact landmark placement
            lat_offset = np.random.normal(0, 0.005)
            lon_offset = np.random.normal(0, 0.005)
            return (landmark['lat'] + lat_offset, landmark['lon'] + lon_offset)
        else:
            # Default to Hoan Kiem Lake
            return self.landmarks['hoan_kiem_lake']['lat'], self.landmarks['hoan_kiem_lake']['lon']
    
    def _generate_random_coordinates(self, n_customers: int) -> List[Tuple[float, float]]:
        """Generate random coordinates within Hanoi boundaries."""
        coordinates = []
        
        for _ in range(n_customers):
            lat = np.random.uniform(self.hanoi_bounds['min_lat'], self.hanoi_bounds['max_lat'])
            lon = np.random.uniform(self.hanoi_bounds['min_lon'], self.hanoi_bounds['max_lon'])
            coordinates.append((lat, lon))
        
        return coordinates
    
    def _generate_kmeans_coordinates(self, n_customers: int, depot_coords: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Generate coordinates using K-means clustering."""
        # Create cluster centers around depot
        n_clusters = min(5, max(2, n_customers // 10))
        
        # Generate cluster centers
        cluster_centers = []
        depot_lat, depot_lon = depot_coords
        
        for i in range(n_clusters):
            # Spread clusters around depot
            angle = 2 * np.pi * i / n_clusters
            distance = np.random.uniform(0.02, 0.08)  # 2-8km radius
            
            lat_offset = distance * np.cos(angle)
            lon_offset = distance * np.sin(angle)
            
            cluster_lat = depot_lat + lat_offset
            cluster_lon = depot_lon + lon_offset
            
            # Ensure within Hanoi bounds
            cluster_lat = np.clip(cluster_lat, self.hanoi_bounds['min_lat'], self.hanoi_bounds['max_lat'])
            cluster_lon = np.clip(cluster_lon, self.hanoi_bounds['min_lon'], self.hanoi_bounds['max_lon'])
            
            cluster_centers.append([cluster_lat, cluster_lon])
        
        # Generate customers around cluster centers
        coordinates = []
        customers_per_cluster = n_customers // n_clusters
        remaining_customers = n_customers % n_clusters
        
        for i, center in enumerate(cluster_centers):
            cluster_size = customers_per_cluster
            if i < remaining_customers:
                cluster_size += 1
            
            for _ in range(cluster_size):
                # Add random offset around cluster center
                lat_offset = np.random.normal(0, 0.01)
                lon_offset = np.random.normal(0, 0.01)
                
                lat = center[0] + lat_offset
                lon = center[1] + lon_offset
                
                # Ensure within Hanoi bounds
                lat = np.clip(lat, self.hanoi_bounds['min_lat'], self.hanoi_bounds['max_lat'])
                lon = np.clip(lon, self.hanoi_bounds['min_lon'], self.hanoi_bounds['max_lon'])
                
                coordinates.append((lat, lon))
        
        return coordinates
    
    def _generate_radial_coordinates(self, n_customers: int, depot_coords: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Generate coordinates in radial pattern around depot."""
        coordinates = []
        depot_lat, depot_lon = depot_coords
        
        for i in range(n_customers):
            # Generate distance and angle
            distance = np.random.uniform(0.01, 0.1)  # 1-10km radius
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Convert to lat/lon offset
            lat_offset = distance * np.cos(angle)
            lon_offset = distance * np.sin(angle)
            
            lat = depot_lat + lat_offset
            lon = depot_lon + lon_offset
            
            # Ensure within Hanoi bounds
            lat = np.clip(lat, self.hanoi_bounds['min_lat'], self.hanoi_bounds['max_lat'])
            lon = np.clip(lon, self.hanoi_bounds['min_lon'], self.hanoi_bounds['max_lon'])
            
            coordinates.append((lat, lon))
        
        return coordinates
    
    def _generate_district_coordinates(self, n_customers: int) -> List[Tuple[float, float]]:
        """Generate coordinates distributed across Hanoi districts."""
        coordinates = []
        district_names = list(self.districts.keys())
        
        for i in range(n_customers):
            # Select random district
            district_name = random.choice(district_names)
            district = self.districts[district_name]
            
            # Generate coordinates within district
            lat_offset = np.random.normal(0, district['radius'] * 0.5)
            lon_offset = np.random.normal(0, district['radius'] * 0.5)
            
            lat = district['lat'] + lat_offset
            lon = district['lon'] + lon_offset
            
            # Ensure within Hanoi bounds
            lat = np.clip(lat, self.hanoi_bounds['min_lat'], self.hanoi_bounds['max_lat'])
            lon = np.clip(lon, self.hanoi_bounds['min_lon'], self.hanoi_bounds['max_lon'])
            
            coordinates.append((lat, lon))
        
        return coordinates
    
    def get_hanoi_bounds(self) -> Dict[str, float]:
        """Get Hanoi city boundaries."""
        return self.hanoi_bounds.copy()
    
    def get_districts(self) -> Dict[str, Dict]:
        """Get Hanoi districts information."""
        return self.districts.copy()
    
    def get_landmarks(self) -> Dict[str, Dict]:
        """Get Hanoi landmarks information."""
        return self.landmarks.copy()


def generate_hanoi_coordinates(n_customers: int,
                             clustering: str = 'random',
                             depot_location: str = 'hoan_kiem_lake') -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """
    Convenience function to generate Hanoi coordinates.
    
    Args:
        n_customers: Number of customers
        clustering: Clustering method
        depot_location: Depot landmark name
        
    Returns:
        Tuple of (customer_coordinates, depot_coordinates)
    """
    generator = HanoiCoordinateGenerator()
    return generator.generate_coordinates(n_customers, clustering, depot_location)
