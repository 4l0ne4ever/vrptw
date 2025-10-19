"""
Distance calculation and caching for VRP problems.
Handles Euclidean distance with traffic factors.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import pickle
from config import VRP_CONFIG


class DistanceCalculator:
    """Calculates and caches distance matrices for VRP problems."""
    
    def __init__(self, traffic_factor: Optional[float] = None):
        """
        Initialize distance calculator.
        
        Args:
            traffic_factor: Multiplier for distance calculation (default from config)
        """
        self.traffic_factor = traffic_factor or VRP_CONFIG['traffic_factor']
        self.distance_matrix = None
        self.coordinates = None
        self.cache_dir = 'data/processed'
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def calculate_distance_matrix(self, 
                                coordinates: List[Tuple[float, float]],
                                use_cache: bool = True) -> np.ndarray:
        """
        Calculate distance matrix between all points.
        
        Args:
            coordinates: List of (x, y) coordinate tuples
            use_cache: Whether to use cached matrix if available
            
        Returns:
            Distance matrix as numpy array
        """
        n_points = len(coordinates)
        
        # Check cache first
        if use_cache:
            cached_matrix = self._load_from_cache(coordinates)
            if cached_matrix is not None:
                self.distance_matrix = cached_matrix
                self.coordinates = coordinates
                return self.distance_matrix
        
        # Calculate new matrix
        self.distance_matrix = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    self.distance_matrix[i, j] = 0.0
                else:
                    dist = self._euclidean_distance(
                        coordinates[i], coordinates[j]
                    )
                    self.distance_matrix[i, j] = dist * self.traffic_factor
        
        self.coordinates = coordinates
        
        # Save to cache
        if use_cache:
            self._save_to_cache()
        
        return self.distance_matrix
    
    def _euclidean_distance(self, 
                          point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_distance(self, from_idx: int, to_idx: int) -> float:
        """
        Get distance between two points by index.
        
        Args:
            from_idx: Source point index
            to_idx: Destination point index
            
        Returns:
            Distance between points
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not calculated yet")
        
        return self.distance_matrix[from_idx, to_idx]
    
    def get_route_distance(self, route: List[int]) -> float:
        """
        Calculate total distance for a route.
        
        Args:
            route: List of point indices representing the route
            
        Returns:
            Total route distance
        """
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self.get_distance(route[i], route[i + 1])
        
        return total_distance
    
    def _get_cache_filename(self, coordinates: List[Tuple[float, float]]) -> str:
        """Generate cache filename based on coordinates."""
        # Create hash from coordinates
        coords_str = str(sorted(coordinates))
        coords_hash = hash(coords_str) % (2**32)
        
        return os.path.join(self.cache_dir, f"dist_matrix_{coords_hash}.pkl")
    
    def _save_to_cache(self):
        """Save distance matrix to cache."""
        if self.distance_matrix is None or self.coordinates is None:
            return
        
        cache_file = self._get_cache_filename(self.coordinates)
        
        cache_data = {
            'distance_matrix': self.distance_matrix,
            'coordinates': self.coordinates,
            'traffic_factor': self.traffic_factor
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_from_cache(self, coordinates: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """Load distance matrix from cache if available."""
        cache_file = self._get_cache_filename(coordinates)
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if coordinates match
            if (cache_data['coordinates'] == coordinates and 
                cache_data['traffic_factor'] == self.traffic_factor):
                return cache_data['distance_matrix']
            
        except (pickle.PickleError, KeyError, EOFError):
            # Cache file corrupted, remove it
            os.remove(cache_file)
        
        return None
    
    def clear_cache(self):
        """Clear all cached distance matrices."""
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.startswith('dist_matrix_') and file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))


def calculate_distance_matrix(coordinates: List[Tuple[float, float]], 
                           traffic_factor: float = 1.0) -> np.ndarray:
    """
    Convenience function to calculate distance matrix.
    
    Args:
        coordinates: List of (x, y) coordinate tuples
        traffic_factor: Multiplier for distance calculation
        
    Returns:
        Distance matrix as numpy array
    """
    calculator = DistanceCalculator(traffic_factor)
    return calculator.calculate_distance_matrix(coordinates)
