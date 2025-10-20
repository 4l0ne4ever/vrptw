"""
Enhanced Hanoi coordinate generator with realistic locations and real routing.
Avoids water bodies and integrates with OpenStreetMap routing.
"""

import numpy as np
import requests
import json
from typing import List, Tuple, Dict, Optional
import random
import time

# Optional import for KMeans
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class EnhancedHanoiCoordinateGenerator:
    """Generates realistic coordinates in Hanoi with real routing support."""
    
    def __init__(self):
        """Initialize enhanced Hanoi coordinate generator."""
        # Hanoi city boundaries (more precise)
        self.hanoi_bounds = {
            'min_lat': 20.7,   # Southern boundary
            'max_lat': 21.4,   # Northern boundary  
            'min_lon': 105.3,  # Western boundary
            'max_lon': 106.0   # Eastern boundary
        }
        
        # Realistic districts with actual road networks
        self.districts = {
            'hoan_kiem': {
                'lat': 21.0285, 'lon': 105.8542, 'radius': 0.03,
                'roads': ['Phố Hàng Bạc', 'Phố Hàng Gai', 'Phố Lý Thái Tổ']
            },
            'ba_dinh': {
                'lat': 21.0333, 'lon': 105.8333, 'radius': 0.05,
                'roads': ['Đường Kim Mã', 'Đường Giảng Võ', 'Đường Đội Cấn']
            },
            'dong_da': {
                'lat': 21.0167, 'lon': 105.8333, 'radius': 0.05,
                'roads': ['Đường Giải Phóng', 'Đường Láng', 'Đường Khâm Thiên']
            },
            'hai_ba_trung': {
                'lat': 21.0167, 'lon': 105.8500, 'radius': 0.05,
                'roads': ['Đường Bạch Mai', 'Đường Minh Khai', 'Đường Trần Khát Chân']
            },
            'cau_giay': {
                'lat': 21.0333, 'lon': 105.8000, 'radius': 0.06,
                'roads': ['Đường Cầu Giấy', 'Đường Duy Tân', 'Đường Hoàng Quốc Việt']
            },
            'thanh_xuan': {
                'lat': 21.0000, 'lon': 105.8000, 'radius': 0.05,
                'roads': ['Đường Nguyễn Trãi', 'Đường Lê Văn Lương', 'Đường Khuất Duy Tiến']
            },
            'long_bien': {
                'lat': 21.0500, 'lon': 105.9000, 'radius': 0.06,
                'roads': ['Đường Nguyễn Văn Cừ', 'Đường Long Biên', 'Đường Ngọc Lâm']
            },
            'tu_liem': {
                'lat': 21.1000, 'lon': 105.7000, 'radius': 0.08,
                'roads': ['Đường Phạm Văn Đồng', 'Đường Cổ Nhuế', 'Đường Xuân Phương']
            }
        }
        
        # Water bodies to avoid (approximate coordinates)
        self.water_bodies = {
            'hoan_kiem_lake': {'lat': 21.0285, 'lon': 105.8542, 'radius': 0.01},
            'west_lake': {'lat': 21.0500, 'lon': 105.8167, 'radius': 0.08},
            'red_river': {'lat': 21.0333, 'lon': 105.8667, 'radius': 0.02},
            'duong_river': {'lat': 21.0167, 'lon': 105.9167, 'radius': 0.02}
        }
        
        # Major landmarks for depot placement
        self.landmarks = {
            'hoan_kiem_lake': {'lat': 21.0285, 'lon': 105.8542},
            'west_lake': {'lat': 21.0500, 'lon': 105.8167},
            'noi_bai_airport': {'lat': 21.2167, 'lon': 105.8000},
            'long_bien_bridge': {'lat': 21.0500, 'lon': 105.8667},
            'temple_of_literature': {'lat': 21.0267, 'lon': 105.8356},
            'dong_xuan_market': {'lat': 21.0333, 'lon': 105.8500},
            'vincom_center': {'lat': 21.0167, 'lon': 105.8333}
        }
        
        # OSRM routing server (free public server)
        self.osrm_server = "http://router.project-osrm.org"
        
    def generate_coordinates(self, n_customers: int, 
                           clustering: str = 'realistic',
                           depot_location: str = 'hoan_kiem_lake') -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
        """
        Generate realistic customer coordinates and depot location.
        
        Args:
            n_customers: Number of customers
            clustering: Clustering method ('realistic', 'district', 'road_based')
            depot_location: Depot landmark name
            
        Returns:
            Tuple of (customer_coordinates, depot_coordinates)
        """
        # Generate depot coordinates
        depot_coords = self._generate_realistic_depot_coordinates(depot_location)
        
        # Generate customer coordinates based on clustering method
        if clustering == 'realistic':
            customer_coords = self._generate_realistic_coordinates(n_customers, depot_coords)
        elif clustering == 'district':
            customer_coords = self._generate_district_coordinates(n_customers)
        elif clustering == 'road_based':
            customer_coords = self._generate_road_based_coordinates(n_customers)
        else:
            customer_coords = self._generate_realistic_coordinates(n_customers, depot_coords)
        
        return customer_coords, depot_coords
    
    def _generate_realistic_depot_coordinates(self, location: str) -> Tuple[float, float]:
        """Generate realistic depot coordinates at specified landmark."""
        if location in self.landmarks:
            landmark = self.landmarks[location]
            # Add small random offset to avoid exact landmark placement
            lat_offset = np.random.normal(0, 0.002)  # Smaller offset
            lon_offset = np.random.normal(0, 0.002)
            return (landmark['lat'] + lat_offset, landmark['lon'] + lon_offset)
        else:
            # Default to Hoan Kiem Lake
            return self.landmarks['hoan_kiem_lake']['lat'], self.landmarks['hoan_kiem_lake']['lon']
    
    def _generate_realistic_coordinates(self, n_customers: int, depot_coords: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Generate realistic coordinates avoiding water bodies."""
        coordinates = []
        attempts = 0
        max_attempts = n_customers * 10  # Allow multiple attempts
        
        while len(coordinates) < n_customers and attempts < max_attempts:
            attempts += 1
            
            # Generate coordinates in realistic areas
            if len(coordinates) < n_customers // 3:
                # Near depot area
                lat_offset = np.random.normal(0, 0.02)
                lon_offset = np.random.normal(0, 0.02)
                lat = depot_coords[0] + lat_offset
                lon = depot_coords[1] + lon_offset
            elif len(coordinates) < 2 * n_customers // 3:
                # In major districts
                district_name = random.choice(list(self.districts.keys()))
                district = self.districts[district_name]
                lat_offset = np.random.normal(0, district['radius'] * 0.5)
                lon_offset = np.random.normal(0, district['radius'] * 0.5)
                lat = district['lat'] + lat_offset
                lon = district['lon'] + lon_offset
            else:
                # Random but realistic areas
                lat = np.random.uniform(self.hanoi_bounds['min_lat'] + 0.1, 
                                     self.hanoi_bounds['max_lat'] - 0.1)
                lon = np.random.uniform(self.hanoi_bounds['min_lon'] + 0.1, 
                                     self.hanoi_bounds['max_lon'] - 0.1)
            
            # Ensure within Hanoi bounds
            lat = np.clip(lat, self.hanoi_bounds['min_lat'], self.hanoi_bounds['max_lat'])
            lon = np.clip(lon, self.hanoi_bounds['min_lon'], self.hanoi_bounds['max_lon'])
            
            # Check if coordinates avoid water bodies
            if self._is_realistic_location(lat, lon):
                coordinates.append((lat, lon))
        
        # If we couldn't generate enough realistic coordinates, fill with district-based
        while len(coordinates) < n_customers:
            district_name = random.choice(list(self.districts.keys()))
            district = self.districts[district_name]
            lat_offset = np.random.normal(0, district['radius'] * 0.3)
            lon_offset = np.random.normal(0, district['radius'] * 0.3)
            lat = district['lat'] + lat_offset
            lon = district['lon'] + lon_offset
            
            lat = np.clip(lat, self.hanoi_bounds['min_lat'], self.hanoi_bounds['max_lat'])
            lon = np.clip(lon, self.hanoi_bounds['min_lon'], self.hanoi_bounds['max_lon'])
            
            coordinates.append((lat, lon))
        
        return coordinates
    
    def _is_realistic_location(self, lat: float, lon: float) -> bool:
        """Check if location is realistic (not in water bodies)."""
        for water_name, water in self.water_bodies.items():
            distance = np.sqrt((lat - water['lat'])**2 + (lon - water['lon'])**2)
            if distance < water['radius']:
                return False
        return True
    
    def _generate_district_coordinates(self, n_customers: int) -> List[Tuple[float, float]]:
        """Generate coordinates distributed across Hanoi districts."""
        coordinates = []
        district_names = list(self.districts.keys())
        
        for i in range(n_customers):
            # Select random district
            district_name = random.choice(district_names)
            district = self.districts[district_name]
            
            # Generate coordinates within district
            lat_offset = np.random.normal(0, district['radius'] * 0.4)
            lon_offset = np.random.normal(0, district['radius'] * 0.4)
            
            lat = district['lat'] + lat_offset
            lon = district['lon'] + lon_offset
            
            # Ensure within Hanoi bounds
            lat = np.clip(lat, self.hanoi_bounds['min_lat'], self.hanoi_bounds['max_lat'])
            lon = np.clip(lon, self.hanoi_bounds['min_lon'], self.hanoi_bounds['max_lon'])
            
            coordinates.append((lat, lon))
        
        return coordinates
    
    def _generate_road_based_coordinates(self, n_customers: int) -> List[Tuple[float, float]]:
        """Generate coordinates along major roads."""
        coordinates = []
        
        # Major roads in Hanoi
        major_roads = [
            {'lat': 21.0285, 'lon': 105.8542, 'name': 'Phố Hàng Bạc'},
            {'lat': 21.0333, 'lon': 105.8333, 'name': 'Đường Kim Mã'},
            {'lat': 21.0167, 'lon': 105.8333, 'name': 'Đường Giải Phóng'},
            {'lat': 21.0167, 'lon': 105.8500, 'name': 'Đường Bạch Mai'},
            {'lat': 21.0333, 'lon': 105.8000, 'name': 'Đường Cầu Giấy'},
            {'lat': 21.0000, 'lon': 105.8000, 'name': 'Đường Nguyễn Trãi'},
            {'lat': 21.0500, 'lon': 105.9000, 'name': 'Đường Nguyễn Văn Cừ'},
            {'lat': 21.1000, 'lon': 105.7000, 'name': 'Đường Phạm Văn Đồng'}
        ]
        
        for i in range(n_customers):
            road = random.choice(major_roads)
            
            # Generate coordinates along road
            lat_offset = np.random.normal(0, 0.01)
            lon_offset = np.random.normal(0, 0.01)
            
            lat = road['lat'] + lat_offset
            lon = road['lon'] + lon_offset
            
            # Ensure within Hanoi bounds
            lat = np.clip(lat, self.hanoi_bounds['min_lat'], self.hanoi_bounds['max_lat'])
            lon = np.clip(lon, self.hanoi_bounds['min_lon'], self.hanoi_bounds['max_lon'])
            
            coordinates.append((lat, lon))
        
        return coordinates
    
    def get_real_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Get real route between two points using OSRM routing.
        
        Args:
            start: Start coordinates (lat, lon)
            end: End coordinates (lat, lon)
            
        Returns:
            List of coordinates forming the route, or None if failed
        """
        try:
            # OSRM API call
            url = f"{self.osrm_server}/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 'Ok' and len(data['routes']) > 0:
                    route = data['routes'][0]
                    geometry = route['geometry']['coordinates']
                    # Convert from [lon, lat] to [lat, lon]
                    return [(coord[1], coord[0]) for coord in geometry]
            
            return None
            
        except Exception as e:
            print(f"Routing error: {e}")
            return None
    
    def get_distance_matrix_real_routes(self, coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """
        Calculate distance matrix using real routes.
        
        Args:
            coordinates: List of (lat, lon) coordinates
            
        Returns:
            Distance matrix with real route distances
        """
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))
        
        print(f"Calculating real route distances for {n} locations...")
        
        for i in range(n):
            for j in range(i + 1, n):
                # Get real route
                route = self.get_real_route(coordinates[i], coordinates[j])
                
                if route:
                    # Calculate route distance
                    distance = self._calculate_route_distance(route)
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance
                else:
                    # Fallback to Euclidean distance
                    distance = self._calculate_euclidean_distance(coordinates[i], coordinates[j])
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance
                
                # Add small delay to avoid overwhelming the server
                time.sleep(0.1)
        
        return distance_matrix
    
    def _calculate_route_distance(self, route: List[Tuple[float, float]]) -> float:
        """Calculate total distance of a route."""
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self._calculate_euclidean_distance(route[i], route[i + 1])
        return total_distance
    
    def _calculate_euclidean_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two coordinates."""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Haversine formula for accurate distance
        R = 6371  # Earth's radius in kilometers
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2) * np.sin(dlat/2) + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2) * np.sin(dlon/2))
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def get_hanoi_bounds(self) -> Dict[str, float]:
        """Get Hanoi city boundaries."""
        return self.hanoi_bounds.copy()
    
    def get_districts(self) -> Dict[str, Dict]:
        """Get Hanoi districts information."""
        return self.districts.copy()
    
    def get_landmarks(self) -> Dict[str, Dict]:
        """Get Hanoi landmarks information."""
        return self.landmarks.copy()
    
    def get_water_bodies(self) -> Dict[str, Dict]:
        """Get water bodies to avoid."""
        return self.water_bodies.copy()


def generate_enhanced_hanoi_coordinates(n_customers: int,
                                      clustering: str = 'realistic',
                                      depot_location: str = 'hoan_kiem_lake') -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """
    Convenience function to generate enhanced Hanoi coordinates.
    
    Args:
        n_customers: Number of customers
        clustering: Clustering method
        depot_location: Depot landmark name
        
    Returns:
        Tuple of (customer_coordinates, depot_coordinates)
    """
    generator = EnhancedHanoiCoordinateGenerator()
    return generator.generate_coordinates(n_customers, clustering, depot_location)
