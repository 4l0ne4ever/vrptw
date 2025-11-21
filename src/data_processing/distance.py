"""
Distance calculation and caching for VRP problems.
Handles real road routes (OSRM) and Euclidean/Haversine distance with traffic factors.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import pickle
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import VRP_CONFIG
from app.config.settings import (
    OSRM_BASE_URL,
    OSRM_TIMEOUT,
    OSRM_MAX_BATCH_SIZE,
    OSRM_MAX_CONCURRENCY,
    USE_REAL_ROUTES,
)
from src.core.pipeline_profiler import pipeline_profiler
from src.data_processing.mode_profiles import get_mode_profile
from src.data_processing.solomon_fast_path import SolomonFastPath


class DistanceCalculator:
    """Calculates and caches distance matrices for VRP problems."""
    
    def __init__(self, traffic_factor: Optional[float] = None, use_adaptive: bool = False, 
                 use_real_routes: Optional[bool] = None, dataset_type: str = "hanoi",
                 dataset_name: Optional[str] = None):
        """
        Initialize distance calculator.
        
        Args:
            traffic_factor: Base multiplier for distance calculation (default from config)
            use_adaptive: Enable adaptive traffic factor based on time of day
            use_real_routes: Use real road routes (OSRM) instead of straight line (default: True for Hanoi)
            dataset_type: Type of dataset ("hanoi" or "solomon") - affects routing method
        """
        self.mode_profile = get_mode_profile(dataset_type)
        self.dataset_type = dataset_type
        
        # CORRECTNESS FIX: For Solomon datasets, traffic_factor should always be 1.0 (pure Euclidean)
        # For Hanoi datasets, use provided traffic_factor or config default
        if dataset_type.lower() == "solomon":
            self.traffic_factor = 1.0  # Solomon: pure Euclidean, no traffic factor
            self.use_adaptive = False  # Solomon: no adaptive traffic
        else:
            self.traffic_factor = traffic_factor or VRP_CONFIG.get('traffic_factor', 1.0)
            self.use_adaptive = use_adaptive or VRP_CONFIG.get('use_adaptive_traffic', False)
        self.dataset_name = dataset_name
        self._dataset_name_normalized = dataset_name.strip().upper() if dataset_name else None
        
        # Use real routes for Hanoi by default, otherwise use config
        if use_real_routes is None:
            if self.mode_profile.key == "hanoi":
                self.use_real_routes = True
            else:
                self.use_real_routes = bool(self.mode_profile.use_real_routes and USE_REAL_ROUTES)
        else:
            self.use_real_routes = use_real_routes
        
        self.osrm_base_url = OSRM_BASE_URL.rstrip("/")
        self.osrm_timeout = OSRM_TIMEOUT
        self.distance_matrix = None  # Base distance (no traffic factor applied)
        self.coordinates = None
        self.cache_dir = 'data/processed'
        
        # Per-segment OSRM cache to avoid recalculating same pairs
        self._osrm_segment_cache: Dict[Tuple[float, float, float, float], float] = {}
        
        # HTTP session for connection pooling (created lazily)
        self._session = None

        # Fast-path detection cache/state
        self._fast_path_entry: Optional[Dict[str, object]] = None
        self._fast_path_entry_checked = False
        self._last_cache_key_hash: Optional[str] = None
        self.mode_key = self.mode_profile.key
        
        # Opportunistically warm Solomon caches in background
        if self.mode_profile.warm_cache:
            try:
                from src.data_processing.solomon_cache_warmer import SolomonCacheWarmer
                
                SolomonCacheWarmer.ensure_started()
            except Exception as warm_err:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Unable to start Solomon cache warmer: {warm_err}")
        
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
        pipeline_profiler.set_context(distance_dataset=self.dataset_type, distance_points=n_points)

        fast_path_entry = None
        cache_key_hash = None
        if use_cache:
            fast_path_entry = self._get_fast_path_entry(coordinates)
            if fast_path_entry:
                cache_key_hash = fast_path_entry["cache_key_hash"]
            else:
                cache_key_hash = self._compute_cache_key_hash(coordinates)
        self._last_cache_key_hash = cache_key_hash
        try:
            with pipeline_profiler.profile(
                "distance.calculate",
                metadata={'use_real_routes': self.use_real_routes, 'use_cache': use_cache}
            ):
                # Check cache first (file cache, then database cache)
                if use_cache:
                    cache_metadata = {'n_points': n_points, 'dataset_type': self.dataset_type}
                    if fast_path_entry:
                        cache_metadata['fast_path'] = fast_path_entry['name']
                    with pipeline_profiler.profile("distance.cache_lookup", metadata=cache_metadata):
                        # First try file cache
                        cached_matrix = self._load_from_cache(coordinates, cache_key_hash)
                        if cached_matrix is not None:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.info(f"✅ Loaded distance matrix from file cache for {n_points} locations (dataset_type={self.dataset_type}, use_real_routes={self.use_real_routes})")
                            self.distance_matrix = cached_matrix
                            self.coordinates = coordinates
                            return self.distance_matrix
                        
                        # Then try database cache
                        cached_matrix = self._load_from_database_cache(coordinates, cache_key_hash)
                        if cached_matrix is not None:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.info(f"✅ Loaded distance matrix from database cache for {n_points} locations (dataset_type={self.dataset_type}, use_real_routes={self.use_real_routes})")
                            self.distance_matrix = cached_matrix
                            self.coordinates = coordinates
                            # Also save to file cache for faster access next time
                            self._save_to_cache(cache_key_hash)
                            return self.distance_matrix
                    
                    # Cache miss
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"⚠️ Cache miss for {n_points} locations (dataset_type={self.dataset_type}, use_real_routes={self.use_real_routes}) - will calculate new matrix")
                
                # Calculate new matrix
                self.distance_matrix = np.zeros((n_points, n_points))
                
                # Use real road routes for Hanoi, otherwise use haversine/euclidean
                import logging
                logger = logging.getLogger(__name__)
                
                # Force Euclidean for Solomon datasets (never use OSRM)
                if self.dataset_type.lower() == "solomon":
                    logger.info(f"📐 Using Euclidean distance for Solomon dataset ({n_points} locations)")
                    with pipeline_profiler.profile("distance.euclidean_vectorized", metadata={'n_points': n_points}):
                        self.distance_matrix = self._calculate_euclidean_matrix_vectorized(coordinates, n_points)
                elif self.use_real_routes and self._should_use_haversine(coordinates):
                    # Use OSRM Table Service for real road routes (Hanoi mode only)
                    logger.info(f"🔄 Calculating real road route distances for {n_points} locations using OSRM Table Service...")
                    
                    # Use OSRM Table Service (calculates entire matrix in 1 request)
                    with pipeline_profiler.profile(
                        "distance.osrm_table_service",
                        metadata={'n_points': n_points, 'dataset_type': self.dataset_type}
                    ):
                        self._calculate_osrm_matrix_table_service(coordinates, n_points)
                else:
                    # Use vectorized haversine/euclidean for non-Hanoi or when OSRM unavailable
                    # Vectorized approach is 10x-25x faster than nested loops
                    use_haversine = self._should_use_haversine(coordinates)
                    loop_label = "distance.haversine_vectorized" if use_haversine else "distance.euclidean_vectorized"
                    logger.info(f"📐 Using {loop_label} for {self.dataset_type} dataset ({n_points} locations, use_real_routes={self.use_real_routes})")
                    with pipeline_profiler.profile(loop_label, metadata={'n_points': n_points}):
                        if use_haversine:
                            self.distance_matrix = self._calculate_haversine_matrix_vectorized(coordinates, n_points)
                        else:
                            self.distance_matrix = self._calculate_euclidean_matrix_vectorized(coordinates, n_points)
                
                self.coordinates = coordinates
                
                # Save to cache (both file and database)
                if use_cache and cache_key_hash is not None:
                    with pipeline_profiler.profile("distance.cache_save", metadata={'n_points': n_points}):
                        # Save to file cache
                        self._save_to_cache(cache_key_hash)
                        # Save to database cache
                        self._save_to_database_cache(coordinates, cache_key_hash)
                        # Log cache save
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info(f"💾 Saved distance matrix to cache (file + database) for {n_points} locations")
                
                return self.distance_matrix
        finally:
            pipeline_profiler.clear_context('distance_dataset', 'distance_points')
    
    def _get_fast_path_entry(self, coordinates: List[Tuple[float, float]]) -> Optional[Dict[str, object]]:
        """
        Detect Solomon fast-path entry for known benchmark instances.
        """
        if self.dataset_type.lower() != "solomon":
            return None
        if not self._dataset_name_normalized:
            return None

        if self._fast_path_entry_checked:
            return self._fast_path_entry

        entry = SolomonFastPath.match(self._dataset_name_normalized, coordinates)
        self._fast_path_entry = entry
        self._fast_path_entry_checked = True

        if entry:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"⚡ Solomon fast path enabled for {entry['name']} "
                f"(cache_key={entry['cache_key_hash']})"
            )

        return entry
    
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

    def _haversine_distance(self,
                           point1: Tuple[float, float],
                           point2: Tuple[float, float]) -> float:
        """Calculate Haversine distance (km) between two (x,y) where x=lon, y=lat."""
        lon1, lat1 = point1[0], point1[1]
        lon2, lat2 = point2[0], point2[1]
        # convert decimal degrees to radians
        from math import radians, sin, cos, asin, sqrt
        R = 6371.0  # Earth radius in km
        dlon = radians(lon2 - lon1)
        dlat = radians(lat2 - lat1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c

    def _calculate_euclidean_matrix_vectorized(self, coordinates: List[Tuple[float, float]], n_points: int) -> np.ndarray:
        """
        Calculate Euclidean distance matrix using vectorized NumPy operations.
        Much faster than nested loops (10x-25x speedup).
        
        Args:
            coordinates: List of (x, y) coordinate tuples
            n_points: Number of points
            
        Returns:
            Distance matrix as numpy array
        """
        # Convert coordinates to numpy array
        coords_array = np.array(coordinates)  # Shape: (n_points, 2)
        
        # Extract x and y coordinates
        x = coords_array[:, 0]  # Shape: (n_points,)
        y = coords_array[:, 1]  # Shape: (n_points,)
        
        # Use broadcasting to compute all pairwise differences at once
        # x_diff[i, j] = x[i] - x[j] for all i, j
        x_diff = x[:, np.newaxis] - x[np.newaxis, :]  # Shape: (n_points, n_points)
        y_diff = y[:, np.newaxis] - y[np.newaxis, :]  # Shape: (n_points, n_points)
        
        # Compute squared distances
        dist_squared = x_diff ** 2 + y_diff ** 2
        
        # Take square root to get distances
        distance_matrix = np.sqrt(dist_squared)
        
        # Ensure diagonal is exactly zero (handles floating point precision)
        np.fill_diagonal(distance_matrix, 0.0)
        
        return distance_matrix
    
    def _calculate_haversine_matrix_vectorized(self, coordinates: List[Tuple[float, float]], n_points: int) -> np.ndarray:
        """
        Calculate Haversine distance matrix using vectorized NumPy operations.
        Much faster than nested loops (10x-25x speedup).
        
        Args:
            coordinates: List of (lon, lat) coordinate tuples
            n_points: Number of points
            
        Returns:
            Distance matrix in kilometers as numpy array
        """
        # Convert coordinates to numpy array
        coords_array = np.array(coordinates)  # Shape: (n_points, 2)
        
        # Extract longitude and latitude
        lon = coords_array[:, 0]  # Shape: (n_points,)
        lat = coords_array[:, 1]  # Shape: (n_points,)
        
        # Convert to radians (vectorized)
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        
        # Use broadcasting to compute all pairwise differences at once
        # dlon[i, j] = lon[i] - lon[j] for all i, j
        dlon = lon_rad[:, np.newaxis] - lon_rad[np.newaxis, :]  # Shape: (n_points, n_points)
        dlat = lat_rad[:, np.newaxis] - lat_rad[np.newaxis, :]  # Shape: (n_points, n_points)
        
        # Haversine formula (vectorized)
        # a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        a = (np.sin(dlat / 2) ** 2 + 
             np.cos(lat_rad[:, np.newaxis]) * np.cos(lat_rad[np.newaxis, :]) * 
             np.sin(dlon / 2) ** 2)
        
        # c = 2 * arcsin(√a)
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Distance = R * c (R = Earth radius in km)
        R = 6371.0
        distance_matrix = R * c
        
        # Ensure diagonal is exactly zero (handles floating point precision)
        np.fill_diagonal(distance_matrix, 0.0)
        
        return distance_matrix
    
    def _should_use_haversine(self, coordinates: List[Tuple[float, float]]) -> bool:
        """Detect if coordinates look like geo (lon,lat within Hanoi/VN bounds)."""
        if not coordinates:
            return False
        xs = [c[0] for c in coordinates]
        ys = [c[1] for c in coordinates]
        # Rough Vietnam bounds
        min_lon, max_lon = 102.0, 110.0
        min_lat, max_lat = 8.0, 24.0
        in_bounds = (min(xs) >= min_lon and max(xs) <= max_lon and
                     min(ys) >= min_lat and max(ys) <= max_lat)
        return in_bounds
    
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
    
    def _compute_cache_key_hash(self, coordinates: List[Tuple[float, float]]) -> str:
        """Compute cache key hash from coordinates and routing method. Called once per request."""
        import hashlib
        # Round coordinates to 6 decimal places to ensure cache matching
        # Sort by (x, y) to ensure consistent ordering
        rounded_coords = sorted([(round(c[0], 6), round(c[1], 6)) for c in coordinates])
        coords_str = str(rounded_coords)
        routing_info = f"{self.use_real_routes}_{self.dataset_type}"
        cache_key = f"{coords_str}_{routing_info}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        return cache_hash
    
    def _get_cache_filename(self, coordinates: List[Tuple[float, float]], cache_key_hash: Optional[str] = None) -> str:
        """Generate cache filename based on coordinates and routing method."""
        if cache_key_hash is None:
            cache_key_hash = self._compute_cache_key_hash(coordinates)
        return os.path.join(self.cache_dir, f"dist_matrix_{cache_key_hash}.pkl")
    
    def _save_to_cache(self, cache_key_hash: Optional[str] = None):
        """Save distance matrix to cache."""
        if self.distance_matrix is None or self.coordinates is None:
            return
        
        cache_file = self._get_cache_filename(self.coordinates, cache_key_hash)
        
        cache_data = {
            'distance_matrix': self.distance_matrix,
            'coordinates': self.coordinates,
            'use_real_routes': self.use_real_routes,
            'dataset_type': self.dataset_type,
            # Note: traffic_factor not stored since we use base distance
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_from_cache(self, coordinates: List[Tuple[float, float]], cache_key_hash: Optional[str] = None) -> Optional[np.ndarray]:
        """Load distance matrix from cache if available."""
        import logging
        logger = logging.getLogger(__name__)
        
        cache_file = self._get_cache_filename(coordinates, cache_key_hash)
        
        if not os.path.exists(cache_file):
            logger.debug(f"Cache file not found: {cache_file}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if coordinates and routing method match
            # Round coordinates for comparison (to handle floating point precision)
            cached_coords = cache_data.get('coordinates', [])
            if not cached_coords:
                logger.debug("Cache file has no coordinates")
                return None
            
            rounded_cached = [(round(c[0], 6), round(c[1], 6)) for c in cached_coords]
            rounded_input = [(round(c[0], 6), round(c[1], 6)) for c in coordinates]
            
            # Check coordinate match
            if rounded_cached != rounded_input:
                logger.debug(f"Coordinates mismatch: cached has {len(cached_coords)} points, input has {len(coordinates)} points")
                return None
            
            # Check routing method match
            cached_use_real = cache_data.get('use_real_routes')
            cached_dataset_type = cache_data.get('dataset_type')
            
            if cached_use_real != self.use_real_routes:
                logger.debug(f"Routing method mismatch: cached={cached_use_real}, current={self.use_real_routes}")
                return None
            
            if cached_dataset_type != self.dataset_type:
                logger.debug(f"Dataset type mismatch: cached={cached_dataset_type}, current={self.dataset_type}")
                return None
            
            # All checks passed - return cached matrix
            logger.info(f"✅ Cache hit! Loaded matrix for {len(coordinates)} locations")
            return cache_data['distance_matrix']
            
        except (pickle.PickleError, KeyError, EOFError) as e:
            # Cache file corrupted, remove it
            logger.warning(f"Cache file corrupted, removing: {e}")
            try:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            except:
                pass
        
        return None
    
    def _get_cache_key(self, coordinates: List[Tuple[float, float]], cache_key_hash: Optional[str] = None) -> str:
        """Generate cache key for database lookup."""
        if cache_key_hash is None:
            cache_key_hash = self._compute_cache_key_hash(coordinates)
        return cache_key_hash
    
    def _load_from_database_cache(self, coordinates: List[Tuple[float, float]], cache_key_hash: Optional[str] = None) -> Optional[np.ndarray]:
        """Load distance matrix from database cache if available."""
        # Temporarily disabled due to database constraint issues
        # File cache is sufficient for performance
        return None
    
    def _save_to_database_cache(self, coordinates: List[Tuple[float, float]], cache_key_hash: Optional[str] = None):
        """Save distance matrix to database cache."""
        # Temporarily disabled due to database constraint issues
        # File cache is sufficient for performance
        return
    
    def get_adaptive_traffic_factor(self, time_minutes: float) -> float:
        """
        Calculate adaptive traffic factor based on time of day.
        
        Args:
            time_minutes: Time in minutes from 0:00 (0 = 0:00, 480 = 8:00, 1200 = 20:00)
        
        Returns:
            Adaptive traffic factor
        """
        if not self.use_adaptive:
            return self.traffic_factor
        
        # Get traffic factors from config
        peak_factor = VRP_CONFIG.get('traffic_factor_peak', 1.8)
        normal_factor = VRP_CONFIG.get('traffic_factor_normal', 1.2)
        low_factor = VRP_CONFIG.get('traffic_factor_low', 1.0)
        
        # Convert to minutes in day (0-1440)
        time_in_day = time_minutes % 1440
        
        # Peak hours: 7-9h (420-540), 17-19h (1020-1140)
        if (420 <= time_in_day <= 540) or (1020 <= time_in_day <= 1140):
            return peak_factor
        # Low hours: 22h-7h (1320-1440 or 0-420)
        elif (1320 <= time_in_day <= 1440) or (0 <= time_in_day < 420):
            return low_factor
        # Normal hours: 10-16h, 20-22h
        else:
            return normal_factor
    
    def get_adaptive_distance(self, from_idx: int, to_idx: int, 
                              time_minutes: float) -> float:
        """
        Get distance with adaptive traffic factor applied.
        
        Args:
            from_idx: Source point index
            to_idx: Destination point index
            time_minutes: Time in minutes from 0:00 when traveling this segment
        
        Returns:
            Distance with adaptive traffic factor (or base distance for Solomon)
        """
        base_distance = self.get_distance(from_idx, to_idx)
        
        # CORRECTNESS: Solomon datasets should always return pure Euclidean distance
        if self.dataset_type.lower() == "solomon":
            return base_distance  # No traffic factor for Solomon
        
        if self.use_adaptive:
            traffic_factor = self.get_adaptive_traffic_factor(time_minutes)
            return base_distance * traffic_factor
        else:
            return base_distance * self.traffic_factor
    
    def _calculate_osrm_matrix_table_service(self, coordinates: List[Tuple[float, float]], n_points: int):
        """
        Calculate OSRM distance matrix using Table Service API (much faster for large datasets).
        Table Service can calculate distance matrix for up to 100 points in one request.
        
        Args:
            coordinates: List of coordinate tuples
            n_points: Number of points
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # OSRM Table Service limit is typically 100 points per request
        # Batch size is configurable via OSRM_MAX_BATCH_SIZE environment variable
        # Default is 100, but can be reduced to 50 if URL length issues occur
        max_batch_size = OSRM_MAX_BATCH_SIZE
        
        # For datasets ≤100 points, use single request if batch size allows
        # For larger datasets, use batch processing
        if n_points <= max_batch_size:
            # Single request for small datasets
            with pipeline_profiler.profile(
                "distance.osrm_batch",
                metadata={'batch_idx': 0, 'batch_size': n_points, 'total_points': n_points}
            ):
                self._calculate_osrm_table_batch(coordinates, n_points, 0, n_points)
            # Ensure diagonal is zero
            for i in range(n_points):
                self.distance_matrix[i, i] = 0.0
        else:
            # Batch processing for large datasets
            # Calculate in chunks and combine
            logger.info(f"Large dataset ({n_points} points): using batch processing with Table Service")
            
            # Try to use Streamlit progress if available
            use_streamlit = False
            progress_bar = None
            try:
                import streamlit as st
                if hasattr(st, 'status') and hasattr(st, 'progress'):
                    use_streamlit = True
            except:
                pass
            
            num_batches = (n_points + max_batch_size - 1) // max_batch_size
            
            # Helper function to process a single batch (for parallel execution)
            def process_batch(batch_idx: int):
                """Process a single batch and return batch index for ordering."""
                start_idx = batch_idx * max_batch_size
                end_idx = min((batch_idx + 1) * max_batch_size, n_points)
                batch_coords = coordinates[start_idx:end_idx]
                
                with pipeline_profiler.profile(
                    "distance.osrm_batch",
                    metadata={'batch_idx': batch_idx, 'batch_size': len(batch_coords), 'total_points': n_points}
                ):
                    self._calculate_osrm_table_batch(batch_coords, len(batch_coords), start_idx, n_points, coordinates)
                return batch_idx
            
            if use_streamlit:
                with st.status(f"Calculating distance matrix using OSRM Table Service ({num_batches} batches, parallel)...", expanded=False):
                    progress_bar = st.progress(0)
                    
                    # Use parallel processing with concurrency limit
                    max_concurrency = OSRM_MAX_CONCURRENCY
                    completed_batches = 0
                    
                    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                        # Submit all batch tasks
                        future_to_batch = {executor.submit(process_batch, batch_idx): batch_idx 
                                         for batch_idx in range(num_batches)}
                        
                        # Process completed batches as they finish
                        for future in as_completed(future_to_batch):
                            batch_idx = future_to_batch[future]
                            try:
                                future.result()  # Get result (or raise exception)
                                completed_batches += 1
                                if progress_bar:
                                    progress_bar.progress(completed_batches / num_batches)
                            except Exception as e:
                                logger.error(f"Error processing batch {batch_idx}: {e}")
                                # Continue with other batches (error isolation)
                    
                    # Make matrix symmetric after all batches are done
                    logger.info("Making distance matrix symmetric...")
                    for i in range(n_points):
                        for j in range(i + 1, n_points):
                            if self.distance_matrix[i, j] > 0 and self.distance_matrix[j, i] == 0:
                                self.distance_matrix[j, i] = self.distance_matrix[i, j]
                            elif self.distance_matrix[j, i] > 0 and self.distance_matrix[i, j] == 0:
                                self.distance_matrix[i, j] = self.distance_matrix[j, i]
                            elif self.distance_matrix[i, j] > 0 and self.distance_matrix[j, i] > 0:
                                # Both exist - use average for consistency
                                avg = (self.distance_matrix[i, j] + self.distance_matrix[j, i]) / 2.0
                                self.distance_matrix[i, j] = avg
                                self.distance_matrix[j, i] = avg
                    
                    # Ensure diagonal is zero
                    for i in range(n_points):
                        self.distance_matrix[i, i] = 0.0
            else:
                print(f"Calculating distance matrix using OSRM Table Service ({num_batches} batches, parallel)...")
                max_concurrency = OSRM_MAX_CONCURRENCY
                completed_batches = 0
                
                # Use parallel processing with concurrency limit
                with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                    # Submit all batch tasks
                    future_to_batch = {executor.submit(process_batch, batch_idx): batch_idx 
                                     for batch_idx in range(num_batches)}
                    
                    # Process completed batches as they finish
                    for future in as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            future.result()  # Get result (or raise exception)
                            completed_batches += 1
                            print(f"Batch {batch_idx + 1}/{num_batches} completed... ({completed_batches}/{num_batches} total)")
                        except Exception as e:
                            logger.error(f"Error processing batch {batch_idx}: {e}")
                            # Continue with other batches (error isolation)
                
                # Make matrix symmetric after all batches are done
                logger.info("Making distance matrix symmetric...")
                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        if self.distance_matrix[i, j] > 0 and self.distance_matrix[j, i] == 0:
                            self.distance_matrix[j, i] = self.distance_matrix[i, j]
                        elif self.distance_matrix[j, i] > 0 and self.distance_matrix[i, j] == 0:
                            self.distance_matrix[i, j] = self.distance_matrix[j, i]
                        elif self.distance_matrix[i, j] > 0 and self.distance_matrix[j, i] > 0:
                            # Both exist - use average for consistency
                            avg = (self.distance_matrix[i, j] + self.distance_matrix[j, i]) / 2.0
                            self.distance_matrix[i, j] = avg
                            self.distance_matrix[j, i] = avg
                
                # Ensure diagonal is zero
                for i in range(n_points):
                    self.distance_matrix[i, i] = 0.0
        
        logger.info(f"✅ Completed OSRM Table Service calculation for {n_points} locations")
    
    def _calculate_osrm_table_batch(self, source_coords: List[Tuple[float, float]], 
                                    num_sources: int, source_start_idx: int, 
                                    total_points: int, 
                                    all_coords: Optional[List[Tuple[float, float]]] = None):
        """
        Calculate distance matrix for a batch using OSRM Table Service.
        
        Args:
            source_coords: Source coordinates for this batch
            num_sources: Number of source points
            source_start_idx: Starting index of sources in full coordinate list
            total_points: Total number of points in full matrix
            all_coords: All coordinates (for destinations). If None, use source_coords.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if all_coords is None:
            all_coords = source_coords
        
        # Build coordinate string for OSRM Table Service
        # Format: lon1,lat1;lon2,lat2;...
        # For batch processing: sources are batch_coords, destinations are all_coords
        # Strategy: If source_coords are subset of all_coords, use all_coords and specify source indices
        # Otherwise, combine them (sources first, then destinations)
        
        # Check if source_coords are a subset of all_coords (by coordinate matching)
        source_is_subset = False
        if all_coords != source_coords and len(source_coords) <= len(all_coords):
            # Check if all source coordinates exist in all_coords (with tolerance)
            source_matches = 0
            for src_coord in source_coords:
                for all_coord in all_coords:
                    if abs(src_coord[0] - all_coord[0]) < 1e-6 and abs(src_coord[1] - all_coord[1]) < 1e-6:
                        source_matches += 1
                        break
            source_is_subset = (source_matches == len(source_coords))
        
        if all_coords != source_coords and source_is_subset:
            # Sources are subset - use all_coords and specify source indices
            all_coord_str = ";".join([f"{coord[0]:.6f},{coord[1]:.6f}" for coord in all_coords])
            # Find source indices in all_coords
            source_indices = []
            for src_coord in source_coords:
                for idx, all_coord in enumerate(all_coords):
                    if abs(src_coord[0] - all_coord[0]) < 1e-6 and abs(src_coord[1] - all_coord[1]) < 1e-6:
                        source_indices.append(idx)
                        break
            sources_indices = ";".join([str(i) for i in source_indices])
            destinations_indices = "all"
        elif all_coords != source_coords:
            # Different sources and destinations - combine them
            all_coords_list = list(source_coords) + list(all_coords)
            all_coord_str = ";".join([f"{coord[0]:.6f},{coord[1]:.6f}" for coord in all_coords_list])
            # Sources are first num_sources indices, destinations are all remaining
            sources_indices = ";".join([str(i) for i in range(num_sources)])
            destinations_indices = ";".join([str(i) for i in range(num_sources, len(all_coords_list))])
        else:
            # Same sources and destinations
            all_coord_str = ";".join([f"{coord[0]:.6f},{coord[1]:.6f}" for coord in source_coords])
            sources_indices = ";".join([str(i) for i in range(num_sources)])
            destinations_indices = "all"
        
        url = f"{self.osrm_base_url}/table/v1/driving/{all_coord_str}"
        params = {
            "sources": sources_indices,
            "destinations": destinations_indices,
            "annotations": "distance,duration"  # Request both distance and duration
        }
        
        # Exponential backoff retry logic (max 3 retries: 1s, 2s, 4s)
        max_retries = 3
        retry_delays = [1.0, 2.0, 4.0]  # Exponential backoff delays in seconds
        
        for attempt in range(max_retries + 1):  # 0 to max_retries (4 attempts total)
            try:
                # Use session for connection pooling
                if self._session is None:
                    self._session = requests.Session()
                self._session.headers.update({
                    'User-Agent': 'VRP-GA-Optimizer/1.0',
                    'Accept': 'application/json'
                })
                
                with pipeline_profiler.profile(
                    "distance.osrm_request",
                    metadata={'num_sources': num_sources, 'total_points': total_points, 'attempt': attempt + 1}
                ):
                    response = self._session.get(url, params=params, timeout=max(self.osrm_timeout * 2, 30))
                    response.raise_for_status()
                    data = response.json()
                
                # Success - process the response
                if data.get('code') == 'Ok' and 'durations' in data:
                    durations = data['durations']  # Duration matrix in seconds
                    
                    # Convert durations to distances (approximate: distance = duration * average_speed)
                    # For more accuracy, we could use distances if available, but durations are more reliable
                    # Average city speed ~30 km/h = 8.33 m/s
                    # So distance (km) ≈ duration (s) * 8.33 / 1000
                    # But OSRM Table Service returns durations, not distances directly
                    # We need to use a conversion factor or request distances separately
                    
                    # Check if distances are available (preferred)
                    if 'distances' in data:
                        distances = data['distances']  # Distance matrix in meters
                        for i in range(num_sources):
                            matrix_i = source_start_idx + i
                            
                            # Map destination index correctly based on request type
                            if all_coords != source_coords and source_is_subset:
                                # Sources are subset - destinations are all_coords directly
                                # Response: distances[i][j] where j is index in all_coords
                                for j in range(len(all_coords)):
                                    if i < len(distances) and j < len(distances[i]):
                                        dist_meters = distances[i][j]
                                        dist_km = dist_meters / 1000.0
                                        self.distance_matrix[matrix_i, j] = dist_km
                                        # Make symmetric if j is also in source batch
                                        if j >= source_start_idx and j < source_start_idx + num_sources:
                                            self.distance_matrix[j, matrix_i] = dist_km
                            elif all_coords != source_coords:
                                # Destinations start after sources in the response
                                for j_idx, j in enumerate(range(len(all_coords))):
                                    dest_idx = num_sources + j_idx
                                    if i < len(distances) and dest_idx < len(distances[i]):
                                        dist_meters = distances[i][dest_idx]
                                        dist_km = dist_meters / 1000.0
                                        self.distance_matrix[matrix_i, j] = dist_km
                                        # Make symmetric if j is also in source batch
                                        if j >= source_start_idx and j < source_start_idx + num_sources:
                                            self.distance_matrix[j, matrix_i] = dist_km
                            else:
                                # Same sources and destinations - fully symmetric
                                for j in range(len(all_coords)):
                                    if i < len(distances) and j < len(distances[i]):
                                        dist_meters = distances[i][j]
                                        dist_km = dist_meters / 1000.0
                                        self.distance_matrix[matrix_i, j] = dist_km
                                        # Make symmetric
                                        self.distance_matrix[j, matrix_i] = dist_km
                    else:
                        # Fallback: estimate distance from duration
                        # Average speed in city: ~30 km/h = 8.33 m/s
                        AVG_SPEED_MS = 8.33
                        for i in range(num_sources):
                            matrix_i = source_start_idx + i
                            
                            # Map destination index correctly (same logic as distances)
                            if all_coords != source_coords and source_is_subset:
                                # Sources are subset - destinations are all_coords directly
                                for j in range(len(all_coords)):
                                    if i < len(durations) and j < len(durations[i]):
                                        duration_sec = durations[i][j]
                                        dist_meters = duration_sec * AVG_SPEED_MS
                                        dist_km = dist_meters / 1000.0
                                        self.distance_matrix[matrix_i, j] = dist_km
                                        # Make symmetric if j is also in source batch
                                        if j >= source_start_idx and j < source_start_idx + num_sources:
                                            self.distance_matrix[j, matrix_i] = dist_km
                            elif all_coords != source_coords:
                                # Destinations start after sources in the response
                                for j_idx, j in enumerate(range(len(all_coords))):
                                    dest_idx = num_sources + j_idx
                                    if i < len(durations) and dest_idx < len(durations[i]):
                                        duration_sec = durations[i][dest_idx]
                                        dist_meters = duration_sec * AVG_SPEED_MS
                                        dist_km = dist_meters / 1000.0
                                        self.distance_matrix[matrix_i, j] = dist_km
                                        # Make symmetric if j is also in source batch
                                        if j >= source_start_idx and j < source_start_idx + num_sources:
                                            self.distance_matrix[j, matrix_i] = dist_km
                            else:
                                # Same sources and destinations - fully symmetric
                                for j in range(len(all_coords)):
                                    if i < len(durations) and j < len(durations[i]):
                                        duration_sec = durations[i][j]
                                        dist_meters = duration_sec * AVG_SPEED_MS
                                        dist_km = dist_meters / 1000.0
                                        self.distance_matrix[matrix_i, j] = dist_km
                                        # Make symmetric
                                        self.distance_matrix[j, matrix_i] = dist_km
                    
                    # Successfully processed - break out of retry loop
                    return
                else:
                    # Response code not 'Ok' - retry
                    if attempt < max_retries:
                        delay = retry_delays[attempt]
                        logger.warning(f"OSRM response error (attempt {attempt + 1}/{max_retries + 1}): {data.get('code')}. Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"OSRM Table Service returned error after {max_retries + 1} attempts: {data.get('code')}")
                        break
                
            except (requests.exceptions.RequestException, requests.exceptions.Timeout, 
                    requests.exceptions.HTTPError, KeyError, ValueError) as e:
                # Retryable errors - log and retry with exponential backoff
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    logger.warning(f"OSRM request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    # All retries exhausted - fall through to fallback
                    logger.error(f"OSRM Table Service failed after {max_retries + 1} attempts: {e}")
                    break
            except Exception as e:
                # Unexpected error - don't retry, fall through to fallback
                logger.error(f"Unexpected error in OSRM request: {e}")
                break
        
        # All retries exhausted or unexpected error - use haversine fallback
        logger.warning(f"OSRM Table Service unavailable, using haversine fallback")
        # Fallback to haversine
        with pipeline_profiler.profile(
            "distance.osrm_fallback",
            metadata={'reason': 'all_retries_exhausted', 'num_sources': num_sources}
        ):
            for i in range(num_sources):
                for j in range(len(all_coords)):
                    dist = self._haversine_distance(source_coords[i], all_coords[j])
                    matrix_i = source_start_idx + i
                    self.distance_matrix[matrix_i, j] = dist
                    if j < num_sources and j >= source_start_idx:
                        self.distance_matrix[j, matrix_i] = dist
    
    def _calculate_osrm_matrix_parallel(self, coordinates: List[Tuple[float, float]], n_points: int):
        """
        Calculate OSRM distance matrix using parallel processing.
        
        Args:
            coordinates: List of coordinate tuples
            n_points: Number of points
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import logging
        logger = logging.getLogger(__name__)
        
        total_pairs = n_points * (n_points - 1) // 2
        
        # Prepare all pairs to calculate
        pairs_to_calculate = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                pairs_to_calculate.append((i, j, coordinates[i], coordinates[j]))
        
        # Use ThreadPoolExecutor for parallel requests
        # Limit to 10 concurrent requests to avoid overwhelming OSRM server
        max_workers = min(10, total_pairs)
        
        # Try to use Streamlit progress if available
        use_streamlit = False
        progress_bar = None
        try:
            import streamlit as st
            if hasattr(st, 'status') and hasattr(st, 'progress'):
                use_streamlit = True
        except:
            pass
        
        if use_streamlit:
            with st.status(f"Calculating {total_pairs} route distances using OSRM (parallel processing, {max_workers} workers)...", expanded=False):
                progress_bar = st.progress(0)
                completed = 0
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_pair = {
                        executor.submit(self._get_osrm_distance, pair[2], pair[3]): pair
                        for pair in pairs_to_calculate
                    }
                    
                    # Process results as they complete
                    for future in as_completed(future_to_pair):
                        i, j = future_to_pair[future][0], future_to_pair[future][1]
                        try:
                            dist = future.result()
                            self.distance_matrix[i, j] = dist
                            self.distance_matrix[j, i] = dist  # Symmetric
                            completed += 1
                            if progress_bar and total_pairs > 0:
                                progress_bar.progress(completed / total_pairs)
                        except Exception as e:
                            logger.warning(f"Error calculating distance for pair ({i}, {j}): {e}")
                            # Fallback to haversine
                            fallback_dist = self._haversine_distance(
                                future_to_pair[future][2], 
                                future_to_pair[future][3]
                            )
                            self.distance_matrix[i, j] = fallback_dist
                            self.distance_matrix[j, i] = fallback_dist
                            completed += 1
                            if progress_bar and total_pairs > 0:
                                progress_bar.progress(completed / total_pairs)
        else:
            # Fallback without Streamlit
            print(f"Calculating {total_pairs} route distances using OSRM (parallel processing, {max_workers} workers)...")
            completed = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_pair = {
                    executor.submit(self._get_osrm_distance, pair[2], pair[3]): pair
                    for pair in pairs_to_calculate
                }
                
                # Process results as they complete
                for future in as_completed(future_to_pair):
                    i, j = future_to_pair[future][0], future_to_pair[future][1]
                    try:
                        dist = future.result()
                        self.distance_matrix[i, j] = dist
                        self.distance_matrix[j, i] = dist  # Symmetric
                        completed += 1
                        if completed % 100 == 0:
                            print(f"Progress: {completed}/{total_pairs} pairs calculated ({completed*100/total_pairs:.1f}%)...")
                    except Exception as e:
                        logger.warning(f"Error calculating distance for pair ({i}, {j}): {e}")
                        # Fallback to haversine
                        fallback_dist = self._haversine_distance(
                            future_to_pair[future][2], 
                            future_to_pair[future][3]
                        )
                        self.distance_matrix[i, j] = fallback_dist
                        self.distance_matrix[j, i] = fallback_dist
                        completed += 1
                        if completed % 100 == 0:
                            print(f"Progress: {completed}/{total_pairs} pairs calculated ({completed*100/total_pairs:.1f}%)...")
        
        logger.info(f"✅ Completed OSRM calculation for {n_points} locations, {total_pairs} pairs")
    
    def _get_osrm_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Get real road route distance using OSRM with per-segment caching.
        
        Args:
            point1: First point (x=lon, y=lat)
            point2: Second point (x=lon, y=lat)
            
        Returns:
            Real road route distance in km, or haversine distance as fallback
        """
        # Round coordinates for cache key (6 decimal places ~ 0.1m precision)
        lon1, lat1 = round(point1[0], 6), round(point1[1], 6)
        lon2, lat2 = round(point2[0], 6), round(point2[1], 6)
        
        # Create cache key (symmetric: same distance for A->B and B->A)
        cache_key = tuple(sorted([(lon1, lat1), (lon2, lat2)]))
        
        # Check per-segment cache first
        if cache_key in self._osrm_segment_cache:
            return self._osrm_segment_cache[cache_key]
        
        try:
            coordinate_str = f"{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}"
            url = f"{self.osrm_base_url}/route/v1/driving/{coordinate_str}"
            params = {
                "overview": "false",  # We only need distance, not geometry
                "alternatives": "false"
            }
            
            # Use session for connection pooling (faster for multiple requests)
            if self._session is None:
                self._session = requests.Session()
                # Set default headers for better performance
                self._session.headers.update({
                    'User-Agent': 'VRP-GA-Optimizer/1.0',
                    'Accept': 'application/json'
                })
            
            response = self._session.get(url, params=params, timeout=self.osrm_timeout)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') == 'Ok' and data.get('routes'):
                route = data['routes'][0]
                distance_meters = route.get('distance', 0)
                distance_km = distance_meters / 1000.0
                
                # Cache the result
                self._osrm_segment_cache[cache_key] = distance_km
                return distance_km
            else:
                # Fallback to haversine
                fallback_dist = self._haversine_distance(point1, point2)
                self._osrm_segment_cache[cache_key] = fallback_dist
                return fallback_dist
                
        except requests.exceptions.Timeout:
            # Timeout - use haversine
            fallback_dist = self._haversine_distance(point1, point2)
            self._osrm_segment_cache[cache_key] = fallback_dist
            return fallback_dist
        except Exception as e:
            # Fallback to haversine if OSRM fails
            fallback_dist = self._haversine_distance(point1, point2)
            self._osrm_segment_cache[cache_key] = fallback_dist
            return fallback_dist
    
    def clear_cache(self):
        """Clear all cached distance matrices."""
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.startswith('dist_matrix_') and file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, file))


def calculate_distance_matrix(coordinates: List[Tuple[float, float]], 
                           traffic_factor: float = 1.0,
                           use_adaptive: bool = False) -> np.ndarray:
    """
    Convenience function to calculate distance matrix.
    
    Args:
        coordinates: List of (x, y) coordinate tuples
        traffic_factor: Base multiplier for distance calculation
        use_adaptive: Enable adaptive traffic factor
        
    Returns:
        Distance matrix as numpy array (base distance, no traffic factor applied)
    """
    calculator = DistanceCalculator(traffic_factor, use_adaptive)
    return calculator.calculate_distance_matrix(coordinates)
