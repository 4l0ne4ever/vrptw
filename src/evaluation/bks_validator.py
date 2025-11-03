"""
Best-Known Solutions (BKS) validator for VRP problems.
Compares solution quality against known best solutions from literature.
"""

import json
import os
from typing import Dict, Optional, List
from src.models.solution import Individual


class BKSValidator:
    """Validate solutions against Best-Known Solutions (BKS)."""
    
    def __init__(self, bks_file: str = "data/solomon_bks.json"):
        """
        Initialize BKS validator.
        
        Args:
            bks_file: Path to BKS data JSON file
        """
        self.bks_file = bks_file
        self.bks_data = {}
        self._load_bks_data()
    
    def _load_bks_data(self):
        """Load BKS data from JSON file."""
        if not os.path.exists(self.bks_file):
            # Create default empty BKS data if file doesn't exist
            self.bks_data = {}
            return
        
        try:
            with open(self.bks_file, 'r', encoding='utf-8') as f:
                self.bks_data = json.load(f)
        except Exception as e:
            # If file exists but can't be loaded, use empty dict
            self.bks_data = {}
    
    def get_bks(self, instance_name: str) -> Dict[str, float]:
        """
        Get BKS for a specific instance.
        
        Args:
            instance_name: Name of the Solomon instance (e.g., "C101")
            
        Returns:
            Dictionary with BKS data (distance, vehicles, etc.) or empty dict
        """
        # Normalize instance name (remove .json extension if present)
        instance_name = instance_name.replace('.json', '').upper()
        return self.bks_data.get(instance_name, {})
    
    def has_bks(self, instance_name: str) -> bool:
        """
        Check if BKS exists for an instance.
        
        Args:
            instance_name: Name of the Solomon instance
            
        Returns:
            True if BKS exists, False otherwise
        """
        instance_name = instance_name.replace('.json', '').upper()
        return instance_name in self.bks_data
    
    def calculate_gap(self, instance_name: str, solution_distance: float) -> Optional[float]:
        """
        Calculate gap (percentage deviation) from BKS.
        
        Args:
            instance_name: Name of the Solomon instance
            solution_distance: Solution total distance
            
        Returns:
            Gap percentage (positive = worse than BKS) or None if BKS not available
        """
        bks = self.get_bks(instance_name)
        
        if 'distance' not in bks or bks['distance'] is None:
            return None
        
        bks_distance = bks['distance']
        
        if bks_distance <= 0:
            return None
        
        gap = ((solution_distance - bks_distance) / bks_distance) * 100
        return gap
    
    def calculate_vehicle_gap(self, instance_name: str, solution_vehicles: int) -> Optional[int]:
        """
        Calculate vehicle count difference from BKS.
        
        Args:
            instance_name: Name of the Solomon instance
            solution_vehicles: Solution number of vehicles
            
        Returns:
            Vehicle difference (positive = more vehicles) or None if BKS not available
        """
        bks = self.get_bks(instance_name)
        
        if 'vehicles' not in bks or bks['vehicles'] is None:
            return None
        
        bks_vehicles = bks['vehicles']
        return solution_vehicles - bks_vehicles
    
    def validate_solution(self, instance_name: str, solution: Individual) -> Dict:
        """
        Validate solution against BKS.
        
        Args:
            instance_name: Name of the Solomon instance
            solution: Solution individual
            
        Returns:
            Dictionary with validation results:
            - instance: Instance name
            - solution_distance: Solution total distance
            - solution_vehicles: Solution number of vehicles
            - bks_distance: BKS distance
            - bks_vehicles: BKS number of vehicles
            - gap_percent: Percentage gap from BKS
            - vehicle_diff: Vehicle count difference
            - quality: Quality rating (EXCELLENT/GOOD/ACCEPTABLE/POOR/UNKNOWN)
            - has_bks: Whether BKS exists for this instance
        """
        instance_name = instance_name.replace('.json', '').upper()
        bks = self.get_bks(instance_name)
        
        solution_distance = solution.total_distance if hasattr(solution, 'total_distance') else 0.0
        solution_vehicles = solution.get_route_count() if hasattr(solution, 'get_route_count') else 0
        
        gap = self.calculate_gap(instance_name, solution_distance)
        vehicle_diff = self.calculate_vehicle_gap(instance_name, solution_vehicles)
        
        quality = self._get_quality_rating(gap)
        
        return {
            'instance': instance_name,
            'solution_distance': solution_distance,
            'solution_vehicles': solution_vehicles,
            'bks_distance': bks.get('distance'),
            'bks_vehicles': bks.get('vehicles'),
            'gap_percent': gap,
            'vehicle_diff': vehicle_diff,
            'quality': quality,
            'has_bks': instance_name in self.bks_data
        }
    
    def _get_quality_rating(self, gap: Optional[float]) -> str:
        """
        Get quality rating based on gap percentage.
        
        Args:
            gap: Gap percentage from BKS
            
        Returns:
            Quality rating string
        """
        if gap is None:
            return 'UNKNOWN'
        elif gap < 1.0:
            return 'EXCELLENT'
        elif gap < 3.0:
            return 'GOOD'
        elif gap < 5.0:
            return 'ACCEPTABLE'
        elif gap < 10.0:
            return 'POOR'
        else:
            return 'VERY_POOR'
    
    def get_all_bks_instances(self) -> List[str]:
        """
        Get list of all instances with BKS data.
        
        Returns:
            List of instance names
        """
        return list(self.bks_data.keys())
    
    def add_bks(self, instance_name: str, distance: float, vehicles: int = None, 
                source: str = None):
        """
        Add BKS entry manually (for testing or new discoveries).
        
        Args:
            instance_name: Instance name
            distance: BKS distance
            vehicles: BKS number of vehicles (optional)
            source: Source of BKS data (optional)
        """
        instance_name = instance_name.replace('.json', '').upper()
        self.bks_data[instance_name] = {
            'distance': distance,
            'vehicles': vehicles,
            'source': source
        }
    
    def save_bks_data(self, output_file: str = None):
        """
        Save BKS data to JSON file.
        
        Args:
            output_file: Output file path (default: original bks_file)
        """
        output_file = output_file or self.bks_file
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.bks_data, f, indent=2, ensure_ascii=False)

