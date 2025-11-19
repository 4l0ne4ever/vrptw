"""
VRP problem model and data structures.
Defines the core VRP problem representation.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Customer:
    """Represents a customer in the VRP problem."""
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float
    
    def __post_init__(self):
        """Validate customer data after initialization."""
        if self.demand < 0:
            raise ValueError(f"Customer {self.id} has negative demand")
        if self.ready_time > self.due_date:
            raise ValueError(f"Customer {self.id} has invalid time window")


@dataclass
class Depot:
    """Represents the depot in the VRP problem."""
    id: int
    x: float
    y: float
    demand: float = 0.0
    ready_time: float = 0.0
    due_date: float = 1000.0
    service_time: float = 0.0


class VRPProblem:
    """Represents a complete VRP problem instance."""
    
    def __init__(self, 
                 customers: List[Customer],
                 depot: Depot,
                 vehicle_capacity: float,
                 num_vehicles: int,
                 distance_matrix: Optional[np.ndarray] = None,
                 use_adaptive_traffic: bool = False):
        """
        Initialize VRP problem.
        
        Args:
            customers: List of customers
            depot: Depot information
            vehicle_capacity: Maximum capacity per vehicle
            num_vehicles: Maximum number of vehicles
            distance_matrix: Optional pre-computed distance matrix (base distance)
            use_adaptive_traffic: Enable adaptive traffic factor based on time
        """
        self.customers = customers
        self.depot = depot
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.distance_matrix = distance_matrix
        self.use_adaptive_traffic = use_adaptive_traffic
        self.distance_calculator = None  # Reference to DistanceCalculator for adaptive traffic
        
        # Create ID to index mapping for distance matrix
        self.id_to_index = {0: 0}  # Depot is always index 0
        for i, customer in enumerate(customers):
            self.id_to_index[customer.id] = i + 1  # Customers start from index 1
        
        # Validate problem
        self._validate_problem()
    
    def _validate_problem(self):
        """Validate VRP problem constraints."""
        if not self.customers:
            raise ValueError("No customers provided")
        
        if self.vehicle_capacity <= 0:
            raise ValueError("Vehicle capacity must be positive")
        
        if self.num_vehicles <= 0:
            raise ValueError("Number of vehicles must be positive")
        
        # Check if any customer demand exceeds vehicle capacity
        max_demand = max(c.demand for c in self.customers)
        if max_demand > self.vehicle_capacity:
            raise ValueError(f"Customer demand {max_demand} exceeds vehicle capacity {self.vehicle_capacity}")
        
        # Check if total demand can be served
        total_demand = sum(c.demand for c in self.customers)
        max_total_capacity = self.num_vehicles * self.vehicle_capacity
        if total_demand > max_total_capacity:
            raise ValueError(f"Total demand {total_demand} exceeds total capacity {max_total_capacity}")
    
    def get_customer_by_id(self, customer_id: int) -> Optional[Customer]:
        """Get customer by ID."""
        for customer in self.customers:
            if customer.id == customer_id:
                return customer
        return None
    
    def get_customer_coordinates(self) -> List[Tuple[float, float]]:
        """Get list of customer coordinates."""
        return [(c.x, c.y) for c in self.customers]
    
    def get_all_coordinates(self) -> List[Tuple[float, float]]:
        """Get list of all coordinates (depot + customers)."""
        coords = [(self.depot.x, self.depot.y)]  # Depot first
        coords.extend(self.get_customer_coordinates())
        return coords
    
    def get_demands(self) -> List[float]:
        """Get list of customer demands."""
        return [c.demand for c in self.customers]
    
    def get_time_windows(self) -> List[Tuple[float, float]]:
        """Get list of time windows."""
        return [(c.ready_time, c.due_date) for c in self.customers]
    
    def get_service_times(self) -> List[float]:
        """Get list of service times."""
        return [c.service_time for c in self.customers]
    
    def set_distance_calculator(self, calculator):
        """Set distance calculator for adaptive traffic factor."""
        self.distance_calculator = calculator
    
    def get_distance(self, from_idx: int, to_idx: int) -> float:
        """Get base distance between two points by ID (no traffic factor)."""
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not available")
        
        # Convert IDs to matrix indices
        from_matrix_idx = self.id_to_index.get(from_idx)
        to_matrix_idx = self.id_to_index.get(to_idx)
        
        if from_matrix_idx is None or to_matrix_idx is None:
            raise ValueError(f"Invalid customer ID: {from_idx} or {to_idx}")
        
        return self.distance_matrix[from_matrix_idx, to_matrix_idx]
    
    def get_adaptive_distance(self, from_id: int, to_id: int, 
                              time_minutes: float) -> float:
        """
        Get distance with adaptive traffic factor applied.
        
        Args:
            from_id: Source customer ID
            to_id: Destination customer ID
            time_minutes: Time in minutes from 0:00 when traveling this segment
        
        Returns:
            Distance with adaptive traffic factor (or base distance if adaptive disabled)
        """
        if not self.use_adaptive_traffic or self.distance_calculator is None:
            return self.get_distance(from_id, to_id)
        
        # Convert IDs to matrix indices
        from_matrix_idx = self.id_to_index.get(from_id)
        to_matrix_idx = self.id_to_index.get(to_id)
        
        if from_matrix_idx is None or to_matrix_idx is None:
            raise ValueError(f"Invalid customer ID: {from_id} or {to_id}")
        
        return self.distance_calculator.get_adaptive_distance(
            from_matrix_idx, to_matrix_idx, time_minutes
        )
    
    def calculate_total_demand(self) -> float:
        """Calculate total demand of all customers."""
        return sum(c.demand for c in self.customers)
    
    def estimate_minimum_vehicles(self) -> int:
        """Estimate minimum number of vehicles needed."""
        total_demand = self.calculate_total_demand()
        return max(1, int(np.ceil(total_demand / self.vehicle_capacity)))
    
    def is_feasible(self) -> bool:
        """Check if the problem is feasible."""
        try:
            self._validate_problem()
            return True
        except ValueError:
            return False
    
    def get_problem_info(self) -> Dict:
        """Get problem information summary."""
        return {
            'num_customers': int(len(self.customers)),
            'vehicle_capacity': float(self.vehicle_capacity),
            'num_vehicles': int(self.num_vehicles),
            'total_demand': float(self.calculate_total_demand()),
            'min_vehicles_needed': int(self.estimate_minimum_vehicles()),
            'is_feasible': bool(self.is_feasible()),
            'depot_location': (float(self.depot.x), float(self.depot.y)),
            'customer_bounds': self._get_customer_bounds()
        }
    
    def _get_customer_bounds(self) -> Dict:
        """Get bounding box of customer locations."""
        if not self.customers:
            return {'x_min': 0.0, 'x_max': 0.0, 'y_min': 0.0, 'y_max': 0.0}
        
        x_coords = [c.x for c in self.customers]
        y_coords = [c.y for c in self.customers]
        
        return {
            'x_min': float(min(x_coords)),
            'x_max': float(max(x_coords)),
            'y_min': float(min(y_coords)),
            'y_max': float(max(y_coords))
        }


def create_vrp_problem_from_dict(data: Dict, distance_matrix: Optional[np.ndarray] = None,
                                 use_adaptive_traffic: bool = False) -> VRPProblem:
    """
    Create VRP problem from dictionary data.
    
    Args:
        data: Dictionary containing VRP data
        distance_matrix: Optional pre-computed distance matrix (base distance)
        use_adaptive_traffic: Enable adaptive traffic factor
        
    Returns:
        VRPProblem instance
    """
    # Create customers
    customers = []
    for customer_data in data['customers']:
        customer = Customer(
            id=customer_data['id'],
            x=customer_data['x'],
            y=customer_data['y'],
            demand=customer_data['demand'],
            ready_time=customer_data['ready_time'],
            due_date=customer_data['due_date'],
            service_time=customer_data['service_time']
        )
        customers.append(customer)
    
    # Create depot
    depot_data = data['depot']
    depot = Depot(
        id=depot_data['id'],
        x=depot_data['x'],
        y=depot_data['y'],
        demand=depot_data['demand'],
        ready_time=depot_data['ready_time'],
        due_date=depot_data['due_date'],
        service_time=depot_data['service_time']
    )
    
    # Create VRP problem
    problem = VRPProblem(
        customers=customers,
        depot=depot,
        vehicle_capacity=data['vehicle_capacity'],
        num_vehicles=data['num_vehicles'],
        distance_matrix=distance_matrix,
        use_adaptive_traffic=use_adaptive_traffic
    )
    
    return problem
