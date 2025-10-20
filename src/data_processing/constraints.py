"""
Constraint handling for VRP problems.
Validates capacity, time window, and other constraints.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class ConstraintHandler:
    """Handles VRP constraints validation and repair."""
    
    def __init__(self, vehicle_capacity: float, num_vehicles: int):
        """
        Initialize constraint handler.
        
        Args:
            vehicle_capacity: Maximum capacity per vehicle
            num_vehicles: Maximum number of vehicles available
        """
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.penalty_weight = 1000  # Penalty for constraint violations
    
    def validate_capacity_constraint(self, 
                                  routes: List[List[int]], 
                                  demands: List[float]) -> Tuple[bool, float]:
        """
        Validate capacity constraints for all routes.
        
        Args:
            routes: List of routes, each route is a list of customer indices
            demands: List of customer demands (index 0 = customer ID 1, etc.)
            
        Returns:
            Tuple of (is_valid, total_penalty)
        """
        total_penalty = 0.0
        is_valid = True
        
        for route in routes:
            if not route:  # Empty route
                continue
            
            route_load = 0.0
            for customer_id in route:
                if customer_id != 0:  # Skip depot
                    # Customer ID 1 corresponds to demands[0], ID 2 to demands[1], etc.
                    if 1 <= customer_id <= len(demands):
                        route_load += demands[customer_id - 1]
            
            if route_load > self.vehicle_capacity:
                is_valid = False
                excess = route_load - self.vehicle_capacity
                total_penalty += self.penalty_weight * excess
        
        return is_valid, total_penalty
    
    def validate_vehicle_count_constraint(self, routes: List[List[int]]) -> Tuple[bool, float]:
        """
        Validate vehicle count constraint.
        
        Args:
            routes: List of routes
            
        Returns:
            Tuple of (is_valid, penalty)
        """
        num_used_vehicles = len([route for route in routes if route])
        
        if num_used_vehicles > self.num_vehicles:
            excess = num_used_vehicles - self.num_vehicles
            penalty = self.penalty_weight * excess
            return False, penalty
        
        return True, 0.0
    
    def validate_customer_visit_constraint(self, 
                                        routes: List[List[int]], 
                                        num_customers: int) -> Tuple[bool, float]:
        """
        Validate that each customer is visited exactly once.
        
        Args:
            routes: List of routes
            num_customers: Total number of customers
            
        Returns:
            Tuple of (is_valid, penalty)
        """
        visited_customers = set()
        
        for route in routes:
            for customer_id in route:
                if customer_id in visited_customers:
                    # Customer visited multiple times
                    return False, self.penalty_weight * 100
                visited_customers.add(customer_id)
        
        # Check if all customers are visited
        expected_customers = set(range(1, num_customers + 1))  # Customers start from 1
        missing_customers = expected_customers - visited_customers
        
        if missing_customers:
            penalty = self.penalty_weight * len(missing_customers)
            return False, penalty
        
        return True, 0.0
    
    def validate_depot_constraint(self, routes: List[List[int]]) -> Tuple[bool, float]:
        """
        Validate that routes start and end at depot (customer 0).
        
        Args:
            routes: List of routes
            
        Returns:
            Tuple of (is_valid, penalty)
        """
        penalty = 0.0
        
        for route in routes:
            if not route:  # Empty route
                continue
            
            # Check if route starts and ends at depot
            if route[0] != 0:
                penalty += self.penalty_weight
            if route[-1] != 0:
                penalty += self.penalty_weight
        
        is_valid = penalty == 0.0
        return is_valid, penalty
    
    def validate_time_window_constraint(self, 
                                      routes: List[List[int]], 
                                      time_windows: List[Tuple[float, float]],
                                      service_times: List[float],
                                      distance_matrix: np.ndarray) -> Tuple[bool, float]:
        """
        Validate time window constraints.
        
        Args:
            routes: List of routes
            time_windows: List of (ready_time, due_date) for each customer
            service_times: List of service times for each customer
            distance_matrix: Distance matrix between all points
            
        Returns:
            Tuple of (is_valid, penalty)
        """
        total_penalty = 0.0
        
        for route in routes:
            if len(route) < 2:  # Skip empty or single-point routes
                continue
            
            current_time = 0.0
            
            for i in range(len(route) - 1):
                from_customer = route[i]
                to_customer = route[i + 1]
                
                # Travel time (assuming unit speed)
                travel_time = distance_matrix[from_customer, to_customer]
                current_time += travel_time
                
                # Check if we arrive too early
                ready_time = time_windows[to_customer][0]
                if current_time < ready_time:
                    current_time = ready_time
                
                # Check if we arrive too late
                due_date = time_windows[to_customer][1]
                if current_time > due_date:
                    # Late arrival penalty
                    lateness = current_time - due_date
                    total_penalty += self.penalty_weight * lateness
                
                # Add service time
                current_time += service_times[to_customer]
        
        is_valid = total_penalty == 0.0
        return is_valid, total_penalty
    
    def validate_all_constraints(self, 
                               routes: List[List[int]], 
                               demands: List[float],
                               num_customers: int,
                               time_windows: Optional[List[Tuple[float, float]]] = None,
                               service_times: Optional[List[float]] = None,
                               distance_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Validate all VRP constraints.
        
        Args:
            routes: List of routes
            demands: List of customer demands
            num_customers: Total number of customers
            time_windows: Optional time windows for customers
            service_times: Optional service times for customers
            distance_matrix: Optional distance matrix
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'total_penalty': 0.0,
            'violations': {}
        }
        
        # Capacity constraint
        cap_valid, cap_penalty = self.validate_capacity_constraint(routes, demands)
        results['violations']['capacity'] = not cap_valid
        results['total_penalty'] += cap_penalty
        
        # Vehicle count constraint
        veh_valid, veh_penalty = self.validate_vehicle_count_constraint(routes)
        results['violations']['vehicle_count'] = not veh_valid
        results['total_penalty'] += veh_penalty
        
        # Customer visit constraint
        visit_valid, visit_penalty = self.validate_customer_visit_constraint(routes, num_customers)
        results['violations']['customer_visit'] = not visit_valid
        results['total_penalty'] += visit_penalty
        
        # Depot constraint
        depot_valid, depot_penalty = self.validate_depot_constraint(routes)
        results['violations']['depot'] = not depot_valid
        results['total_penalty'] += depot_penalty
        
        # Time window constraint (if data provided)
        if (time_windows is not None and service_times is not None and 
            distance_matrix is not None):
            tw_valid, tw_penalty = self.validate_time_window_constraint(
                routes, time_windows, service_times, distance_matrix
            )
            results['violations']['time_windows'] = not tw_valid
            results['total_penalty'] += tw_penalty
        
        # Overall validity
        results['is_valid'] = all(results['violations'].values())
        
        return results
    
    def repair_capacity_violations(self, 
                                 routes: List[List[int]], 
                                 demands: List[float]) -> List[List[int]]:
        """
        Repair capacity violations by moving customers to other routes.
        
        Args:
            routes: List of routes with potential capacity violations
            demands: List of customer demands
            
        Returns:
            Repaired routes
        """
        repaired_routes = [route.copy() for route in routes]
        
        for i, route in enumerate(repaired_routes):
            if not route:
                continue
            
            route_load = sum(demands[customer_id] for customer_id in route)
            
            while route_load > self.vehicle_capacity:
                # Find customer with highest demand in this route
                max_demand = 0
                max_customer_idx = -1
                
                for j, customer_id in enumerate(route):
                    if demands[customer_id] > max_demand:
                        max_demand = demands[customer_id]
                        max_customer_idx = j
                
                if max_customer_idx == -1:
                    break
                
                # Remove customer from current route
                removed_customer = route.pop(max_customer_idx)
                route_load -= demands[removed_customer]
                
                # Try to add to another route
                added = False
                for j, other_route in enumerate(repaired_routes):
                    if j == i:  # Skip current route
                        continue
                    
                    other_load = sum(demands[cid] for cid in other_route)
                    if other_load + demands[removed_customer] <= self.vehicle_capacity:
                        other_route.append(removed_customer)
                        added = True
                        break
                
                # If couldn't add to existing route, create new route
                if not added:
                    repaired_routes.append([removed_customer])
        
        return repaired_routes
    
    def calculate_route_load(self, route: List[int], demands: List[float]) -> float:
        """Calculate total load for a route."""
        return sum(demands[customer_id] for customer_id in route)
    
    def calculate_route_utilization(self, route: List[int], demands: List[float]) -> float:
        """Calculate utilization percentage for a route."""
        load = self.calculate_route_load(route, demands)
        return (load / self.vehicle_capacity) * 100 if self.vehicle_capacity > 0 else 0.0
