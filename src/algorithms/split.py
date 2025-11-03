"""
Optimal Split Algorithm for VRP giant tour representation.
Implements Prins (2004) split algorithm for optimal route splitting.
"""

from typing import List, Tuple, Optional, Dict
from src.models.vrp_model import VRPProblem


class SplitAlgorithm:
    """
    Optimal split algorithm for giant tour (Prins 2004).
    
    Uses dynamic programming to find optimal way to split a giant tour
    into routes while respecting capacity constraints.
    """
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize split algorithm.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
    
    def split(self, giant_tour: List[int]) -> Tuple[List[List[int]], float]:
        """
        Split giant tour into optimal routes using dynamic programming.
        
        Args:
            giant_tour: Giant tour (list of customer IDs, excluding depot)
            
        Returns:
            Tuple of (routes, total_cost):
            - routes: List of routes (each route includes depot at start and end)
            - total_cost: Total distance cost of split
            
        Example:
            >>> splitter = SplitAlgorithm(problem)
            >>> routes, cost = splitter.split([1, 2, 3, 4, 5])
            >>> print(f"Total cost: {cost}")
        """
        if not giant_tour:
            return [], 0.0
        
        n = len(giant_tour)
        
        # DP arrays
        # V[i] = minimum cost to cover customers 0..i-1
        V = [float('inf')] * (n + 1)
        V[0] = 0.0  # No customers = zero cost
        
        # pred[i] = best predecessor for position i
        pred = [-1] * (n + 1)
        
        # Dynamic programming
        for i in range(n):
            if V[i] == float('inf'):
                continue
            
            # Try building route starting from position i
            load = 0.0
            route_cost = 0.0
            
            # j represents end position of route starting at i
            for j in range(i + 1, n + 1):
                # Get customer at position j-1
                customer_id = giant_tour[j - 1]
                customer = self.problem.get_customer_by_id(customer_id)
                
                if customer is None:
                    break
                
                customer_demand = customer.demand
                
                # Check if adding this customer exceeds capacity
                if load + customer_demand > self.problem.vehicle_capacity:
                    break
                
                # Calculate route cost
                if j == i + 1:
                    # First customer in route: depot -> customer -> depot
                    route_cost = (
                        self.problem.get_distance(0, customer_id) +
                        self.problem.get_distance(customer_id, 0)
                    )
                else:
                    # Add customer to existing route
                    prev_customer_id = giant_tour[j - 2]
                    
                    # Remove: prev_customer -> depot
                    # Add: prev_customer -> customer -> depot
                    route_cost = (
                        route_cost -
                        self.problem.get_distance(prev_customer_id, 0) +
                        self.problem.get_distance(prev_customer_id, customer_id) +
                        self.problem.get_distance(customer_id, 0)
                    )
                
                load += customer_demand
                
                # Update DP if this split is better
                new_cost = V[i] + route_cost
                if new_cost < V[j]:
                    V[j] = new_cost
                    pred[j] = i
        
        # Reconstruct routes from DP solution
        routes = []
        j = n
        
        while j > 0:
            i = pred[j]
            if i < 0:
                # No valid split found - use fallback decoder
                return self._fallback_split(giant_tour)
            
            # Extract route segment
            route_segment = giant_tour[i:j]
            
            # Build route: depot -> customers -> depot
            route = [0] + route_segment + [0]
            routes.insert(0, route)
            
            j = i
        
        total_cost = V[n]
        
        # Validate routes
        if not self._validate_routes(routes):
            # If validation fails, use fallback
            return self._fallback_split(giant_tour)
        
        return routes, total_cost
    
    def _fallback_split(self, giant_tour: List[int]) -> Tuple[List[List[int]], float]:
        """
        Fallback split using simple greedy approach.
        
        Args:
            giant_tour: Giant tour
            
        Returns:
            Tuple of (routes, total_cost)
        """
        routes = []
        current_route = [0]
        current_load = 0.0
        total_cost = 0.0
        
        for customer_id in giant_tour:
            customer = self.problem.get_customer_by_id(customer_id)
            if customer is None:
                continue
            
            customer_demand = customer.demand
            
            # Check capacity
            if current_load + customer_demand <= self.problem.vehicle_capacity:
                # Add to current route
                if current_route == [0]:
                    # First customer in route
                    total_cost += self.problem.get_distance(0, customer_id)
                else:
                    # Add edge cost
                    prev_customer_id = current_route[-1]
                    total_cost += self.problem.get_distance(prev_customer_id, customer_id)
                
                current_route.append(customer_id)
                current_load += customer_demand
            else:
                # Finish current route
                if current_route != [0]:
                    last_customer_id = current_route[-1]
                    total_cost += self.problem.get_distance(last_customer_id, 0)
                    current_route.append(0)
                    routes.append(current_route)
                
                # Start new route
                total_cost += self.problem.get_distance(0, customer_id)
                current_route = [0, customer_id]
                current_load = customer_demand
        
        # Finish last route
        if current_route != [0]:
            last_customer_id = current_route[-1]
            total_cost += self.problem.get_distance(last_customer_id, 0)
            current_route.append(0)
            routes.append(current_route)
        
        return routes, total_cost
    
    def _validate_routes(self, routes: List[List[int]]) -> bool:
        """
        Validate routes for correctness.
        
        Args:
            routes: List of routes to validate
            
        Returns:
            True if routes are valid, False otherwise
        """
        if not routes:
            return False
        
        # Check all customers are visited exactly once
        visited_customers = set()
        
        for route in routes:
            # Check route starts and ends at depot
            if not route or route[0] != 0 or route[-1] != 0:
                return False
            
            # Check capacity
            route_load = 0.0
            for customer_id in route:
                if customer_id == 0:
                    continue
                
                if customer_id in visited_customers:
                    return False
                
                visited_customers.add(customer_id)
                customer = self.problem.get_customer_by_id(customer_id)
                if customer is None:
                    return False
                
                route_load += customer.demand
            
            if route_load > self.problem.vehicle_capacity:
                return False
        
        # Check all customers are visited
        all_customers = {c.id for c in self.problem.customers}
        if visited_customers != all_customers:
            return False
        
        return True
    
    def split_with_time_windows(self, giant_tour: List[int]) -> Tuple[List[List[int]], float, Dict]:
        """
        Split giant tour with time window constraints.
        
        Args:
            giant_tour: Giant tour
            
        Returns:
            Tuple of (routes, total_cost, violations):
            - routes: List of routes
            - total_cost: Total distance cost
            - violations: Dictionary with violation information
        """
        routes, total_cost = self.split(giant_tour)
        
        violations = {
            'time_window_violations': 0,
            'capacity_violations': 0,
            'details': []
        }
        
        for route_idx, route in enumerate(routes):
            # Check time windows
            current_time = 0.0
            
            for i, customer_id in enumerate(route):
                if customer_id == 0:
                    # Depot - reset time
                    current_time = 0.0
                    continue
                
                customer = self.problem.get_customer_by_id(customer_id)
                if customer is None:
                    continue
                
                # Travel time from previous location
                if i > 0:
                    prev_id = route[i - 1]
                    travel_time = self.problem.get_distance(prev_id, customer_id)
                    current_time += travel_time
                
                # Check time window
                if current_time < customer.ready_time:
                    current_time = customer.ready_time  # Wait
                    violations['time_window_violations'] += 1
                
                if current_time > customer.due_date:
                    violations['time_window_violations'] += 1
                    violations['details'].append({
                        'route': route_idx,
                        'customer': customer_id,
                        'arrival_time': current_time,
                        'due_date': customer.due_date
                    })
                
                # Service time
                current_time += customer.service_time
        
        return routes, total_cost, violations
    
    def get_split_statistics(self, routes: List[List[int]]) -> Dict:
        """
        Get statistics for split routes.
        
        Args:
            routes: List of routes
            
        Returns:
            Dictionary with statistics
        """
        if not routes:
            return {
                'num_routes': 0,
                'avg_route_length': 0.0,
                'avg_load': 0.0,
                'avg_utilization': 0.0
            }
        
        route_lengths = []
        route_loads = []
        
        for route in routes:
            customers = [c for c in route if c != 0]
            route_lengths.append(len(customers))
            
            route_load = sum(
                self.problem.get_customer_by_id(c).demand
                for c in customers
            )
            route_loads.append(route_load)
        
        route_utilizations = [
            (load / self.problem.vehicle_capacity) * 100
            for load in route_loads
        ]
        
        return {
            'num_routes': len(routes),
            'avg_route_length': sum(route_lengths) / len(route_lengths) if route_lengths else 0.0,
            'avg_load': sum(route_loads) / len(route_loads) if route_loads else 0.0,
            'avg_utilization': sum(route_utilizations) / len(route_utilizations) if route_utilizations else 0.0,
            'min_utilization': min(route_utilizations) if route_utilizations else 0.0,
            'max_utilization': max(route_utilizations) if route_utilizations else 0.0
        }

