"""
Optimal Split Algorithm for VRP giant tour representation.
Implements Prins (2004) split algorithm for optimal route splitting with TW awareness.
"""

from typing import List, Tuple, Dict
from src.models.vrp_model import VRPProblem
from src.core.pipeline_profiler import pipeline_profiler


class SplitAlgorithm:
    """
    Optimal split algorithm for giant tour (Prins 2004).
    
    Uses dynamic programming to find optimal way to split a giant tour
    into routes while respecting capacity constraints and soft time-window constraints.
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
        Split giant tour into optimal routes using full DP/Bellman algorithm.
        
        Uses optimal dynamic programming (Prins 2004) optimizing for distance only.
        Time window constraints are handled by the fitness function and GA evolution.
        
        This separation of concerns ensures generalization:
        - Split: Fast, optimal distance-based decoding
        - Fitness: Constraint evaluation and penalties  
        - GA: Search for TW-compatible giant tours
        
        Args:
            giant_tour: Giant tour (list of customer IDs, excluding depot)
            
        Returns:
            Tuple of (routes, total_cost):
            - routes: List of routes (each route includes depot at start and end)
            - total_cost: Total distance cost of split
        """
        if not giant_tour:
            return [], 0.0
        
        n = len(giant_tour)
        with pipeline_profiler.profile("split.execute", metadata={'n_customers': n}):
            # Use full DP/Bellman for optimal distance-based solution
            return self._split_full_dp(giant_tour)
    
    def _split_full_dp(self, giant_tour: List[int]) -> Tuple[List[List[int]], float]:
        """
        Full DP/Bellman split algorithm (Prins 2004).
        
        Optimal solution using dynamic programming, optimizing for distance only.
        Time window constraints are handled by the penalty system in fitness evaluation.
        """
        n = len(giant_tour)
        V = [float('inf')] * (n + 1)
        V[0] = 0.0
        pred = [-1] * (n + 1)
        
        # Pre-compute customer data for faster lookup
        customer_data = {}
        for idx, customer_id in enumerate(giant_tour):
            customer = self.problem.get_customer_by_id(customer_id)
            if customer:
                customer_data[idx] = (customer_id, customer.demand)
        
        capacity = self.problem.vehicle_capacity
        
        for i in range(n):
            if V[i] == float('inf'):
                continue
            
            load = 0.0
            route_cost = 0.0
            prev_customer_id = 0  # Start from depot
            
            for j in range(i + 1, n + 1):
                if j - 1 not in customer_data:
                    break
                
                customer_id, customer_demand = customer_data[j - 1]
                
                # Capacity check (ONLY constraint checked during split)
                # Time windows are handled by penalty system
                if load + customer_demand > capacity:
                    break
                
                # Incremental cost calculation
                if j == i + 1:
                    # First customer: depot -> customer -> depot
                    route_cost = (
                        self.problem.get_distance(0, customer_id) +
                        self.problem.get_distance(customer_id, 0)
                    )
                else:
                    # Add customer: remove prev->depot, add prev->customer->depot
                    route_cost = (
                        route_cost -
                        self.problem.get_distance(prev_customer_id, 0) +
                        self.problem.get_distance(prev_customer_id, customer_id) +
                        self.problem.get_distance(customer_id, 0)
                    )
                
                load += customer_demand
                prev_customer_id = customer_id
                
                new_cost = V[i] + route_cost
                if new_cost < V[j]:
                    V[j] = new_cost
                    pred[j] = i
        
        return self._reconstruct_routes(giant_tour, V, pred, n)
    
    def _reconstruct_routes(self, giant_tour: List[int], V: List[float], 
                           pred: List[int], n: int) -> Tuple[List[List[int]], float]:
        """Reconstruct routes from DP solution."""
        routes = []
        j = n
        
        while j > 0:
            i = pred[j]
            if i < 0:
                return self._fallback_split(giant_tour)
            
            route_segment = giant_tour[i:j]
            route = [0] + route_segment + [0]
            routes.insert(0, route)
            j = i
        
        total_cost = V[n]
        
        if not self._validate_routes(routes):
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
        with pipeline_profiler.profile("split.fallback", metadata={'n_customers': len(giant_tour)}):
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

