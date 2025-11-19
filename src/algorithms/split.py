"""
Optimal Split Algorithm for VRP giant tour representation.
Implements Prins (2004) split algorithm for optimal route splitting.
"""

from typing import List, Tuple, Dict
from collections import deque
from src.models.vrp_model import VRPProblem
from src.core.pipeline_profiler import pipeline_profiler


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
        Split giant tour into optimal routes using optimized dynamic programming.
        
        Uses adaptive algorithm selection:
        - Small problems (n <= 200): Full DP for optimal solution
        - Medium problems (200 < n <= 500): Beam search DP with beam width
        - Large problems (n > 500): Approximate DP with limited candidates
        
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
            return self._split_linear_mqo(giant_tour)
    
    def _split_full_dp(self, giant_tour: List[int]) -> Tuple[List[List[int]], float]:
        """Full DP split for small problems (optimal solution)."""
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
    
    def _split_beam_search(self, giant_tour: List[int], beam_width: int = 20) -> Tuple[List[List[int]], float]:
        """
        Beam search DP for medium problems.
        Maintains only top-k best states at each position.
        """
        n = len(giant_tour)
        
        # Beam: list of (position, cost, predecessor)
        beam = [(0, 0.0, -1)]  # Start at position 0 with cost 0
        
        # Track best cost and predecessor for each position
        best_cost = [float('inf')] * (n + 1)
        best_cost[0] = 0.0
        pred = [-1] * (n + 1)
        
        # Pre-compute customer data
        customer_data = {}
        for idx, customer_id in enumerate(giant_tour):
            customer = self.problem.get_customer_by_id(customer_id)
            if customer:
                customer_data[idx] = (customer_id, customer.demand)
        
        capacity = self.problem.vehicle_capacity
        
        for i in range(n):
            if not beam:
                break
            
            # Expand all states in current beam
            candidates = []
            
            for pos, cost, _ in beam:
                if pos >= n:
                    continue
                
                load = 0.0
                route_cost = 0.0
                prev_customer_id = 0
                
                # Try extending route from this position
                for j in range(pos + 1, min(pos + 50, n + 1)):  # Limit route length
                    if j - 1 not in customer_data:
                        break
                    
                    customer_id, customer_demand = customer_data[j - 1]
                    
                    if load + customer_demand > capacity:
                        break
                    
                    # Calculate incremental cost
                    if j == pos + 1:
                        route_cost = (
                            self.problem.get_distance(0, customer_id) +
                            self.problem.get_distance(customer_id, 0)
                        )
                    else:
                        route_cost = (
                            route_cost -
                            self.problem.get_distance(prev_customer_id, 0) +
                            self.problem.get_distance(prev_customer_id, customer_id) +
                            self.problem.get_distance(customer_id, 0)
                        )
                    
                    load += customer_demand
                    prev_customer_id = customer_id
                    
                    new_cost = cost + route_cost
                    candidates.append((j, new_cost, pos))
                    
                    # Update best if better
                    if new_cost < best_cost[j]:
                        best_cost[j] = new_cost
                        pred[j] = pos
            
            # Keep only top beam_width candidates
            candidates.sort(key=lambda x: x[1])
            beam = candidates[:beam_width]
        
        return self._reconstruct_routes(giant_tour, best_cost, pred, n)
    
    def _split_linear_mqo(self, giant_tour: List[int]) -> Tuple[List[List[int]], float]:
        """
        MQO-enhanced O(n) split algorithm (Vidal 2016).
        Uses monotonic multi-queue optimization to maintain feasible predecessors.
        """
        n = len(giant_tour)
        if n == 0:
            return [], 0.0

        with pipeline_profiler.profile("split.linear", metadata={'n_customers': n}):
            capacity = self.problem.vehicle_capacity

            # Prepare 1-indexed arrays
            customer_seq = [0] + giant_tour
            cum_demand = [0.0] * (n + 1)
            for idx in range(1, n + 1):
                customer = self.problem.get_customer_by_id(customer_seq[idx])
                cum_demand[idx] = cum_demand[idx - 1] + (customer.demand if customer else 0.0)

            # Prefix travel cost along the giant tour (D array in Vidal's notation)
            prefix_cost = [0.0] * (n + 1)
            for idx in range(1, n):
                from_id = customer_seq[idx]
                to_id = customer_seq[idx + 1]
                prefix_cost[idx + 1] = prefix_cost[idx] + self.problem.get_distance(from_id, to_id)

            # h(j) = prefix_cost[j] + dist(node_j, depot)
            h_vals = [0.0] * (n + 1)
            for j in range(1, n + 1):
                h_vals[j] = prefix_cost[j] + self.problem.get_distance(customer_seq[j], 0)

            V = [float('inf')] * (n + 1)
            V[0] = 0.0
            pred = [-1] * (n + 1)

            g_vals = [float('inf')] * n  # defined for i = 0..n-1

            def compute_g(idx: int) -> float:
                """Compute g(i) = V[i] + dist(depot, node_{i+1}) - prefix_cost[i+1]."""
                if idx >= n:
                    return float('inf')
                next_customer_id = customer_seq[idx + 1]
                return V[idx] + self.problem.get_distance(0, next_customer_id) - prefix_cost[idx + 1]

            candidates: deque[int] = deque()
            g_vals[0] = compute_g(0)
            candidates.append(0)

            for j in range(1, n + 1):
                # Drop candidates that violate capacity
                while candidates and (cum_demand[j] - cum_demand[candidates[0]]) > capacity:
                    candidates.popleft()

                if not candidates:
                    V[j] = float('inf')
                else:
                    best_idx = candidates[0]
                    V[j] = g_vals[best_idx] + h_vals[j]
                    pred[j] = best_idx

                # Update candidate queue with index j (future predecessor)
                if j < n and V[j] < float('inf'):
                    g_j = compute_g(j)
                    g_vals[j] = g_j
                    while candidates and g_j <= g_vals[candidates[-1]]:
                        candidates.pop()
                    candidates.append(j)

            if V[n] == float('inf'):
                return self._fallback_split(giant_tour)

            return self._reconstruct_routes(giant_tour, V, pred, n)
    
    def _calculate_route_cost(self, giant_tour: List[int], customer_data: dict,
                             start_pos: int, end_pos: int) -> float:
        """
        Calculate cost of route from start_pos to end_pos in giant_tour.
        Optimized incremental calculation.
        """
        if start_pos >= end_pos:
            return 0.0
        
        total_cost = 0.0
        
        # First customer: depot -> customer
        if start_pos not in customer_data:
            return float('inf')
        
        first_customer_id = customer_data[start_pos][0]
        total_cost += self.problem.get_distance(0, first_customer_id)
        
        # Intermediate customers
        prev_customer_id = first_customer_id
        for pos in range(start_pos + 1, end_pos):
            if pos not in customer_data:
                return float('inf')
            customer_id = customer_data[pos][0]
            total_cost += self.problem.get_distance(prev_customer_id, customer_id)
            prev_customer_id = customer_id
        
        # Last customer -> depot
        total_cost += self.problem.get_distance(prev_customer_id, 0)
        
        return total_cost
    
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


