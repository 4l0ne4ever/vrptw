"""
2-opt local search optimization for VRP.
Implements intra-route and inter-route improvements.
"""

import random
from typing import List, Tuple, Optional
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.algorithms.decoder import RouteDecoder
from src.algorithms.tw_repair import TWRepairOperator
from config import GA_CONFIG


class TwoOptOptimizer:
    """2-opt local search optimizer for VRP routes."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize 2-opt optimizer.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
        self.dataset_type = getattr(problem, 'dataset_type', None)
        # Use RouteDecoder with Split Algorithm if enabled in config
        self.decoder = RouteDecoder(problem, use_split_algorithm=GA_CONFIG.get('use_split_algorithm', False))
        tw_cfg = GA_CONFIG.get('tw_repair', {})
        self.tw_penalty_weight = tw_cfg.get('violation_weight', 50.0)
        self.tw_repair_operator = None
        self.apply_tw_repair_after_local_search = tw_cfg.get('apply_after_local_search', False)
        allow_solomon = tw_cfg.get('apply_after_local_search_solomon', True)
        if self.dataset_type and str(self.dataset_type).lower() == 'solomon' and not allow_solomon:
            self.apply_tw_repair_after_local_search = False
        
        if tw_cfg.get('enabled', False) and self.apply_tw_repair_after_local_search:
            self.tw_repair_operator = TWRepairOperator(
                problem,
                max_iterations=tw_cfg.get('max_iterations', 50),
                violation_weight=tw_cfg.get('violation_weight', 50.0),
                max_relocations_per_route=tw_cfg.get('max_relocations_per_route', 2),
                max_routes_to_try=tw_cfg.get('max_routes_to_try', None),
                max_positions_to_try=tw_cfg.get('max_positions_to_try', None),
                max_iterations_soft=tw_cfg.get('max_iterations_soft'),
                max_routes_soft_limit=tw_cfg.get('max_routes_soft_limit'),
                max_positions_soft_limit=tw_cfg.get('max_positions_soft_limit'),
                lateness_soft_threshold=tw_cfg.get('lateness_soft_threshold'),
                lateness_skip_threshold=tw_cfg.get('lateness_skip_threshold'),
            )
    
    def optimize_individual(self, individual: Individual, 
                           max_iterations: int = 100) -> Individual:
        """
        Optimize individual using 2-opt local search.
        
        Args:
            individual: Individual to optimize
            max_iterations: Maximum iterations
            
        Returns:
            Optimized individual
        """
        if individual.is_empty():
            return individual
        
        # Decode chromosome to routes
        routes = self.decoder.decode_chromosome(individual.chromosome)
        
        # Apply 2-opt optimization
        optimized_routes = self.optimize_routes(routes, max_iterations)
        
        # Apply TW repair selectively after local search (cheaper than per-decode)
        if self.tw_repair_operator:
            optimized_routes = self.tw_repair_operator.repair_routes(optimized_routes)
        
        # Update individual
        optimized_individual = individual.copy()
        optimized_individual.routes = optimized_routes
        
        # Re-encode chromosome
        optimized_individual.chromosome = self.decoder.encode_routes(optimized_routes)
        
        # Recalculate total distance after optimization
        total_distance = 0.0
        for route in optimized_routes:
            if not route:
                continue
            for i in range(len(route) - 1):
                total_distance += self.problem.get_distance(route[i], route[i + 1])
        optimized_individual.total_distance = total_distance
        
        return optimized_individual
    
    def optimize_routes(self, routes: List[List[int]], 
                       max_iterations: int = 100) -> List[List[int]]:
        """
        Optimize routes using 2-opt local search.
        
        Args:
            routes: List of routes to optimize
            max_iterations: Maximum iterations
            
        Returns:
            Optimized routes
        """
        if not routes:
            return routes
        
        optimized_routes = []
        
        # Optimize each route individually
        for route in routes:
            if len(route) <= 3:  # Only depot visits
                optimized_routes.append(route)
                continue
            
            optimized_route = self._two_opt_single_route(route, max_iterations)
            optimized_routes.append(optimized_route)
        
        # Apply inter-route optimizations (can be slow, so limit or skip for small iterations)
        # Inter-route optimization has O(n^2 * m^2) complexity where n=routes, m=customers per route
        # Skip it if max_iterations is too low to avoid performance issues
        if max_iterations >= 20 and len(optimized_routes) <= 10:  # Only for reasonable cases
            try:
                optimized_routes = self._inter_route_optimization(optimized_routes, min(max_iterations // 2, 5))
            except Exception:
                # If inter-route fails, continue with intra-route optimized routes
                pass
        # Skip inter-route for high iteration counts or many routes to avoid hanging
        
        return optimized_routes
    
    def _two_opt_single_route(self, route: List[int], 
                            max_iterations: int = 100) -> List[int]:
        """
        Apply 2-opt optimization to a single route.
        
        Args:
            route: Route to optimize
            max_iterations: Maximum iterations
            
        Returns:
            Optimized route
        """
        if len(route) <= 3:  # Only depot visits
            return route
        
        current_route = route.copy()
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            best_distance = self._calculate_route_distance(current_route)
            
            # Try all possible 2-opt swaps
            for i in range(1, len(current_route) - 2):  # Skip depot at start
                for j in range(i + 1, len(current_route) - 1):  # Skip depot at end
                    # Create new route by reversing segment
                    new_route = self._reverse_segment(current_route, i, j)
                    new_distance = self._calculate_route_distance(new_route)
                    
                    # Check if improvement and capacity constraints are maintained
                    if (new_distance < best_distance and 
                        self._check_route_capacity(new_route)):
                        current_route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_route
    
    def _reverse_segment(self, route: List[int], i: int, j: int) -> List[int]:
        """
        Reverse segment of route between positions i and j.
        
        Args:
            route: Original route
            i: Start position
            j: End position
            
        Returns:
            Route with reversed segment
        """
        new_route = route.copy()
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return new_route
    
    def _calculate_route_distance(self, route: List[int]) -> float:
        """
        Calculate total distance for a route.
        
        Args:
            route: Route to calculate distance for
            
        Returns:
            Total route distance
        """
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self.problem.get_distance(route[i], route[i + 1])
        
        return total_distance

    def _calculate_route_tw_penalty(self, route: List[int]) -> float:
        """Calculate aggregate lateness for a route."""
        if len(route) < 2:
            return 0.0
        
        lateness = 0.0
        current_time = 0.0
        for idx in range(1, len(route) - 1):
            prev = route[idx - 1]
            cid = route[idx]
            customer = self.problem.get_customer_by_id(cid)
            if customer is None:
                # Invalid customer ID, skip
                continue
            travel = self.problem.get_distance(prev, cid)
            arrival = current_time + travel
            if arrival < customer.ready_time:
                arrival = customer.ready_time
            lateness += max(0.0, arrival - customer.due_date)
            current_time = arrival + customer.service_time
        return lateness
    
    def _check_route_capacity(self, route: List[int]) -> bool:
        """
        Check if route respects capacity constraints.
        
        Args:
            route: Route to check
            
        Returns:
            True if route respects capacity constraints, False otherwise
        """
        if not route:
            return True
        
        total_load = 0.0
        for customer_id in route:
            if customer_id != 0:  # Skip depot
                try:
                    customer = self.problem.get_customer_by_id(customer_id)
                    total_load += customer.demand
                except:
                    # If customer not found, assume zero demand
                    pass
        
        return total_load <= self.problem.vehicle_capacity
    
    def _inter_route_optimization(self, routes: List[List[int]], 
                                 max_iterations: int = 100) -> List[List[int]]:
        """
        Apply inter-route optimizations (customer swaps between routes).
        
        Args:
            routes: List of routes to optimize
            max_iterations: Maximum iterations
            
        Returns:
            Optimized routes
        """
        if len(routes) < 2:
            return routes
        
        optimized_routes = [route.copy() for route in routes]
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            current_distance = sum(self._calculate_route_distance(route) for route in optimized_routes)
            current_penalty = sum(self._calculate_route_tw_penalty(route) for route in optimized_routes)
            current_score = current_distance + self.tw_penalty_weight * current_penalty
            
            # Try swapping customers between routes
            for i in range(len(optimized_routes)):
                for j in range(i + 1, len(optimized_routes)):
                    route1 = optimized_routes[i]
                    route2 = optimized_routes[j]
                    
                    # Try swapping customers
                    for k in range(1, len(route1) - 1):  # Skip depot visits
                        for l in range(1, len(route2) - 1):  # Skip depot visits
                            customer1 = route1[k]
                            customer2 = route2[l]
                            
                            # Check capacity constraints
                            if self._can_swap_customers(route1, route2, k, l):
                                # Perform swap
                                new_route1 = route1.copy()
                                new_route2 = route2.copy()
                                new_route1[k] = customer2
                                new_route2[l] = customer1
                                
                                # Calculate new total distance and TW penalty
                                new_distance = (
                                    self._calculate_route_distance(new_route1)
                                    + self._calculate_route_distance(new_route2)
                                )
                                new_penalty = (
                                    self._calculate_route_tw_penalty(new_route1)
                                    + self._calculate_route_tw_penalty(new_route2)
                                )
                                new_score = new_distance + self.tw_penalty_weight * new_penalty
                                
                                if new_score < current_score:
                                    optimized_routes[i] = new_route1
                                    optimized_routes[j] = new_route2
                                    current_distance = new_distance
                                    current_penalty = new_penalty
                                    current_score = new_score
                                    improved = True
                                    break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return optimized_routes
    
    def _can_swap_customers(self, route1: List[int], route2: List[int], 
                           pos1: int, pos2: int) -> bool:
        """
        Check if swapping customers between routes maintains capacity constraints.
        
        Args:
            route1: First route
            route2: Second route
            pos1: Position in first route
            pos2: Position in second route
            
        Returns:
            True if swap is feasible, False otherwise
        """
        customer1 = route1[pos1]
        customer2 = route2[pos2]
        
        # Get customer demands
        demand1 = self.problem.get_customer_by_id(customer1).demand
        demand2 = self.problem.get_customer_by_id(customer2).demand
        
        # Calculate current loads
        load1 = sum(self.problem.get_customer_by_id(c).demand for c in route1 if c != 0)
        load2 = sum(self.problem.get_customer_by_id(c).demand for c in route2 if c != 0)
        
        # Check if swap maintains capacity constraints
        new_load1 = load1 - demand1 + demand2
        new_load2 = load2 - demand2 + demand1
        
        return (new_load1 <= self.problem.vehicle_capacity and 
                new_load2 <= self.problem.vehicle_capacity)
    
    def optimize_with_restarts(self, individual: Individual, 
                             num_restarts: int = 5,
                             max_iterations: int = 100) -> Individual:
        """
        Optimize individual with multiple restarts for better results.
        
        Args:
            individual: Individual to optimize
            num_restarts: Number of restart attempts
            max_iterations: Maximum iterations per restart
            
        Returns:
            Best optimized individual found
        """
        if individual.is_empty():
            return individual
        
        best_individual = individual.copy()
        best_distance = self._calculate_solution_distance(best_individual)
        
        for _ in range(num_restarts):
            # Create a copy for optimization
            current_individual = individual.copy()
            
            # Randomize chromosome slightly for different starting point
            if len(current_individual.chromosome) > 1:
                # Random swap to create different starting point
                pos1, pos2 = random.sample(range(len(current_individual.chromosome)), 2)
                current_individual.chromosome[pos1], current_individual.chromosome[pos2] = \
                    current_individual.chromosome[pos2], current_individual.chromosome[pos1]
            
            # Optimize
            optimized_individual = self.optimize_individual(current_individual, max_iterations)
            optimized_distance = self._calculate_solution_distance(optimized_individual)
            
            # Keep best solution
            if optimized_distance < best_distance:
                best_individual = optimized_individual
                best_distance = optimized_distance
        
        return best_individual
    
    def _calculate_solution_distance(self, individual: Individual) -> float:
        """
        Calculate total distance for a solution.
        
        Args:
            individual: Individual to calculate distance for
            
        Returns:
            Total solution distance
        """
        if individual.routes:
            return sum(self._calculate_route_distance(route) for route in individual.routes)
        else:
            routes = self.decoder.decode_chromosome(individual.chromosome)
            return sum(self._calculate_route_distance(route) for route in routes)
    
    def get_improvement_statistics(self, original: Individual, 
                                 optimized: Individual) -> dict:
        """
        Get improvement statistics.
        
        Args:
            original: Original individual
            optimized: Optimized individual
            
        Returns:
            Dictionary with improvement statistics
        """
        original_distance = self._calculate_solution_distance(original)
        optimized_distance = self._calculate_solution_distance(optimized)
        
        improvement = original_distance - optimized_distance
        improvement_percent = (improvement / original_distance) * 100 if original_distance > 0 else 0
        
        return {
            'original_distance': original_distance,
            'optimized_distance': optimized_distance,
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'is_improved': improvement > 0
        }


def optimize_with_two_opt(individual: Individual, problem: VRPProblem, 
                         max_iterations: int = 100) -> Individual:
    """
    Convenience function to optimize individual with 2-opt.
    
    Args:
        individual: Individual to optimize
        problem: VRP problem instance
        max_iterations: Maximum iterations
        
    Returns:
        Optimized individual
    """
    optimizer = TwoOptOptimizer(problem)
    return optimizer.optimize_individual(individual, max_iterations)
