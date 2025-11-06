"""
Nearest Neighbor baseline heuristic for VRP.
Provides a simple greedy solution for comparison.
"""

import random
from typing import List, Tuple, Optional
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.algorithms.decoder import RouteDecoder
from config import GA_CONFIG


class NearestNeighborHeuristic:
    """Nearest Neighbor heuristic for VRP baseline solution."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize Nearest Neighbor heuristic.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
        # Use RouteDecoder with Split Algorithm if enabled in config
        self.decoder = RouteDecoder(problem, use_split_algorithm=GA_CONFIG.get('use_split_algorithm', False))
    
    def solve(self, num_vehicles: Optional[int] = None) -> Individual:
        """
        Solve VRP using Nearest Neighbor heuristic.
        
        Args:
            num_vehicles: Maximum number of vehicles to use
            
        Returns:
            Solution as Individual
        """
        num_vehicles = num_vehicles or self.problem.num_vehicles
        
        # Get customer IDs
        customer_ids = [c.id for c in self.problem.customers]
        
        # Build routes using Nearest Neighbor
        routes = self._build_routes(customer_ids, num_vehicles)
        
        # Create individual
        chromosome = self.decoder.encode_routes(routes)
        individual = Individual(chromosome=chromosome, routes=routes)
        
        # Calculate fitness
        total_distance = self._calculate_total_distance(routes)
        individual.total_distance = total_distance
        individual.fitness = 1.0 / (total_distance + 1.0)  # Simple fitness
        individual.is_valid = True
        
        return individual
    
    def _build_routes(self, customer_ids: List[int], num_vehicles: int) -> List[List[int]]:
        """
        Build routes using Nearest Neighbor heuristic.
        
        Args:
            customer_ids: List of customer IDs to visit
            num_vehicles: Maximum number of vehicles
            
        Returns:
            List of routes
        """
        routes = []
        unvisited = customer_ids.copy()
        
        while unvisited and len(routes) < num_vehicles:
            route = self._build_single_route(unvisited)
            routes.append(route)
        
        # If there are still unvisited customers, try to fit them in existing routes
        if unvisited:
            routes = self._fit_remaining_customers(routes, unvisited)
        
        return routes
    
    def _build_single_route(self, unvisited: List[int]) -> List[int]:
        """
        Build a single route using Nearest Neighbor.
        
        Args:
            unvisited: List of unvisited customer IDs
            
        Returns:
            Single route
        """
        if not unvisited:
            return [0, 0]  # Empty route with depot visits
        
        route = [0]  # Start at depot
        current_load = 0.0
        current_location = 0  # Start at depot
        
        while unvisited:
            # Find nearest unvisited customer that fits capacity
            nearest_customer = None
            nearest_distance = float('inf')
            
            for customer_id in unvisited:
                customer = self.problem.get_customer_by_id(customer_id)
                if customer.demand + current_load <= self.problem.vehicle_capacity:
                    distance = self.problem.get_distance(current_location, customer_id)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_customer = customer_id
            
            if nearest_customer is None:
                break  # No more customers fit
            
            # Add customer to route
            route.append(nearest_customer)
            current_load += self.problem.get_customer_by_id(nearest_customer).demand
            current_location = nearest_customer
            unvisited.remove(nearest_customer)
        
        route.append(0)  # Return to depot
        return route
    
    def _fit_remaining_customers(self, routes: List[List[int]], 
                                unvisited: List[int]) -> List[List[int]]:
        """
        Try to fit remaining customers into existing routes.
        
        Args:
            routes: Existing routes
            unvisited: List of unvisited customer IDs
            
        Returns:
            Updated routes
        """
        for customer_id in unvisited.copy():
            customer = self.problem.get_customer_by_id(customer_id)
            best_route_idx = None
            best_position = None
            best_cost_increase = float('inf')
            
            # Try to insert customer into each route
            for route_idx, route in enumerate(routes):
                if len(route) <= 2:  # Only depot visits
                    continue
                
                # Check if customer fits in this route
                route_load = sum(
                    self.problem.get_customer_by_id(c).demand 
                    for c in route if c != 0
                )
                
                if route_load + customer.demand <= self.problem.vehicle_capacity:
                    # Find best insertion position
                    for pos in range(1, len(route)):  # Skip depot at start
                        # Calculate cost increase
                        cost_increase = self._calculate_insertion_cost(
                            route, customer_id, pos
                        )
                        
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_route_idx = route_idx
                            best_position = pos
            
            # Insert customer if found good position
            if best_route_idx is not None:
                routes[best_route_idx].insert(best_position, customer_id)
                unvisited.remove(customer_id)
        
        return routes
    
    def _calculate_insertion_cost(self, route: List[int], 
                                customer_id: int, position: int) -> float:
        """
        Calculate cost increase of inserting customer at given position.
        
        Args:
            route: Route to insert into
            customer_id: Customer to insert
            position: Position to insert at
            
        Returns:
            Cost increase
        """
        if position < 1 or position >= len(route):
            return float('inf')
        
        # Calculate original cost
        original_cost = 0.0
        for i in range(len(route) - 1):
            original_cost += self.problem.get_distance(route[i], route[i + 1])
        
        # Calculate new cost
        new_cost = 0.0
        for i in range(position):
            new_cost += self.problem.get_distance(route[i], route[i + 1])
        
        new_cost += self.problem.get_distance(route[position - 1], customer_id)
        new_cost += self.problem.get_distance(customer_id, route[position])
        
        for i in range(position, len(route) - 1):
            new_cost += self.problem.get_distance(route[i], route[i + 1])
        
        return new_cost - original_cost
    
    def _calculate_total_distance(self, routes: List[List[int]]) -> float:
        """
        Calculate total distance for all routes.
        
        Args:
            routes: List of routes
            
        Returns:
            Total distance
        """
        total_distance = 0.0
        
        for route in routes:
            if len(route) < 2:
                continue
            
            for i in range(len(route) - 1):
                total_distance += self.problem.get_distance(route[i], route[i + 1])
        
        return total_distance
    
    def solve_with_improvements(self, num_vehicles: Optional[int] = None) -> Individual:
        """
        Solve VRP with additional improvements.
        
        Args:
            num_vehicles: Maximum number of vehicles to use
            
        Returns:
            Improved solution
        """
        # Get initial solution
        solution = self.solve(num_vehicles)
        
        # Apply improvements
        improved_solution = self._apply_improvements(solution)
        
        return improved_solution
    
    def _apply_improvements(self, solution: Individual) -> Individual:
        """
        Apply improvements to the solution.
        
        Args:
            solution: Initial solution
            
        Returns:
            Improved solution
        """
        improved_solution = solution.copy()
        
        # Optimize route order within each route
        optimized_routes = []
        for route in improved_solution.routes:
            if len(route) > 3:  # More than just depot visits
                optimized_route = self._optimize_route_order(route)
                optimized_routes.append(optimized_route)
            else:
                optimized_routes.append(route)
        
        improved_solution.routes = optimized_routes
        improved_solution.chromosome = self.decoder.encode_routes(optimized_routes)
        
        # Recalculate fitness
        total_distance = self._calculate_total_distance(optimized_routes)
        improved_solution.total_distance = total_distance
        improved_solution.fitness = 1.0 / (total_distance + 1.0)
        
        return improved_solution
    
    def _optimize_route_order(self, route: List[int]) -> List[int]:
        """
        Optimize customer order within a route using 2-opt.
        
        Args:
            route: Route to optimize
            
        Returns:
            Optimized route
        """
        if len(route) <= 3:  # Only depot visits
            return route
        
        # Extract customers (exclude depot)
        customers = [c for c in route if c != 0]
        
        if len(customers) <= 1:
            return route
        
        # Apply simple 2-opt optimization
        optimized_customers = self._two_opt_customers(customers)
        
        # Rebuild route with depot
        return [0] + optimized_customers + [0]
    
    def _two_opt_customers(self, customers: List[int]) -> List[int]:
        """
        Apply 2-opt optimization to customer order.
        
        Args:
            customers: List of customer IDs
            
        Returns:
            Optimized customer order
        """
        if len(customers) <= 1:
            return customers
        
        current_order = customers.copy()
        improved = True
        
        while improved:
            improved = False
            best_distance = self._calculate_customer_sequence_distance(current_order)
            
            # Try all possible 2-opt swaps
            for i in range(len(current_order) - 1):
                for j in range(i + 1, len(current_order)):
                    # Create new order by reversing segment
                    new_order = current_order.copy()
                    new_order[i:j+1] = reversed(new_order[i:j+1])
                    
                    new_distance = self._calculate_customer_sequence_distance(new_order)
                    
                    if new_distance < best_distance:
                        current_order = new_order
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_order
    
    def _calculate_customer_sequence_distance(self, customers: List[int]) -> float:
        """
        Calculate distance for a sequence of customers (including depot).
        
        Args:
            customers: List of customer IDs
            
        Returns:
            Total distance
        """
        if not customers:
            return 0.0
        
        total_distance = 0.0
        
        # Distance from depot to first customer
        total_distance += self.problem.get_distance(0, customers[0])
        
        # Distance between customers
        for i in range(len(customers) - 1):
            total_distance += self.problem.get_distance(customers[i], customers[i + 1])
        
        # Distance from last customer to depot
        total_distance += self.problem.get_distance(customers[-1], 0)
        
        return total_distance
    
    def get_solution_statistics(self, solution: Individual) -> dict:
        """
        Get statistics for the solution.
        
        Args:
            solution: Solution individual
            
        Returns:
            Dictionary with solution statistics
        """
        routes = solution.routes
        if not routes:
            routes = self.decoder.decode_chromosome(solution.chromosome)
        
        route_loads = []
        route_distances = []
        
        for route in routes:
            if not route:
                continue
            
            # Calculate route load
            route_load = sum(
                self.problem.get_customer_by_id(c).demand 
                for c in route if c != 0
            )
            route_loads.append(route_load)
            
            # Calculate route distance
            route_distance = 0.0
            for i in range(len(route) - 1):
                route_distance += self.problem.get_distance(route[i], route[i + 1])
            route_distances.append(route_distance)
        
        return {
            'num_routes': len([r for r in routes if r]),
            'total_distance': sum(route_distances),
            'avg_route_distance': sum(route_distances) / len(route_distances) if route_distances else 0.0,
            'route_loads': route_loads,
            'route_distances': route_distances,
            'avg_utilization': sum(route_loads) / (len(route_loads) * self.problem.vehicle_capacity) * 100 if route_loads else 0.0,
            'total_customers': sum(len([c for c in route if c != 0]) for route in routes)
        }


def solve_with_nearest_neighbor(problem: VRPProblem, 
                              num_vehicles: Optional[int] = None) -> Individual:
    """
    Convenience function to solve VRP with Nearest Neighbor.
    
    Args:
        problem: VRP problem instance
        num_vehicles: Maximum number of vehicles
        
    Returns:
        Solution individual
    """
    heuristic = NearestNeighborHeuristic(problem)
    return heuristic.solve(num_vehicles)
