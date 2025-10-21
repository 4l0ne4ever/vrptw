"""
Route decoder for VRP solutions.
Converts chromosome to routes with capacity constraints.
"""

from typing import List, Tuple, Optional
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem


class RouteDecoder:
    """Decodes chromosome to routes with capacity constraints."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize route decoder.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
    
    def decode_chromosome(self, chromosome: List[int]) -> List[List[int]]:
        """
        Decode chromosome to routes using capacity constraint.
        
        Args:
            chromosome: Customer order chromosome
            
        Returns:
            List of routes (each route includes depot at start and end)
        """
        if not chromosome:
            return []
        
        # Sanitize chromosome to ensure each customer appears exactly once
        chromosome = self._sanitize_chromosome(chromosome)
        
        routes = []
        current_route = [0]  # Start at depot
        current_load = 0.0
        
        for customer_id in chromosome:
            customer = self.problem.get_customer_by_id(customer_id)
            if customer is None:
                continue
            
            customer_demand = customer.demand
            
            # Check if customer fits in current route
            if current_load + customer_demand <= self.problem.vehicle_capacity:
                current_route.append(customer_id)
                current_load += customer_demand
            else:
                # Finish current route and start new one
                current_route.append(0)  # Return to depot
                routes.append(current_route)
                
                # Start new route
                current_route = [0, customer_id]  # Start at depot, add customer
                current_load = customer_demand
        
        # Finish last route
        if current_route:
            current_route.append(0)  # Return to depot
            routes.append(current_route)
        
        return routes

    def _sanitize_chromosome(self, chromosome: List[int]) -> List[int]:
        """Ensure chromosome covers all customers exactly once.
        - Removes invalid IDs
        - Removes duplicates (keeps first occurrence)
        - Appends any missing customers at the end
        """
        valid_customers = {c.id for c in self.problem.customers}

        seen = set()
        sanitized: List[int] = []
        for cid in chromosome:
            if cid in valid_customers and cid not in seen:
                sanitized.append(cid)
                seen.add(cid)

        # Append any missing customers
        missing = [cid for cid in sorted(valid_customers) if cid not in seen]
        sanitized.extend(missing)

        return sanitized
    
    def decode_individual(self, individual: Individual) -> Individual:
        """
        Decode individual's chromosome to routes.
        
        Args:
            individual: Individual to decode
            
        Returns:
            Individual with decoded routes
        """
        routes = self.decode_chromosome(individual.chromosome)
        individual.routes = routes
        return individual
    
    def encode_routes(self, routes: List[List[int]]) -> List[int]:
        """
        Encode routes back to chromosome.
        
        Args:
            routes: List of routes
            
        Returns:
            Chromosome representation
        """
        chromosome = []
        
        for route in routes:
            # Skip depot visits (0)
            for customer_id in route:
                if customer_id != 0:
                    chromosome.append(customer_id)
        
        return chromosome
    
    def validate_routes(self, routes: List[List[int]]) -> Tuple[bool, List[str]]:
        """
        Validate decoded routes.
        
        Args:
            routes: List of routes to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if all customers are visited
        visited_customers = set()
        for route in routes:
            for customer_id in route:
                if customer_id != 0:  # Skip depot
                    if customer_id in visited_customers:
                        errors.append(f"Customer {customer_id} visited multiple times")
                    visited_customers.add(customer_id)
        
        # Check if all customers are visited
        all_customers = set(c.id for c in self.problem.customers)
        missing_customers = all_customers - visited_customers
        if missing_customers:
            errors.append(f"Missing customers: {missing_customers}")
        
        # Check capacity constraints
        for i, route in enumerate(routes):
            route_load = 0.0
            for customer_id in route:
                if customer_id != 0:  # Skip depot
                    customer = self.problem.get_customer_by_id(customer_id)
                    route_load += customer.demand
            
            if route_load > self.problem.vehicle_capacity:
                errors.append(f"Route {i} exceeds capacity: {route_load} > {self.problem.vehicle_capacity}")
        
        # Check depot constraints
        for i, route in enumerate(routes):
            if route and (route[0] != 0 or route[-1] != 0):
                errors.append(f"Route {i} doesn't start/end at depot")
        
        return len(errors) == 0, errors
    
    def get_route_statistics(self, routes: List[List[int]]) -> dict:
        """
        Get statistics for routes.
        
        Args:
            routes: List of routes
            
        Returns:
            Dictionary with route statistics
        """
        if not routes:
            return {
                'num_routes': 0,
                'total_customers': 0,
                'avg_route_length': 0.0,
                'route_loads': [],
                'route_utilizations': [],
                'avg_utilization': 0.0
            }
        
        route_loads = []
        route_lengths = []
        
        for route in routes:
            if not route:
                continue
            
            # Calculate route load
            route_load = 0.0
            customer_count = 0
            
            for customer_id in route:
                if customer_id != 0:  # Skip depot
                    customer = self.problem.get_customer_by_id(customer_id)
                    route_load += customer.demand
                    customer_count += 1
            
            route_loads.append(route_load)
            route_lengths.append(customer_count)
        
        # Calculate utilizations
        route_utilizations = [
            (load / self.problem.vehicle_capacity) * 100 
            for load in route_loads
        ]
        
        return {
            'num_routes': len([r for r in routes if r]),
            'total_customers': sum(route_lengths),
            'avg_route_length': sum(route_lengths) / len(route_lengths) if route_lengths else 0.0,
            'route_loads': route_loads,
            'route_utilizations': route_utilizations,
            'avg_utilization': sum(route_utilizations) / len(route_utilizations) if route_utilizations else 0.0,
            'min_utilization': min(route_utilizations) if route_utilizations else 0.0,
            'max_utilization': max(route_utilizations) if route_utilizations else 0.0
        }
    
    def optimize_route_order(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Optimize order of customers within routes using nearest neighbor.
        
        Args:
            routes: List of routes to optimize
            
        Returns:
            Optimized routes
        """
        optimized_routes = []
        
        for route in routes:
            if len(route) <= 3:  # Only depot visits
                optimized_routes.append(route)
                continue
            
            # Extract customers (exclude depot)
            customers = [c for c in route if c != 0]
            
            if not customers:
                optimized_routes.append(route)
                continue
            
            # Optimize customer order using nearest neighbor
            optimized_customers = self._nearest_neighbor_route(customers)
            
            # Rebuild route with depot
            optimized_route = [0] + optimized_customers + [0]
            optimized_routes.append(optimized_route)
        
        return optimized_routes
    
    def _nearest_neighbor_route(self, customers: List[int]) -> List[int]:
        """
        Optimize customer order using nearest neighbor heuristic.
        
        Args:
            customers: List of customer IDs
            
        Returns:
            Optimized customer order
        """
        if len(customers) <= 1:
            return customers
        
        optimized = []
        unvisited = customers.copy()
        current = 0  # Start at depot
        
        while unvisited:
            # Find nearest unvisited customer
            nearest = min(unvisited, key=lambda c: self.problem.get_distance(current, c))
            optimized.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return optimized
    
    def split_route_by_capacity(self, chromosome: List[int], 
                               max_capacity: Optional[float] = None) -> List[List[int]]:
        """
        Split chromosome into routes respecting capacity constraint.
        
        Args:
            chromosome: Customer order chromosome
            max_capacity: Maximum capacity per route
            
        Returns:
            List of routes
        """
        max_capacity = max_capacity or self.problem.vehicle_capacity
        
        routes = []
        current_route = [0]  # Start at depot
        current_load = 0.0
        
        for customer_id in chromosome:
            customer = self.problem.get_customer_by_id(customer_id)
            if customer is None:
                continue
            
            customer_demand = customer.demand
            
            # Check if customer fits in current route
            if current_load + customer_demand <= max_capacity:
                current_route.append(customer_id)
                current_load += customer_demand
            else:
                # Finish current route and start new one
                current_route.append(0)  # Return to depot
                routes.append(current_route)
                
                # Start new route
                current_route = [0, customer_id]  # Start at depot, add customer
                current_load = customer_demand
        
        # Finish last route
        if current_route:
            current_route.append(0)  # Return to depot
            routes.append(current_route)
        
        return routes
    
    def merge_routes(self, routes: List[List[int]]) -> List[int]:
        """
        Merge routes into a single chromosome.
        
        Args:
            routes: List of routes
            
        Returns:
            Merged chromosome
        """
        chromosome = []
        
        for route in routes:
            # Skip depot visits (0)
            for customer_id in route:
                if customer_id != 0:
                    chromosome.append(customer_id)
        
        return chromosome
    
    def repair_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Repair routes by fixing capacity violations.
        
        Args:
            routes: List of routes to repair
            
        Returns:
            Repaired routes
        """
        repaired_routes = []
        
        for route in routes:
            if not route:
                continue
            
            # Check capacity
            route_load = sum(
                self.problem.get_customer_by_id(c).demand 
                for c in route if c != 0
            )
            
            if route_load <= self.problem.vehicle_capacity:
                repaired_routes.append(route)
            else:
                # Split route if capacity exceeded
                split_routes = self.split_route_by_capacity(
                    [c for c in route if c != 0]
                )
                repaired_routes.extend(split_routes)
        
        return repaired_routes


def decode_chromosome(chromosome: List[int], problem: VRPProblem) -> List[List[int]]:
    """
    Convenience function to decode chromosome.
    
    Args:
        chromosome: Customer order chromosome
        problem: VRP problem instance
        
    Returns:
        List of routes
    """
    decoder = RouteDecoder(problem)
    return decoder.decode_chromosome(chromosome)
