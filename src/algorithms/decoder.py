"""
Route decoder for VRP solutions.
Converts chromosome to routes with capacity constraints.
"""

from typing import List, Tuple, Optional
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.core.pipeline_profiler import pipeline_profiler
from config import GA_CONFIG
from src.algorithms.tw_repair import TWRepairOperator


class RouteDecoder:
    """Decodes chromosome to routes with capacity constraints."""
    
    def __init__(self, problem: VRPProblem, use_split_algorithm: bool = None):
        """
        Initialize route decoder.
        
        Args:
            problem: VRP problem instance
            use_split_algorithm: Whether to use Split Algorithm (Prins 2004).
                                 If None, uses config setting or defaults to False.
        """
        self.problem = problem
        self.dataset_type = getattr(problem, 'dataset_type', None)
        
        # Determine if we should use Split Algorithm
        if use_split_algorithm is None:
            # Check if split_algorithm is in config, default to False
            self.use_split = GA_CONFIG.get('use_split_algorithm', False)
        else:
            self.use_split = use_split_algorithm
        
        # Note: Split algorithm now uses adaptive optimization
        # - Small problems (≤200): Full DP (optimal)
        # - Medium problems (200-500): Beam search (near-optimal, fast)
        # - Large problems (>500): Approximate DP (very fast)
        
        # Initialize Split Algorithm if enabled
        if self.use_split:
            try:
                from src.algorithms.split import SplitAlgorithm
                self.splitter = SplitAlgorithm(problem)
            except ImportError:
                self.use_split = False
                self.splitter = None
        else:
            self.splitter = None

        tw_repair_cfg = GA_CONFIG.get('tw_repair', {})
        self.use_tw_repair = tw_repair_cfg.get('enabled', False)
        self.apply_tw_repair_in_decoder = tw_repair_cfg.get('apply_in_decoder', False)
        
        # Get mode-specific config for repair intensity
        dataset_type = getattr(problem, 'dataset_type', None)
        # Properly detect mode: check if it's solomon, otherwise default to hanoi
        if dataset_type is None:
            # Try to infer from metadata
            metadata = getattr(problem, 'metadata', {}) or {}
            dataset_type = metadata.get('dataset_type', 'hanoi')
        dataset_type = str(dataset_type).strip().lower()
        if not dataset_type.startswith('solomon'):
            dataset_type = 'hanoi'
        
        from src.data_processing.mode_configs import get_mode_config
        self.mode_config = get_mode_config(dataset_type)
        allow_solomon_decoder = tw_repair_cfg.get('apply_in_decoder_solomon', False)
        if self.dataset_type and str(self.dataset_type).lower() == 'solomon' and not allow_solomon_decoder:
            self.apply_tw_repair_in_decoder = False

        if self.use_tw_repair:
            # Use mode-specific repair intensity
            repair_intensity = self.mode_config.repair_intensity
            if repair_intensity == "aggressive":
                # Solomon mode: aggressive repair
                max_iter = self.mode_config.max_repair_iterations
                max_iter_soft = max(1, max_iter // 3)
            else:
                # Hanoi mode: moderate repair
                max_iter = self.mode_config.max_repair_iterations
                max_iter_soft = max(1, max_iter // 2)
            
            self.tw_repair = TWRepairOperator(
                problem,
                max_iterations=max_iter,
                violation_weight=tw_repair_cfg.get('violation_weight', 50.0),
                max_relocations_per_route=tw_repair_cfg.get('max_relocations_per_route', 2),
                max_routes_to_try=tw_repair_cfg.get('max_routes_to_try', None),
                max_positions_to_try=tw_repair_cfg.get('max_positions_to_try', None),
                max_iterations_soft=max_iter_soft,
                max_routes_soft_limit=tw_repair_cfg.get('max_routes_soft_limit'),
                max_positions_soft_limit=tw_repair_cfg.get('max_positions_soft_limit'),
                lateness_soft_threshold=tw_repair_cfg.get('lateness_soft_threshold'),
                lateness_skip_threshold=tw_repair_cfg.get('lateness_skip_threshold'),
            )
        else:
            self.tw_repair = None
    
    def decode_chromosome(self, chromosome: List[int]) -> List[List[int]]:
        """
        Decode chromosome to routes using capacity and vehicle count constraints.
        
        Uses Split Algorithm (Prins 2004) if enabled, otherwise uses greedy decoder.
        IMPORTANT: Enforces num_vehicles as a HARD LIMIT - will not create more routes than allowed.
        
        Args:
            chromosome: Customer order chromosome
            
        Returns:
            List of routes (each route includes depot at start and end)
        """
        if not chromosome:
            return []
        
        with pipeline_profiler.profile(
            "decoder.decode",
            metadata={'use_split': self.use_split, 'chromosome_length': len(chromosome)}
        ):
            # Sanitize chromosome to ensure each customer appears exactly once
            chromosome = self._sanitize_chromosome(chromosome)
            
            # Use Split Algorithm if enabled
            if self.use_split and self.splitter is not None:
                try:
                    # Split Algorithm expects giant tour (list of customer IDs)
                    giant_tour = [cid for cid in chromosome if cid != 0]
                    with pipeline_profiler.profile(
                        "decoder.split",
                        metadata={'n_customers': len(giant_tour)}
                    ):
                        # Split algorithm determines vehicle count naturally via capacity
                        routes, _ = self.splitter.split(giant_tour)
                    # Validate that routes were created
                    if routes and len(routes) > 0:
                        if self.tw_repair and self.apply_tw_repair_in_decoder:
                            routes = self.tw_repair.repair_routes(routes)
                        return routes
                    else:
                        # Empty routes, fall back to greedy
                        pass
                except Exception as e:
                    # Log exception for debugging but fall back to greedy decoder
                    # This can happen if distance matrix is not available or other issues
                    import warnings
                    warnings.warn(f"Split Algorithm failed, using greedy decoder: {str(e)}")
                    pass
            
            # Greedy decoder (optimized implementation with vehicle limit)
            with pipeline_profiler.profile("decoder.greedy"):
                routes = []
                current_route = [0]  # Start at depot
                current_load = 0.0
                max_vehicles = self.problem.num_vehicles
                
                # Optimization: Pre-compute customer demands for faster lookup
                customer_demands = {c.id: c.demand for c in self.problem.customers}
                capacity = self.problem.vehicle_capacity
                
                for customer_id in chromosome:
                    if customer_id == 0:  # Skip depot in chromosome
                        continue
                        
                    customer_demand = customer_demands.get(customer_id)
                    if customer_demand is None:
                        continue
                    
                    # Check if customer fits in current route
                    if current_load + customer_demand <= capacity:
                        current_route.append(customer_id)
                        current_load += customer_demand
                    else:
                        # Check vehicle limit BEFORE creating new route
                        if len(routes) >= max_vehicles:
                            # Cannot create more routes - vehicle limit reached
                            # Try to fit customer in current route anyway (will violate capacity)
                            # This will be caught and penalized by ConstraintHandler
                            current_route.append(customer_id)
                            current_load += customer_demand
                            continue
                        
                        # Finish current route and start new one
                        current_route.append(0)  # Return to depot
                        routes.append(current_route)
                        
                        # Start new route
                        current_route = [0, customer_id]  # Start at depot, add customer
                        current_load = customer_demand
                
                # Finish last route
                if len(current_route) > 1:  # More than just depot
                    current_route.append(0)  # Return to depot
                    routes.append(current_route)
                elif current_route == [0]:
                    # Empty route, remove it
                    pass
            
            if self.tw_repair and self.apply_tw_repair_in_decoder:
                routes = self.tw_repair.repair_routes(routes)
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
                    if customer is None:
                        errors.append(f"Route {i} contains invalid customer ID: {customer_id}")
                        continue
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
                (self.problem.get_customer_by_id(c).demand if self.problem.get_customer_by_id(c) is not None else 0.0)
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
