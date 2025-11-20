"""
Fitness evaluation for VRP solutions.
Implements multi-objective fitness function with penalty handling.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.data_processing.constraints import ConstraintHandler
from src.algorithms.decoder import RouteDecoder
from src.core.pipeline_profiler import pipeline_profiler
from config import GA_CONFIG, VRP_CONFIG
import os
import time
import json


class FitnessEvaluator:
    """Evaluates fitness of VRP solutions."""
    
    def __init__(self, problem: VRPProblem, penalty_weight: Optional[float] = None):
        """
        Initialize fitness evaluator.
        
        Args:
            problem: VRP problem instance
            penalty_weight: Weight for constraint violations (defaults to VRP_CONFIG or GA config)
        """
        self.problem = problem
        
        # Get penalty_weight from config if not provided
        if penalty_weight is None:
            # Try to get from VRP_CONFIG first, then default to 5000
            penalty_weight = VRP_CONFIG.get('penalty_weight', 5000)
        
        self.penalty_weight = penalty_weight
        self.constraint_handler = ConstraintHandler(
            problem.vehicle_capacity, 
            problem.num_vehicles,
            penalty_weight=self.penalty_weight  # Pass penalty_weight to ConstraintHandler
        )
        # Use RouteDecoder to ensure Split Algorithm is used when enabled
        self.decoder = RouteDecoder(problem, use_split_algorithm=GA_CONFIG.get('use_split_algorithm', False))
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness value (higher is better)
        """
        if individual.is_empty():
            return 0.0
        
        chromosome_length = len(individual.chromosome) if getattr(individual, 'chromosome', None) else 0
        with pipeline_profiler.profile("fitness.evaluate", metadata={'chromosome_length': chromosome_length}):
            # Decode chromosome to routes using RouteDecoder (which uses Split Algorithm if enabled)
            with pipeline_profiler.profile("fitness.decode"):
                routes = self.decoder.decode_chromosome(individual.chromosome)

            # Early capacity repair (two-phase) to ensure feasible routes before scoring
            demands = [c.demand for c in self.problem.customers]
            # Ensure demands array covers all possible customer IDs (1 to N)
            # BUG FIX: Limit max_customer_id to prevent infinite growth
            max_expected_id = len(self.problem.customers)
            max_customer_id = max(max(route) for route in routes) if routes else 0
            
            # Cap customer ID to prevent runaway growth
            if max_customer_id > max_expected_id * 2:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Abnormal max_customer_id={max_customer_id}, capping to {max_expected_id}")
                max_customer_id = max_expected_id
            
            while len(demands) < max_customer_id:
                demands.append(0.0)  # Add zero demand for missing customers
            
            # Sanitize routes to valid node ids [0..N] (N = total nodes including depot)
            num_nodes = len(self.problem.customers) + 1  # +1 for depot
            routes = [
                [node for node in route if 0 <= int(node) <= num_nodes]
                for route in routes
            ]
            
            cap_valid, _ = self.constraint_handler.validate_capacity_constraint(routes, demands)
            if not cap_valid:
                try:
                    with pipeline_profiler.profile("fitness.repair_capacity"):
                        routes = self.constraint_handler.repair_capacity_violations(routes, demands)
                except Exception:
                    # if repair fails, keep original routes and let penalty handle infeasibility
                    pass
            individual.routes = routes
            
            # Selective TW repair: only for individuals with moderate violations
            # Skip repair for catastrophic violations (let penalty handle) and no violations (already good)
            tw_repair_cfg = GA_CONFIG.get('tw_repair', {})
            if (tw_repair_cfg.get('enabled', False) and 
                hasattr(self.decoder, 'tw_repair') and 
                self.decoder.tw_repair is not None):
                # Quick check: calculate total lateness
                try:
                    total_lateness = sum(
                        self.decoder.tw_repair._route_lateness(route) for route in routes
                    )
                    lateness_soft = tw_repair_cfg.get('lateness_soft_threshold', 5000.0)
                    lateness_skip = tw_repair_cfg.get('lateness_skip_threshold', 100000.0)
                    
                    # Repair if lateness is above soft threshold (moderate to high violations)
                    # Skip only if catastrophic (above skip threshold)
                    if lateness_soft < total_lateness < lateness_skip:
                        # Apply quick repair (soft mode - limited iterations)
                        try:
                            with pipeline_profiler.profile("fitness.selective_tw_repair"):
                                routes = self.decoder.tw_repair.repair_routes(routes)
                                individual.routes = routes
                        except Exception:
                            # If repair fails, continue with original routes
                            pass
                except Exception:
                    # If lateness calculation fails, skip repair
                    pass
            
            # Calculate total distance
            total_distance = self._calculate_total_distance(routes)
            individual.total_distance = total_distance
            
            # Calculate penalty for constraint violations
            raw_penalty = self._calculate_penalty(routes)
            
            # Calculate balance factor
            balance_factor = self._calculate_balance_factor(routes)
            
            # Distance-first fitness (like NN and BKS)
            # Penalties guide toward feasibility but don't dominate
            if raw_penalty > 0:
                # Adaptive penalty cap based on mode and problem size
                # Loose time windows need higher caps (violations less severe)
                # Tight time windows need lower caps (violations more critical)
                
                if self.penalty_weight <= 1500:  # Hanoi mode
                    max_penalty_cap = total_distance * 10
                else:  # Solomon mode
                    # Adaptive cap for Solomon datasets
                    # Tight TW (C1/R1/RC1): 2× distance (strict)
                    # Loose TW (C2/R2/RC2): 10× distance (allow more violations)
                    num_customers = len(self.problem.customers)
                    
                    # Heuristic: Loose TW if vehicle_capacity >= 700
                    # C1/R1/RC1 use capacity 200, C2/R2/RC2 use 700/1000
                    is_loose_tw = self.problem.vehicle_capacity >= 700
                    
                    if is_loose_tw:
                        # Loose TW: much higher cap to allow GA to see penalty gradient
                        # Increased from 20× to 50× because violations can be very high (36000-65000)
                        # Penalty was hitting 100% cap, preventing GA from learning
                        max_penalty_cap = total_distance * 50.0
                    else:
                        # Tight TW: lower cap to enforce constraints
                        max_penalty_cap = total_distance * 2.0
                
                capped_penalty = min(raw_penalty, max_penalty_cap)
                
                # Use capped penalty directly
                fitness = 1.0 / (total_distance + capped_penalty + balance_factor + 1.0)
            else:
                fitness = 1.0 / (total_distance + balance_factor + 1.0)
            
            individual.fitness = fitness
            
            # Store capped penalty for reporting
            individual.penalty = capped_penalty if raw_penalty > 0 else 0.0
            
            # Mark as valid if no penalty
            individual.is_valid = raw_penalty == 0.0
            
            return fitness
    
    def _decode_chromosome(self, chromosome: List[int]) -> List[List[int]]:
        """
        Decode chromosome to routes using RouteDecoder.
        
        This method is kept for backward compatibility but now uses RouteDecoder
        which supports Split Algorithm when enabled.
        
        Args:
            chromosome: Customer order chromosome
            
        Returns:
            List of routes (each route includes depot at start and end)
        """
        # Use RouteDecoder which handles Split Algorithm if enabled
        return self.decoder.decode_chromosome(chromosome)
    
    def _calculate_total_distance(self, routes: List[List[int]]) -> float:
        """
        Calculate total distance for all routes with adaptive traffic factor.
        
        Args:
            routes: List of routes
            
        Returns:
            Total distance (with adaptive traffic factor if enabled)
        """
        with pipeline_profiler.profile("fitness.distance", metadata={'num_routes': len(routes)}):
            total_distance = 0.0
            
            # Get time window start (default 8:00 = 480 minutes)
            time_window_start = VRP_CONFIG.get('time_window_start', 480)
            
            for route in routes:
                if len(route) < 2:
                    continue
                
                # Calculate route schedule to determine travel times
                current_time = time_window_start  # Start from time window start
                route_distance = 0.0
                
                for i in range(len(route) - 1):
                    from_id = route[i]
                    to_id = route[i + 1]
                    
                    # Get base distance
                    base_dist = self.problem.get_distance(from_id, to_id)
                    
                    # Apply adaptive traffic factor if enabled
                    if self.problem.use_adaptive_traffic:
                        # Assume average speed 30 km/h = 0.5 km/min
                        # Travel time in minutes = distance / 0.5
                        # But we need to account for traffic, so use base distance for time estimation
                        travel_time_minutes = base_dist / 0.5  # Rough estimate
                        
                        # Get adaptive distance with current time
                        segment_distance = self.problem.get_adaptive_distance(
                            from_id, to_id, current_time
                        )
                        
                        # Update current time (account for traffic in travel time)
                        # Travel time increases with traffic factor
                        if self.problem.distance_calculator:
                            traffic_factor = self.problem.distance_calculator.get_adaptive_traffic_factor(current_time)
                            actual_travel_time = travel_time_minutes * traffic_factor
                        else:
                            actual_travel_time = travel_time_minutes
                        
                        current_time += actual_travel_time
                        
                        # Add service time if arriving at customer (not depot)
                        if to_id != 0:
                            customer = self.problem.get_customer_by_id(to_id)
                            if customer:
                                # Wait if arrived early
                                if current_time < customer.ready_time:
                                    current_time = customer.ready_time
                                current_time += customer.service_time
                        
                        route_distance += segment_distance
                    else:
                        # Non-adaptive mode: use base distance with fixed traffic factor
                        traffic_factor = VRP_CONFIG.get('traffic_factor', 1.0)
                        route_distance += base_dist * traffic_factor
                
                total_distance += route_distance
            
            return total_distance
    
    def _calculate_penalty(self, routes: List[List[int]]) -> float:
        """
        Calculate penalty for constraint violations.
        
        Args:
            routes: List of routes
            
        Returns:
            Total penalty
        """
        with pipeline_profiler.profile("fitness.penalty", metadata={'num_routes': len(routes)}):
            demands = self.problem.get_demands()
            num_customers = len(self.problem.customers)
            
            # Get time windows and service times for validation
            time_windows = self.problem.get_time_windows()
            service_times = self.problem.get_service_times()
            distance_matrix = self.problem.distance_matrix
            
            # Build distance matrix with proper ID mapping for constraint validation
            # Constraint handler needs distance matrix indexed by customer IDs, not matrix indices
            # We'll build a helper matrix or pass id_to_index mapping
            id_to_index = self.problem.id_to_index if hasattr(self.problem, 'id_to_index') else None
            
            # Validate constraints (including time windows if available)
            validation_results = self.constraint_handler.validate_all_constraints(
                routes, 
                demands, 
                self.problem.customers,
                time_windows=time_windows if time_windows else None,
                service_times=service_times if service_times else None,
                distance_matrix=distance_matrix if distance_matrix is not None else None,
                id_to_index=id_to_index  # Pass ID to index mapping
            )
            
            return validation_results['total_penalty']
    
    def _calculate_balance_factor(self, routes: List[List[int]]) -> float:
        """
        Calculate balance factor to encourage balanced route loads.
        
        Args:
            routes: List of routes
            
        Returns:
            Balance factor (lower is better)
        """
        with pipeline_profiler.profile("fitness.balance", metadata={'num_routes': len(routes)}):
            if not routes:
                return 0.0
            
            route_loads = []
            for route in routes:
                if not route:
                    continue
                
                route_load = 0.0
                for customer_id in route:
                    if customer_id != 0:  # Skip depot
                        customer = self.problem.get_customer_by_id(customer_id)
                        if customer is None:
                            # Invalid customer ID, skip
                            continue
                        route_load += customer.demand
                
                route_loads.append(route_load)
            
            if not route_loads:
                return 0.0
            
            # Calculate standard deviation of route loads
            if len(route_loads) == 1:
                return 0.0
            
            load_std = np.std(route_loads)
            return load_std * 10  # Scale factor
    
    def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """
        Evaluate fitness for entire population.
        
        Args:
            population: List of individuals
            
        Returns:
            Population with updated fitness values
        """
        for individual in population:
            self.evaluate_fitness(individual)
        
        return population
    
    def get_route_statistics(self, routes: List[List[int]]) -> Dict:
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
                'total_distance': 0.0,
                'avg_route_length': 0.0,
                'route_loads': [],
                'route_utilizations': [],
                'avg_utilization': 0.0
            }
        
        route_loads = []
        route_distances = []
        route_lengths = []
        
        for route in routes:
            if not route:
                continue
            
            # Calculate route load
            route_load = 0.0
            for customer_id in route:
                if customer_id != 0:  # Skip depot
                    customer = self.problem.get_customer_by_id(customer_id)
                    route_load += customer.demand
            
            route_loads.append(route_load)
            
            # Calculate route distance
            route_distance = 0.0
            for i in range(len(route) - 1):
                route_distance += self.problem.get_distance(route[i], route[i + 1])
            
            route_distances.append(route_distance)
            route_lengths.append(len(route) - 2)  # Exclude depot visits
        
        # Calculate utilizations
        route_utilizations = [
            (load / self.problem.vehicle_capacity) * 100 
            for load in route_loads
        ]
        
        return {
            'num_routes': len([r for r in routes if r]),
            'total_distance': sum(route_distances),
            'avg_route_length': np.mean(route_lengths) if route_lengths else 0.0,
            'route_loads': route_loads,
            'route_distances': route_distances,
            'route_utilizations': route_utilizations,
            'avg_utilization': np.mean(route_utilizations) if route_utilizations else 0.0,
            'load_balance': np.std(route_loads) if len(route_loads) > 1 else 0.0
        }
    
    def is_feasible_solution(self, individual: Individual) -> bool:
        """
        Check if solution is feasible.
        
        Args:
            individual: Individual to check
            
        Returns:
            True if feasible, False otherwise
        """
        if individual.is_empty():
            return False
        
        routes = self._decode_chromosome(individual.chromosome)
        penalty = self._calculate_penalty(routes)
        
        return penalty == 0.0
    
    def get_best_feasible_solution(self, population: List[Individual]) -> Optional[Individual]:
        """
        Get best feasible solution from population.
        
        Args:
            population: List of individuals
            
        Returns:
            Best feasible individual or None
        """
        feasible_solutions = [
            ind for ind in population 
            if self.is_feasible_solution(ind)
        ]
        
        if not feasible_solutions:
            return None
        
        return max(feasible_solutions, key=lambda ind: ind.fitness)


def evaluate_fitness(individual: Individual, problem: VRPProblem, 
                    penalty_weight: float = 1000) -> float:
    """
    Convenience function to evaluate fitness.
    
    Args:
        individual: Individual to evaluate
        problem: VRP problem instance
        penalty_weight: Weight for constraint violations
        
    Returns:
        Fitness value
    """
    evaluator = FitnessEvaluator(problem, penalty_weight)
    return evaluator.evaluate_fitness(individual)
