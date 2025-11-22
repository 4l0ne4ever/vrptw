"""
Fitness evaluation for VRP solutions.
Implements multi-objective fitness function with penalty handling.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
import logging
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.data_processing.constraints import ConstraintHandler
from src.algorithms.decoder import RouteDecoder
from src.core.pipeline_profiler import pipeline_profiler
from config import GA_CONFIG, VRP_CONFIG
import os
import time
import json

logger = logging.getLogger(__name__)


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
        # Get dataset_type from problem for mode-specific behavior
        dataset_type = getattr(problem, 'dataset_type', None)
        self.constraint_handler = ConstraintHandler(
            problem.vehicle_capacity, 
            problem.num_vehicles,
            penalty_weight=self.penalty_weight,  # Pass penalty_weight to ConstraintHandler
            dataset_type=dataset_type  # Pass dataset_type for mode-specific behavior
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
            # OPTIMIZATION: Reuse decoded routes if available (avoid duplicate decode)
            # Routes are decoded after genetic operators for TW repair
            if hasattr(individual, 'routes') and individual.routes:
                routes = individual.routes
            else:
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
            
            # OPTIMIZATION: Skip TW repair in fitness if already repaired after genetic operators
            # This eliminates duplicate repair work (2x speedup)
            if not getattr(individual, '_tw_repaired', False):
                # Mode-specific TW repair: aggressive for Solomon, selective for Hanoi
                tw_repair_cfg = GA_CONFIG.get('tw_repair', {})
                dataset_type = getattr(self.problem, 'dataset_type', None)
                # Properly detect mode: check if it's solomon, otherwise default to hanoi
                if dataset_type is None:
                    # Try to infer from metadata
                    metadata = getattr(self.problem, 'metadata', {}) or {}
                    dataset_type = metadata.get('dataset_type', 'hanoi')
                dataset_type = str(dataset_type).strip().lower()
                if not dataset_type.startswith('solomon'):
                    dataset_type = 'hanoi'
                
                from src.data_processing.mode_configs import get_mode_config
                mode_config = get_mode_config(dataset_type)
                
                if (tw_repair_cfg.get('enabled', False) and 
                    hasattr(self.decoder, 'tw_repair') and 
                    self.decoder.tw_repair is not None):
                    try:
                        total_lateness_before = sum(
                            self.decoder.tw_repair._route_lateness(route) for route in routes
                        )
                        lateness_soft = tw_repair_cfg.get('lateness_soft_threshold', 5000.0)
                        lateness_skip = tw_repair_cfg.get('lateness_skip_threshold', 100000.0)
                        
                        # Solomon mode: Always repair if violations exist (aggressive)
                        # Hanoi mode: Selective repair (only moderate violations)
                        should_repair = False
                        if not mode_config.accept_near_feasible:
                            # Solomon: repair if any violations (below skip threshold)
                            should_repair = total_lateness_before > 0 and total_lateness_before < lateness_skip
                        else:
                            # Hanoi: repair only moderate violations
                            should_repair = lateness_soft < total_lateness_before < lateness_skip
                        
                        if should_repair:
                            try:
                                with pipeline_profiler.profile("fitness.selective_tw_repair"):
                                    routes_before = copy.deepcopy(routes)
                                    routes = self.decoder.tw_repair.repair_routes(routes)
                                    total_lateness_after = sum(
                                        self.decoder.tw_repair._route_lateness(route) for route in routes
                                    )
                                    
                                    # DEBUG: Log repair effectiveness
                                    import logging
                                    logger = logging.getLogger(__name__)
                                    logger.info(f"TW_REPAIR_FITNESS: before={total_lateness_before:.2f}, "
                                               f"after={total_lateness_after:.2f}, "
                                               f"improvement={total_lateness_before-total_lateness_after:.2f}, "
                                               f"num_routes={len(routes)}, mode={dataset_type}")
                                    
                                    individual.routes = routes
                                    individual._tw_repaired = True
                            except Exception as e:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning(f"TW repair failed in fitness: {e}")
                                pass
                        else:
                            # DEBUG: Log why repair was skipped
                            if total_lateness_before > 0:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.debug(f"TW_REPAIR_SKIP: lateness={total_lateness_before:.2f}, "
                                           f"soft_threshold={lateness_soft:.2f}, "
                                           f"skip_threshold={lateness_skip:.2f}, "
                                           f"should_repair={should_repair}, mode={dataset_type}")
                    except Exception as e:
                        logger.warning(f"TW repair check failed: {e}")
                        pass
            else:
                # Already repaired, skip duplicate work
                pass
            
            # Calculate total distance
            total_distance = self._calculate_total_distance(routes)
            individual.total_distance = total_distance
            
            # Calculate penalty for constraint violations
            raw_penalty = self._calculate_penalty(routes)
            
            # Calculate balance factor
            balance_factor = self._calculate_balance_factor(routes)
            
            # Mode-specific penalty handling
            dataset_type = getattr(self.problem, 'dataset_type', None)
            # Properly detect mode: check if it's solomon, otherwise default to hanoi
            if dataset_type is None:
                # Try to infer from metadata
                metadata = getattr(self.problem, 'metadata', {}) or {}
                dataset_type = metadata.get('dataset_type', 'hanoi')
            dataset_type = str(dataset_type).strip().lower()
            if not dataset_type.startswith('solomon'):
                dataset_type = 'hanoi'
            
            from src.data_processing.mode_configs import get_mode_config
            mode_config = get_mode_config(dataset_type)
            
            # DEBUG: Log mode detection
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Fitness calculation: dataset_type={dataset_type}, "
                        f"mode_config={type(mode_config).__name__}, "
                        f"accept_near_feasible={mode_config.accept_near_feasible}")
            
            if raw_penalty > 0:
                # Adaptive penalty cap based on mode and problem size
                # Loose time windows need higher caps (violations less severe)
                # Tight time windows need lower caps (violations more critical)
                # This prevents GA from being overwhelmed by huge penalty values
                # while still guiding toward feasible solutions
                
                if dataset_type and dataset_type.startswith('solomon'):
                    # Solomon mode: Adaptive cap based on TW tightness
                    num_customers = len(self.problem.customers)
                    
                    # Heuristic: Loose TW if vehicle_capacity >= 700
                    # C1/R1/RC1 use capacity 200, C2/R2/RC2 use 700/1000
                    is_loose_tw = self.problem.vehicle_capacity >= 700
                    
                    if is_loose_tw:
                        # Loose TW (C2/R2/RC2): much higher cap to allow GA to see penalty gradient
                        # Increased from 20× to 50× because violations can be very high (36000-65000)
                        # Penalty was hitting 100% cap, preventing GA from learning
                        max_penalty_cap = total_distance * 50.0
                    else:
                        # Tight TW (C1/R1/RC1): lower cap to enforce constraints
                        max_penalty_cap = total_distance * 2.0
                else:
                    # Hanoi mode: Higher cap to allow more flexibility
                    max_penalty_cap = total_distance * 10.0
                
                capped_penalty = min(raw_penalty, max_penalty_cap)
                
                # CRITICAL FIX: Penalties are ALREADY multiplied by penalty_weight (5000) in ConstraintHandler
                # DO NOT multiply again here! Penalty values are already HUGE (billions)
                # fitness = -(penalty + distance)
                
                # Debug logging for distance calculation
                logger.debug(f"FITNESS: raw_penalty={raw_penalty:.2e}, capped={capped_penalty:.2e}, "
                             f"distance={total_distance:.2f}, mode={dataset_type}, "
                             f"is_solomon={dataset_type.startswith('solomon') if dataset_type else False}, "
                             f"cap_ratio={max_penalty_cap/total_distance:.1f}x")
                
                fitness = -(capped_penalty + total_distance + balance_factor)
            else:
                # Feasible solutions: optimize distance only
                fitness = -(total_distance + balance_factor)
            
            individual.fitness = fitness
            
            # Store penalties for reporting and debugging
            if raw_penalty > 0:
                import logging
                logger = logging.getLogger(__name__)
                # Add route info to debug distance calculation
                routes = self._decode_chromosome(individual.chromosome)
                num_routes = len(routes)
                route_lengths = [len(r) for r in routes]
                total_customers = sum(len(r) - 2 for r in routes)  # Exclude depot visits
                logger.warning(f"PENALTY: raw={raw_penalty:.2e}, capped={capped_penalty:.2e}, "
                           f"distance={total_distance:.2f}, fitness={fitness:.2e}, mode={dataset_type}, "
                           f"num_routes={num_routes}, route_lengths={route_lengths[:5]}, "
                           f"total_customers={total_customers}")
            
            # Store raw_penalty for display (actual penalty before capping)
            # Store capped_penalty for fitness calculation tracking
            individual.penalty = raw_penalty if raw_penalty > 0 else 0.0
            individual.capped_penalty = capped_penalty if raw_penalty > 0 else 0.0
            
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
            
            # DEBUG: Log route count and structure
            if routes:
                import logging
                debug_logger = logging.getLogger(__name__)
                debug_logger.info(f"DISTANCE_START: num_routes={len(routes)}, "
                                 f"route_lengths={[len(r) for r in routes[:5]]}")
            
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
                    
                    # CORRECTNESS FIX: Check dataset type to determine if traffic factor should be applied
                    # Solomon datasets should use pure Euclidean distance (no traffic factor)
                    # Hanoi datasets use traffic factor for realistic routing
                    dataset_type = None
                    if hasattr(self.problem, 'metadata') and self.problem.metadata:
                        dataset_type = self.problem.metadata.get('dataset_type', 'hanoi')
                    elif hasattr(self.problem, 'dataset_type'):
                        dataset_type = self.problem.dataset_type
                    else:
                        dataset_type = 'hanoi'
                    
                    dataset_type = str(dataset_type).strip().lower()
                    is_solomon = dataset_type.startswith('solomon')
                    
                    # DEBUG: Log distance calculation for first few segments
                    if i == 0 and len(routes) > 0 and route == routes[0]:
                        import logging
                        debug_logger = logging.getLogger(__name__)
                        debug_logger.info(f"DISTANCE_CALC: from_id={from_id}, to_id={to_id}, "
                                         f"base_dist={base_dist:.4f}, is_solomon={is_solomon}, "
                                         f"use_adaptive_traffic={self.problem.use_adaptive_traffic}, "
                                         f"dataset_type={dataset_type}")
                    
                    # Apply adaptive traffic factor if enabled (Hanoi mode only)
                    if self.problem.use_adaptive_traffic and not is_solomon:
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
                        # For Solomon datasets: use pure Euclidean distance (no traffic factor)
                        # For Hanoi non-adaptive: use base distance with fixed traffic factor
                        if is_solomon:
                            # Solomon: pure Euclidean, no multiplier
                            route_distance += base_dist
                            # DEBUG: Verify no traffic factor is applied
                            if i == 0 and len(routes) > 0 and route == routes[0]:
                                import logging
                                debug_logger = logging.getLogger(__name__)
                                debug_logger.info(f"SOLOMON_DIST: base_dist={base_dist:.4f}, "
                                                 f"route_distance={route_distance:.4f}, "
                                                 f"NO traffic factor applied")
                        else:
                            # Hanoi non-adaptive: apply traffic factor
                            traffic_factor = VRP_CONFIG.get('traffic_factor', 1.0)
                            route_distance += base_dist * traffic_factor
                
                total_distance += route_distance
                
                # DEBUG: Log route distance for first few routes to understand structure
                route_idx = routes.index(route) if route in routes else -1
                if route_idx < 3:  # Log first 3 routes
                    import logging
                    debug_logger = logging.getLogger(__name__)
                    # Calculate manual distance for verification
                    manual_dist = 0.0
                    for i in range(len(route) - 1):
                        manual_dist += self.problem.get_distance(route[i], route[i + 1])
                    debug_logger.warning(f"ROUTE_DIST[{route_idx}]: route={route[:8]}{'...' if len(route) > 8 else ''}{route[-3:]}, "
                                     f"len={len(route)}, customers={len(route)-2}, "
                                     f"route_distance={route_distance:.2f}, manual={manual_dist:.2f}, "
                                     f"total_distance={total_distance:.2f}, num_routes={len(routes)}")
            
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
        Check if solution is feasible (mode-specific acceptance criteria).
        
        Args:
            individual: Individual to check
            
        Returns:
            True if acceptable according to mode-specific criteria, False otherwise
        """
        if individual.is_empty():
            return False
        
        routes = self._decode_chromosome(individual.chromosome)
        penalty = self._calculate_penalty(routes)
        
        # Mode-specific acceptance criteria
        dataset_type = getattr(self.problem, 'dataset_type', None)
        # Properly detect mode: check if it's solomon, otherwise default to hanoi
        if dataset_type is None:
            # Try to infer from metadata
            metadata = getattr(self.problem, 'metadata', {}) or {}
            dataset_type = metadata.get('dataset_type', 'hanoi')
        dataset_type = str(dataset_type).strip().lower()
        if not dataset_type.startswith('solomon'):
            dataset_type = 'hanoi'
        
        from src.data_processing.mode_configs import get_mode_config
        mode_config = get_mode_config(dataset_type)
        
        if not mode_config.accept_near_feasible:
            # Strict mode (Solomon): must be perfectly feasible
            return penalty == 0.0
        
        # Relaxed mode (Hanoi): check if acceptable
        return self._is_acceptable_solution(routes, mode_config)
    
    def _is_acceptable_solution(self, routes: List[List[int]], mode_config) -> bool:
        """
        Check if solution is acceptable according to mode-specific criteria.
        
        Args:
            routes: List of routes
            mode_config: Mode configuration (HanoiConfig or SolomonConfig)
            
        Returns:
            True if acceptable, False otherwise
        """
        # Check capacity violations (must be 0)
        demands = self.problem.get_demands()
        cap_valid, cap_penalty = self.constraint_handler.validate_capacity_constraint(routes, demands)
        if not cap_valid:
            return False
        
        # Count time window violations
        total_customers = len(self.problem.customers)
        violations = self._count_time_window_violations(routes)
        hard_violations = self._count_hard_time_window_violations(routes)
        
        # Check acceptance criteria
        if violations > mode_config.max_acceptable_violations:
            return False
        
        if hard_violations > mode_config.max_acceptable_hard_violations:
            return False
        
        # Check compliance rate
        compliance_rate = (total_customers - violations) / total_customers if total_customers > 0 else 0.0
        if compliance_rate < mode_config.feasibility_threshold:
            return False
        
        return True
    
    def _count_time_window_violations(self, routes: List[List[int]]) -> int:
        """
        Count total time window violations by simulating routes.
        
        Args:
            routes: List of routes
            
        Returns:
            Number of violations
        """
        violations = 0
        from config import VRP_CONFIG
        time_window_start = VRP_CONFIG.get('time_window_start', 480)
        
        for route in routes:
            if len(route) < 2:
                continue
            
            current_time = float(time_window_start)
            
            for i in range(len(route) - 1):
                to_customer_id = route[i + 1]
                if to_customer_id == 0:  # Skip depot
                    continue
                
                customer = self.problem.get_customer_by_id(to_customer_id)
                if customer is None:
                    continue
                
                # Travel time
                from_id = route[i]
                travel_time = self.problem.get_distance(from_id, to_customer_id)
                arrival_time = current_time + travel_time
                
                # Wait if early
                if arrival_time < customer.ready_time:
                    arrival_time = customer.ready_time
                
                # Check violation
                if arrival_time > customer.due_date:
                    violations += 1
                
                # Service time
                current_time = arrival_time + customer.service_time
        
        return violations
    
    def _count_hard_time_window_violations(self, routes: List[List[int]]) -> int:
        """
        Count hard time window violations (outside buffer for Hanoi mode).
        
        Args:
            routes: List of routes
            
        Returns:
            Number of hard violations
        """
        dataset_type = getattr(self.problem, 'dataset_type', None)
        # Properly detect mode: check if it's solomon, otherwise default to hanoi
        if dataset_type is None:
            # Try to infer from metadata
            metadata = getattr(self.problem, 'metadata', {}) or {}
            dataset_type = metadata.get('dataset_type', 'hanoi')
        dataset_type = str(dataset_type).strip().lower()
        if not dataset_type.startswith('solomon'):
            dataset_type = 'hanoi'
        
        from src.data_processing.mode_configs import get_mode_config
        mode_config = get_mode_config(dataset_type)
        
        if not mode_config.allow_time_flexibility:
            # Strict mode: all violations are hard
            return self._count_time_window_violations(routes)
        
        # Soft mode: count violations outside buffer
        hard_violations = 0
        from config import VRP_CONFIG
        time_window_start = VRP_CONFIG.get('time_window_start', 480)
        buffer = mode_config.time_window_buffer
        
        for route in routes:
            if len(route) < 2:
                continue
            
            current_time = float(time_window_start)
            
            for i in range(len(route) - 1):
                to_customer_id = route[i + 1]
                if to_customer_id == 0:  # Skip depot
                    continue
                
                customer = self.problem.get_customer_by_id(to_customer_id)
                if customer is None:
                    continue
                
                # Travel time
                from_id = route[i]
                travel_time = self.problem.get_distance(from_id, to_customer_id)
                arrival_time = current_time + travel_time
                
                # Check hard violation (outside buffer)
                early_buffer = customer.ready_time - buffer
                late_buffer = customer.due_date + buffer
                
                if arrival_time < early_buffer or arrival_time > late_buffer:
                    hard_violations += 1
                
                # Update time
                if arrival_time < customer.ready_time:
                    arrival_time = customer.ready_time
                current_time = arrival_time + customer.service_time
        
        return hard_violations
    
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
