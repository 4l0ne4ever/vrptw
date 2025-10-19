"""
Fitness evaluation for VRP solutions.
Implements multi-objective fitness function with penalty handling.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.data_processing.constraints import ConstraintHandler


class FitnessEvaluator:
    """Evaluates fitness of VRP solutions."""
    
    def __init__(self, problem: VRPProblem, penalty_weight: float = 1000):
        """
        Initialize fitness evaluator.
        
        Args:
            problem: VRP problem instance
            penalty_weight: Weight for constraint violations
        """
        self.problem = problem
        self.penalty_weight = penalty_weight
        self.constraint_handler = ConstraintHandler(
            problem.vehicle_capacity, 
            problem.num_vehicles
        )
    
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
        
        # Decode chromosome to routes
        routes = self._decode_chromosome(individual.chromosome)
        individual.routes = routes
        
        # Calculate total distance
        total_distance = self._calculate_total_distance(routes)
        individual.total_distance = total_distance
        
        # Calculate penalty for constraint violations
        penalty = self._calculate_penalty(routes)
        individual.penalty = penalty
        
        # Calculate balance factor
        balance_factor = self._calculate_balance_factor(routes)
        
        # Calculate fitness (higher is better)
        fitness = 1.0 / (total_distance + penalty + balance_factor + 1.0)
        individual.fitness = fitness
        
        # Mark as valid if no penalty
        individual.is_valid = penalty == 0.0
        
        return fitness
    
    def _decode_chromosome(self, chromosome: List[int]) -> List[List[int]]:
        """
        Decode chromosome to routes using capacity constraint.
        
        Args:
            chromosome: Customer order chromosome
            
        Returns:
            List of routes (each route includes depot at start and end)
        """
        if not chromosome:
            return []
        
        routes = []
        current_route = [0]  # Start at depot
        current_load = 0.0
        
        for customer_id in chromosome:
            customer_demand = self.problem.get_customer_by_id(customer_id).demand
            
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
            
            route_distance = 0.0
            for i in range(len(route) - 1):
                from_idx = route[i]
                to_idx = route[i + 1]
                route_distance += self.problem.get_distance(from_idx, to_idx)
            
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
        demands = self.problem.get_demands()
        num_customers = len(self.problem.customers)
        
        # Validate constraints
        validation_results = self.constraint_handler.validate_all_constraints(
            routes, demands, num_customers
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
