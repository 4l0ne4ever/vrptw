"""
KPI metrics calculation for VRP solutions.
Implements comprehensive performance evaluation.
"""

import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem


class KPICalculator:
    """Calculates Key Performance Indicators for VRP solutions."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize KPI calculator.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
    
    def calculate_kpis(self, individual: Individual, 
                      execution_time: Optional[float] = None) -> Dict:
        """
        Calculate comprehensive KPIs for a solution.
        
        Args:
            individual: Solution individual
            execution_time: Execution time in seconds
            
        Returns:
            Dictionary with KPI values
        """
        if individual.is_empty():
            return self._get_empty_kpis()
        
        # Basic metrics
        total_distance = individual.total_distance
        num_routes = individual.get_route_count()
        num_customers = individual.get_customer_count()
        
        # Route statistics
        route_stats = self._calculate_route_statistics(individual.routes)
        
        # Load balance metrics
        load_balance = self._calculate_load_balance(individual.routes)
        
        # Constraint violations
        constraint_violations = self._calculate_constraint_violations(individual)
        
        # Cost metrics
        cost_metrics = self._calculate_cost_metrics(total_distance, num_routes)
        
        # Efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            individual, route_stats, load_balance
        )
        
        # Compile all KPIs
        kpis = {
            # Basic metrics
            'total_distance': total_distance,
            'num_routes': num_routes,
            'num_customers': num_customers,
            'num_vehicles_used': num_routes,
            'execution_time': execution_time or 0.0,
            
            # Route metrics
            'avg_route_length': route_stats['avg_route_length'],
            'max_route_length': route_stats['max_route_length'],
            'min_route_length': route_stats['min_route_length'],
            'route_length_std': route_stats['route_length_std'],
            
            # Load metrics
            'avg_utilization': load_balance['avg_utilization'],
            'max_utilization': load_balance['max_utilization'],
            'min_utilization': load_balance['min_utilization'],
            'utilization_std': load_balance['utilization_std'],
            'load_balance_index': load_balance['balance_index'],
            
            # Cost metrics
            'total_cost': cost_metrics['total_cost'],
            'cost_per_km': cost_metrics['cost_per_km'],
            'cost_per_customer': cost_metrics['cost_per_customer'],
            'cost_per_route': cost_metrics['cost_per_route'],
            
            # Efficiency metrics
            'solution_quality': efficiency_metrics['solution_quality'],
            'efficiency_score': efficiency_metrics['efficiency_score'],
            'feasibility_score': efficiency_metrics['feasibility_score'],
            
            # Constraint violations
            'constraint_violations': constraint_violations,
            'is_feasible': constraint_violations['total_violations'] == 0,
            
            # Fitness
            'fitness': individual.fitness,
            'penalty': individual.penalty
        }
        
        return kpis
    
    def _get_empty_kpis(self) -> Dict:
        """Get KPIs for empty solution."""
        return {
            'total_distance': 0.0,
            'num_routes': 0,
            'num_customers': 0,
            'num_vehicles_used': 0,
            'execution_time': 0.0,
            'avg_route_length': 0.0,
            'max_route_length': 0,
            'min_route_length': 0,
            'route_length_std': 0.0,
            'avg_utilization': 0.0,
            'max_utilization': 0.0,
            'min_utilization': 0.0,
            'utilization_std': 0.0,
            'load_balance_index': 0.0,
            'total_cost': 0.0,
            'cost_per_km': 0.0,
            'cost_per_customer': 0.0,
            'cost_per_route': 0.0,
            'solution_quality': 0.0,
            'efficiency_score': 0.0,
            'feasibility_score': 0.0,
            'constraint_violations': {'total_violations': 0},
            'is_feasible': True,
            'fitness': 0.0,
            'penalty': 0.0
        }
    
    def _calculate_route_statistics(self, routes: List[List[int]]) -> Dict:
        """Calculate route length statistics."""
        if not routes:
            return {
                'avg_route_length': 0.0,
                'max_route_length': 0,
                'min_route_length': 0,
                'route_length_std': 0.0
            }
        
        route_lengths = []
        for route in routes:
            if route:
                # Count customers (exclude depot visits)
                customer_count = len([c for c in route if c != 0])
                route_lengths.append(customer_count)
        
        if not route_lengths:
            return {
                'avg_route_length': 0.0,
                'max_route_length': 0,
                'min_route_length': 0,
                'route_length_std': 0.0
            }
        
        return {
            'avg_route_length': np.mean(route_lengths),
            'max_route_length': max(route_lengths),
            'min_route_length': min(route_lengths),
            'route_length_std': np.std(route_lengths)
        }
    
    def _calculate_load_balance(self, routes: List[List[int]]) -> Dict:
        """Calculate load balance metrics."""
        if not routes:
            return {
                'avg_utilization': 0.0,
                'max_utilization': 0.0,
                'min_utilization': 0.0,
                'utilization_std': 0.0,
                'balance_index': 0.0
            }
        
        route_loads = []
        for route in routes:
            if route:
                route_load = sum(
                    self.problem.get_customer_by_id(c).demand 
                    for c in route if c != 0
                )
                route_loads.append(route_load)
        
        if not route_loads:
            return {
                'avg_utilization': 0.0,
                'max_utilization': 0.0,
                'min_utilization': 0.0,
                'utilization_std': 0.0,
                'balance_index': 0.0
            }
        
        utilizations = [(load / self.problem.vehicle_capacity) * 100 for load in route_loads]
        
        # Balance index: 1 - (std_deviation / mean) (higher is better)
        balance_index = 1 - (np.std(utilizations) / np.mean(utilizations)) if np.mean(utilizations) > 0 else 0
        
        return {
            'avg_utilization': np.mean(utilizations),
            'max_utilization': max(utilizations),
            'min_utilization': min(utilizations),
            'utilization_std': np.std(utilizations),
            'balance_index': max(0, balance_index)
        }
    
    def _calculate_constraint_violations(self, individual: Individual) -> Dict:
        """Calculate constraint violations."""
        from src.data_processing.constraints import ConstraintHandler
        
        constraint_handler = ConstraintHandler(
            self.problem.vehicle_capacity, 
            self.problem.num_vehicles
        )
        
        demands = self.problem.get_demands()
        num_customers = len(self.problem.customers)
        
        validation_results = constraint_handler.validate_all_constraints(
            individual.routes, demands, num_customers
        )
        
        return {
            'total_violations': validation_results['total_penalty'],
            'capacity_violations': validation_results['violations'].get('capacity', False),
            'vehicle_count_violations': validation_results['violations'].get('vehicle_count', False),
            'customer_visit_violations': validation_results['violations'].get('customer_visit', False),
            'depot_violations': validation_results['violations'].get('depot', False),
            'time_window_violations': validation_results['violations'].get('time_windows', False)
        }
    
    def _calculate_cost_metrics(self, total_distance: float, num_routes: int) -> Dict:
        """Calculate cost-related metrics."""
        cost_per_km = 1.0  # Base cost per km
        total_cost = total_distance * cost_per_km
        
        num_customers = len(self.problem.customers)
        
        return {
            'total_cost': total_cost,
            'cost_per_km': cost_per_km,
            'cost_per_customer': total_cost / num_customers if num_customers > 0 else 0,
            'cost_per_route': total_cost / num_routes if num_routes > 0 else 0
        }
    
    def _calculate_efficiency_metrics(self, individual: Individual, 
                                    route_stats: Dict, load_balance: Dict) -> Dict:
        """Calculate efficiency metrics."""
        # Solution quality based on distance and feasibility
        distance_score = 1.0 / (individual.total_distance + 1.0)
        feasibility_score = 1.0 if individual.is_valid else 0.0
        
        # Efficiency score combining multiple factors
        efficiency_score = (
            distance_score * 0.4 +
            feasibility_score * 0.3 +
            load_balance['balance_index'] * 0.2 +
            (1.0 / (route_stats['route_length_std'] + 1.0)) * 0.1
        )
        
        return {
            'solution_quality': distance_score,
            'efficiency_score': efficiency_score,
            'feasibility_score': feasibility_score
        }
    
    def compare_solutions(self, solution1: Individual, solution2: Individual,
                        name1: str = "Solution 1", name2: str = "Solution 2") -> Dict:
        """
        Compare two solutions and calculate improvement metrics.
        
        Args:
            solution1: First solution
            solution2: Second solution
            name1: Name for first solution
            name2: Name for second solution
            
        Returns:
            Comparison results
        """
        kpis1 = self.calculate_kpis(solution1)
        kpis2 = self.calculate_kpis(solution2)
        
        # Calculate improvements
        distance_improvement = kpis1['total_distance'] - kpis2['total_distance']
        distance_improvement_percent = (
            (distance_improvement / kpis1['total_distance']) * 100 
            if kpis1['total_distance'] > 0 else 0
        )
        
        cost_improvement = kpis1['total_cost'] - kpis2['total_cost']
        cost_improvement_percent = (
            (cost_improvement / kpis1['total_cost']) * 100 
            if kpis1['total_cost'] > 0 else 0
        )
        
        efficiency_improvement = kpis2['efficiency_score'] - kpis1['efficiency_score']
        efficiency_improvement_percent = (
            (efficiency_improvement / kpis1['efficiency_score']) * 100 
            if kpis1['efficiency_score'] > 0 else 0
        )
        
        return {
            'solution1': {
                'name': name1,
                'kpis': kpis1
            },
            'solution2': {
                'name': name2,
                'kpis': kpis2
            },
            'improvements': {
                'distance_improvement': distance_improvement,
                'distance_improvement_percent': distance_improvement_percent,
                'cost_improvement': cost_improvement,
                'cost_improvement_percent': cost_improvement_percent,
                'efficiency_improvement': efficiency_improvement,
                'efficiency_improvement_percent': efficiency_improvement_percent,
                'is_improved': distance_improvement > 0
            }
        }
    
    def get_summary_statistics(self, solutions: List[Individual]) -> Dict:
        """
        Get summary statistics for multiple solutions.
        
        Args:
            solutions: List of solutions
            
        Returns:
            Summary statistics
        """
        if not solutions:
            return {}
        
        kpis_list = [self.calculate_kpis(sol) for sol in solutions]
        
        # Extract metrics
        distances = [kpi['total_distance'] for kpi in kpis_list]
        costs = [kpi['total_cost'] for kpi in kpis_list]
        efficiencies = [kpi['efficiency_score'] for kpi in kpis_list]
        execution_times = [kpi['execution_time'] for kpi in kpis_list]
        
        return {
            'num_solutions': len(solutions),
            'distance_stats': {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': min(distances),
                'max': max(distances),
                'median': np.median(distances)
            },
            'cost_stats': {
                'mean': np.mean(costs),
                'std': np.std(costs),
                'min': min(costs),
                'max': max(costs),
                'median': np.median(costs)
            },
            'efficiency_stats': {
                'mean': np.mean(efficiencies),
                'std': np.std(efficiencies),
                'min': min(efficiencies),
                'max': max(efficiencies),
                'median': np.median(efficiencies)
            },
            'execution_time_stats': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'median': np.median(execution_times)
            }
        }


def calculate_kpis(individual: Individual, problem: VRPProblem, 
                  execution_time: Optional[float] = None) -> Dict:
    """
    Convenience function to calculate KPIs.
    
    Args:
        individual: Solution individual
        problem: VRP problem instance
        execution_time: Execution time in seconds
        
    Returns:
        Dictionary with KPI values
    """
    calculator = KPICalculator(problem)
    return calculator.calculate_kpis(individual, execution_time)
