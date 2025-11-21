"""
KPI metrics calculation for VRP solutions.
Implements comprehensive performance evaluation.
"""

import time
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.evaluation.shipping_cost import ShippingCostCalculator

logger = logging.getLogger(__name__)


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
        
        # Add shipping cost calculation
        shipping_cost_data = self._calculate_shipping_cost(individual)
        kpis.update(shipping_cost_data)
        
        return kpis
    
    def _calculate_shipping_cost(self, individual: Individual) -> Dict:
        """Calculate shipping cost for the solution."""
        try:
            # Initialize shipping cost calculator
            from config import VRP_CONFIG
            shipping_calculator = ShippingCostCalculator(
                cost_model="ahamove",
                use_waiting_fee=bool(VRP_CONFIG.get('use_waiting_fee', False)),
                cod_fee_rate_override=VRP_CONFIG.get('cod_fee_rate', None)
            )
            
            # Decode routes - use Split Algorithm if enabled to match GA behavior
            from src.algorithms.decoder import RouteDecoder
            from config import GA_CONFIG
            decoder = RouteDecoder(self.problem, use_split_algorithm=GA_CONFIG.get('use_split_algorithm', False))
            # Use routes from individual if available, otherwise decode from chromosome
            if individual.routes:
                routes = individual.routes
            else:
                routes = decoder.decode_chromosome(individual.chromosome)
            
            # Generate order values and waiting times
            order_values = shipping_calculator.generate_order_values(self.problem.customers)
            waiting_times = shipping_calculator.generate_waiting_times(self.problem.customers)
            
            # Calculate shipping cost with operational costs
            shipping_cost_data = shipping_calculator.calculate_solution_cost(
                routes, self.problem, service_type="express", 
                order_values=order_values, waiting_times=waiting_times,
                include_operational_costs=True
            )
            
            # Extract cost breakdown
            cost_breakdown = shipping_cost_data.get('cost_breakdown', {})
            operational_costs = shipping_cost_data.get('operational_costs', {})
            
            return {
                # Total costs
                'shipping_total_cost': shipping_cost_data.get('shipping_cost', 0),
                'total_cost': shipping_cost_data.get('total_cost', 0),
                
                # Operational costs breakdown
                'fuel_cost': operational_costs.get('fuel_cost', 0),
                'driver_cost': operational_costs.get('driver_cost', 0),
                'vehicle_fixed_cost': operational_costs.get('vehicle_fixed_cost', 0),
                'total_operational_cost': operational_costs.get('total_operational_cost', 0),
                
                # Per-unit costs
                'shipping_cost_per_km': shipping_cost_data.get('shipping_cost', 0) / max(individual.total_distance, 1),
                'total_cost_per_km': shipping_cost_data.get('total_cost', 0) / max(individual.total_distance, 1),
                'shipping_cost_per_customer': shipping_cost_data.get('shipping_cost', 0) / max(individual.get_customer_count(), 1),
                'total_cost_per_customer': shipping_cost_data.get('total_cost', 0) / max(individual.get_customer_count(), 1),
                'shipping_cost_per_route': shipping_cost_data.get('shipping_cost', 0) / max(individual.get_route_count(), 1),
                'total_cost_per_route': shipping_cost_data.get('total_cost', 0) / max(individual.get_route_count(), 1),
                
                # Additional metrics
                'total_distance': shipping_cost_data.get('total_distance', individual.total_distance),
                'total_duration_hours': shipping_cost_data.get('total_duration_hours', 0),
                'num_routes': shipping_cost_data.get('num_routes', individual.get_route_count()),
                'shipping_service_type': shipping_cost_data.get('service_type', 'express'),
                'shipping_cost_model': shipping_cost_data.get('cost_model', 'ahamove'),
                'shipping_route_costs': shipping_cost_data.get('route_costs', []),
                'cost_breakdown': cost_breakdown
            }
            
        except Exception as e:
            # Return default values if calculation fails
            return {
                'shipping_total_cost': 0.0,
                'shipping_cost_per_km': 0.0,
                'shipping_cost_per_customer': 0.0,
                'shipping_cost_per_route': 0.0,
                'shipping_service_type': 'unknown',
                'shipping_cost_model': 'unknown',
                'shipping_route_costs': []
            }
    
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
                route_load = 0.0
                for c in route:
                    if c == 0:
                        continue
                    customer = self.problem.get_customer_by_id(c)
                    if customer is None:
                        logger.warning(
                            "Customer ID %s not found in problem definition. Skipping in load calculation.",
                            c
                        )
                        continue
                    route_load += customer.demand
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
        
        dataset_type = getattr(self.problem, 'dataset_type', None)
        constraint_handler = ConstraintHandler(
            self.problem.vehicle_capacity, 
            self.problem.num_vehicles,
            dataset_type=dataset_type
        )
        
        demands = self.problem.get_demands()
        num_customers = len(self.problem.customers)
        
        validation_results = constraint_handler.validate_all_constraints(
            individual.routes, demands, self.problem.customers
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
        """
        Calculate comprehensive cost-related metrics including operational costs.
        
        Args:
            total_distance: Total distance traveled
            num_routes: Number of routes
            
        Returns:
            Dictionary with cost metrics
        """
        from config import VRP_CONFIG
        
        # Base cost per km (for simple distance-based cost)
        cost_per_km = VRP_CONFIG.get('cost_per_km', 1.0)
        base_cost = total_distance * cost_per_km
        
        # Operational costs
        fuel_cost_per_km = VRP_CONFIG.get('fuel_cost_per_km', 4000)
        fuel_cost = total_distance * fuel_cost_per_km
        
        # Estimate driver cost (assume average speed 30 km/h)
        avg_speed_kmh = 30.0
        total_duration_hours = total_distance / avg_speed_kmh
        driver_cost_per_hour = VRP_CONFIG.get('driver_cost_per_hour', 40000)
        driver_cost = total_duration_hours * driver_cost_per_hour
        
        # Vehicle fixed cost
        vehicle_fixed_cost = VRP_CONFIG.get('vehicle_fixed_cost', 75000)
        vehicle_fixed_total = num_routes * vehicle_fixed_cost
        
        # Total operational cost
        total_operational_cost = fuel_cost + driver_cost + vehicle_fixed_total
        
        # Total cost (base + operational)
        total_cost = base_cost + total_operational_cost
        
        num_customers = len(self.problem.customers)
        
        return {
            'base_cost': base_cost,
            'fuel_cost': fuel_cost,
            'driver_cost': driver_cost,
            'vehicle_fixed_cost': vehicle_fixed_total,
            'total_operational_cost': total_operational_cost,
            'total_cost': total_cost,
            'cost_per_km': cost_per_km,
            'cost_per_customer': total_cost / num_customers if num_customers > 0 else 0,
            'cost_per_route': total_cost / num_routes if num_routes > 0 else 0,
            'fuel_cost_per_km': fuel_cost_per_km,
            'driver_cost_per_hour': driver_cost_per_hour,
            'vehicle_fixed_cost_per_vehicle': vehicle_fixed_cost,
            'total_duration_hours': total_duration_hours
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
