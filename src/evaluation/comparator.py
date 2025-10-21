"""
Solution comparison and validation for VRP.
Implements comprehensive solution analysis.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.evaluation.metrics import KPICalculator


class SolutionComparator:
    """Compares and analyzes VRP solutions."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize solution comparator.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
        self.kpi_calculator = KPICalculator(problem)
    
    def compare_methods(self, ga_solution: Individual, nn_solution: Individual) -> Dict:
        """
        Compare GA solution with Nearest Neighbor baseline.
        
        Args:
            ga_solution: GA solution
            nn_solution: Nearest Neighbor solution
            
        Returns:
            Comparison results
        """
        comparison = self.kpi_calculator.compare_solutions(
            ga_solution, nn_solution, "GA", "Nearest Neighbor"
        )
        
        # Add method-specific analysis
        comparison['method_analysis'] = self._analyze_methods(ga_solution, nn_solution)
        
        return comparison
    
    def _analyze_methods(self, ga_solution: Individual, nn_solution: Individual) -> Dict:
        """Analyze characteristics of different methods."""
        return {
            'ga_characteristics': self._analyze_solution_characteristics(ga_solution),
            'nn_characteristics': self._analyze_solution_characteristics(nn_solution),
            'method_differences': self._calculate_method_differences(ga_solution, nn_solution)
        }
    
    def _analyze_solution_characteristics(self, solution: Individual) -> Dict:
        """Analyze characteristics of a solution."""
        if solution.is_empty():
            return {}
        
        routes = solution.routes
        if not routes:
            return {}
        
        # Route analysis
        route_lengths = [len([c for c in route if c != 0]) for route in routes if route]
        route_distances = []
        
        for route in routes:
            if route:
                route_distance = 0.0
                for i in range(len(route) - 1):
                    route_distance += self.problem.get_distance(route[i], route[i + 1])
                route_distances.append(route_distance)
        
        # Load analysis
        route_loads = []
        for route in routes:
            if route:
                route_load = sum(
                    self.problem.get_customer_by_id(c).demand 
                    for c in route if c != 0
                )
                route_loads.append(route_load)
        
        return {
            'num_routes': len([r for r in routes if r]),
            'avg_route_length': np.mean(route_lengths) if route_lengths else 0,
            'route_length_variance': np.var(route_lengths) if route_lengths else 0,
            'avg_route_distance': np.mean(route_distances) if route_distances else 0,
            'route_distance_variance': np.var(route_distances) if route_distances else 0,
            'avg_route_load': np.mean(route_loads) if route_loads else 0,
            'load_variance': np.var(route_loads) if route_loads else 0,
            'solution_complexity': len(solution.chromosome),
            'is_feasible': solution.is_valid
        }
    
    def _calculate_method_differences(self, ga_solution: Individual, nn_solution: Individual) -> Dict:
        """Calculate differences between methods."""
        ga_chars = self._analyze_solution_characteristics(ga_solution)
        nn_chars = self._analyze_solution_characteristics(nn_solution)
        
        if not ga_chars or not nn_chars:
            return {}
        
        return {
            'route_count_difference': ga_chars['num_routes'] - nn_chars['num_routes'],
            'avg_route_length_difference': ga_chars['avg_route_length'] - nn_chars['avg_route_length'],
            'load_balance_difference': ga_chars['load_variance'] - nn_chars['load_variance'],
            'complexity_difference': ga_chars['solution_complexity'] - nn_chars['solution_complexity']
        }
    
    def rank_solutions(self, solutions: List[Individual], 
                      weights: Optional[Dict[str, float]] = None) -> List[Tuple[Individual, float]]:
        """
        Rank solutions based on multiple criteria.
        
        Args:
            solutions: List of solutions to rank
            weights: Weights for different criteria
            
        Returns:
            List of (solution, score) tuples sorted by score
        """
        if not solutions:
            return []
        
        # Default weights
        default_weights = {
            'distance': 0.4,
            'feasibility': 0.3,
            'load_balance': 0.2,
            'efficiency': 0.1
        }
        weights = weights or default_weights
        
        scored_solutions = []
        
        for solution in solutions:
            kpis = self.kpi_calculator.calculate_kpis(solution)
            
            # Calculate composite score
            score = (
                weights['distance'] * (1.0 / (kpis['total_distance'] + 1.0)) +
                weights['feasibility'] * (1.0 if kpis['is_feasible'] else 0.0) +
                weights['load_balance'] * kpis['load_balance_index'] +
                weights['efficiency'] * kpis['efficiency_score']
            )
            
            scored_solutions.append((solution, score))
        
        # Sort by score (descending)
        scored_solutions.sort(key=lambda x: x[1], reverse=True)
        
        return scored_solutions
    
    def find_pareto_front(self, solutions: List[Individual]) -> List[Individual]:
        """
        Find Pareto-optimal solutions.
        
        Args:
            solutions: List of solutions
            
        Returns:
            List of Pareto-optimal solutions
        """
        if not solutions:
            return []
        
        pareto_solutions = []
        
        for i, solution1 in enumerate(solutions):
            is_dominated = False
            
            for j, solution2 in enumerate(solutions):
                if i == j:
                    continue
                
                if self._dominates(solution2, solution1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(solution1)
        
        return pareto_solutions
    
    def _dominates(self, solution1: Individual, solution2: Individual) -> bool:
        """
        Check if solution1 dominates solution2.
        
        Args:
            solution1: First solution
            solution2: Second solution
            
        Returns:
            True if solution1 dominates solution2
        """
        kpis1 = self.kpi_calculator.calculate_kpis(solution1)
        kpis2 = self.kpi_calculator.calculate_kpis(solution2)
        
        # Solution1 dominates solution2 if:
        # 1. It has better or equal distance
        # 2. It has better or equal feasibility
        # 3. It has better or equal load balance
        # 4. At least one criterion is strictly better
        
        distance_better = kpis1['total_distance'] <= kpis2['total_distance']
        feasibility_better = kpis1['is_feasible'] >= kpis2['is_feasible']
        balance_better = kpis1['load_balance_index'] >= kpis2['load_balance_index']
        
        distance_strictly_better = kpis1['total_distance'] < kpis2['total_distance']
        feasibility_strictly_better = kpis1['is_feasible'] and not kpis2['is_feasible']
        balance_strictly_better = kpis1['load_balance_index'] > kpis2['load_balance_index']
        
        return (distance_better and feasibility_better and balance_better and
                (distance_strictly_better or feasibility_strictly_better or balance_strictly_better))
    
    def calculate_improvement_metrics(self, baseline: Individual, 
                                   improved: Individual) -> Dict:
        """
        Calculate improvement metrics between solutions.
        
        Args:
            baseline: Baseline solution
            improved: Improved solution
            
        Returns:
            Improvement metrics
        """
        baseline_kpis = self.kpi_calculator.calculate_kpis(baseline)
        improved_kpis = self.kpi_calculator.calculate_kpis(improved)
        
        improvements = {}
        
        # Distance improvement
        distance_improvement = baseline_kpis['total_distance'] - improved_kpis['total_distance']
        distance_improvement_percent = (
            (distance_improvement / baseline_kpis['total_distance']) * 100 
            if baseline_kpis['total_distance'] > 0 else 0
        )
        
        improvements['distance'] = {
            'absolute': distance_improvement,
            'percentage': distance_improvement_percent,
            'is_improved': distance_improvement > 0
        }
        
        # Cost improvement
        cost_improvement = baseline_kpis['total_cost'] - improved_kpis['total_cost']
        cost_improvement_percent = (
            (cost_improvement / baseline_kpis['total_cost']) * 100 
            if baseline_kpis['total_cost'] > 0 else 0
        )
        
        improvements['cost'] = {
            'absolute': cost_improvement,
            'percentage': cost_improvement_percent,
            'is_improved': cost_improvement > 0
        }
        
        # Efficiency improvement
        efficiency_improvement = improved_kpis['efficiency_score'] - baseline_kpis['efficiency_score']
        efficiency_improvement_percent = (
            (efficiency_improvement / baseline_kpis['efficiency_score']) * 100 
            if baseline_kpis['efficiency_score'] > 0 else 0
        )
        
        improvements['efficiency'] = {
            'absolute': efficiency_improvement,
            'percentage': efficiency_improvement_percent,
            'is_improved': efficiency_improvement > 0
        }
        
        # Load balance improvement
        balance_improvement = improved_kpis['load_balance_index'] - baseline_kpis['load_balance_index']
        balance_improvement_percent = (
            (balance_improvement / baseline_kpis['load_balance_index']) * 100 
            if baseline_kpis['load_balance_index'] > 0 else 0
        )
        
        improvements['load_balance'] = {
            'absolute': balance_improvement,
            'percentage': balance_improvement_percent,
            'is_improved': balance_improvement > 0
        }
        
        # Overall improvement score
        overall_score = (
            improvements['distance']['is_improved'] * 0.4 +
            improvements['cost']['is_improved'] * 0.3 +
            improvements['efficiency']['is_improved'] * 0.2 +
            improvements['load_balance']['is_improved'] * 0.1
        )
        
        improvements['overall'] = {
            'score': overall_score,
            'is_improved': overall_score > 0.5
        }
        
        return improvements
    
    def generate_comparison_report(self, solutions: List[Individual], 
                                 solution_names: Optional[List[str]] = None) -> Dict:
        """
        Generate comprehensive comparison report.
        
        Args:
            solutions: List of solutions to compare
            solution_names: Names for solutions
            
        Returns:
            Comparison report
        """
        if not solutions:
            return {}
        
        if solution_names is None:
            solution_names = [f"Solution {i+1}" for i in range(len(solutions))]
        
        # Calculate KPIs for all solutions
        solution_kpis = []
        for solution in solutions:
            kpis = self.kpi_calculator.calculate_kpis(solution)
            solution_kpis.append(kpis)
        
        # Find best solution for each metric
        best_solutions = {}
        metrics = ['total_distance', 'total_cost', 'efficiency_score', 'load_balance_index']
        
        for metric in metrics:
            if metric in ['total_distance', 'total_cost']:
                # Lower is better
                best_idx = min(range(len(solution_kpis)), 
                             key=lambda i: solution_kpis[i][metric])
            else:
                # Higher is better
                best_idx = max(range(len(solution_kpis)), 
                             key=lambda i: solution_kpis[i][metric])
            
            best_solutions[metric] = {
                'solution_name': solution_names[best_idx],
                'value': solution_kpis[best_idx][metric]
            }
        
        # Calculate summary statistics
        summary_stats = self.kpi_calculator.get_summary_statistics(solutions)
        
        # Rank solutions
        ranked_solutions = self.rank_solutions(solutions)
        
        return {
            'solutions': [
                {
                    'name': name,
                    'kpis': kpis
                }
                for name, kpis in zip(solution_names, solution_kpis)
            ],
            'best_solutions': best_solutions,
            'summary_statistics': summary_stats,
            'rankings': [
                {
                    'rank': i + 1,
                    'solution_name': solution_names[solutions.index(solution)],
                    'score': score
                }
                for i, (solution, score) in enumerate(ranked_solutions)
            ],
            'pareto_solutions': [
                solution_names[solutions.index(sol)] 
                for sol in self.find_pareto_front(solutions)
            ]
        }


class SolutionValidator:
    """Validates VRP solutions for correctness."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize solution validator.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
    
    def validate_solution(self, individual: Individual) -> Dict:
        """
        Validate a VRP solution.
        
        Args:
            individual: Solution to validate
            
        Returns:
            Validation results
        """
        if individual.is_empty():
            return {
                'is_valid': False,
                'errors': ['Empty solution'],
                'warnings': []
            }
        
        errors = []
        warnings = []
        
        # Check chromosome validity
        chromosome_errors = self._validate_chromosome(individual.chromosome)
        errors.extend(chromosome_errors)
        
        # Check routes validity
        route_errors, route_warnings = self._validate_routes(individual.routes)
        errors.extend(route_errors)
        warnings.extend(route_warnings)
        
        # Check constraint satisfaction
        constraint_errors = self._validate_constraints(individual)
        errors.extend(constraint_errors)
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_chromosome(self, chromosome: List[int]) -> List[str]:
        """Validate chromosome structure."""
        errors = []
        
        if not chromosome:
            errors.append("Empty chromosome")
            return errors
        
        # Check for duplicates
        if len(chromosome) != len(set(chromosome)):
            errors.append("Duplicate customers in chromosome")
        
        # Check for invalid customer IDs
        valid_customer_ids = set(c.id for c in self.problem.customers)
        invalid_ids = set(chromosome) - valid_customer_ids
        if invalid_ids:
            errors.append(f"Invalid customer IDs: {invalid_ids}")
        
        # Check if all customers are included
        missing_customers = valid_customer_ids - set(chromosome)
        if missing_customers:
            errors.append(f"Missing customers: {missing_customers}")
        
        return errors
    
    def _validate_routes(self, routes: List[List[int]]) -> Tuple[List[str], List[str]]:
        """Validate route structure."""
        errors = []
        warnings = []
        
        if not routes:
            errors.append("No routes provided")
            return errors, warnings
        
        for i, route in enumerate(routes):
            if not route:
                warnings.append(f"Route {i} is empty")
                continue
            
            # Check depot constraints
            if route[0] != 0:
                errors.append(f"Route {i} doesn't start at depot")
            
            if route[-1] != 0:
                errors.append(f"Route {i} doesn't end at depot")
            
            # Check for duplicate customers in route
            customers = [c for c in route if c != 0]
            if len(customers) != len(set(customers)):
                errors.append(f"Route {i} has duplicate customers")
            
            # Check capacity constraint
            route_load = sum(
                self.problem.get_customer_by_id(c).demand 
                for c in customers
            )
            
            if route_load > self.problem.vehicle_capacity:
                errors.append(f"Route {i} exceeds capacity: {route_load} > {self.problem.vehicle_capacity}")
            
            # Check for very short routes
            if len(customers) == 1:
                warnings.append(f"Route {i} has only one customer")
        
        return errors, warnings
    
    def _validate_constraints(self, individual: Individual) -> List[str]:
        """Validate constraint satisfaction."""
        errors = []
        
        from src.data_processing.constraints import ConstraintHandler
        
        constraint_handler = ConstraintHandler(
            self.problem.vehicle_capacity, 
            self.problem.num_vehicles
        )
        
        demands = self.problem.get_demands()
        num_customers = len(self.problem.customers)
        
        validation_results = constraint_handler.validate_all_constraints(
            individual.routes, demands, self.problem.customers
        )
        
        if not validation_results['is_valid']:
            for violation_type, is_violated in validation_results['violations'].items():
                if is_violated:
                    errors.append(f"Constraint violation: {violation_type}")
        
        return errors


def compare_solutions(ga_solution: Individual, nn_solution: Individual, 
                    problem: VRPProblem) -> Dict:
    """
    Convenience function to compare solutions.
    
    Args:
        ga_solution: GA solution
        nn_solution: Nearest Neighbor solution
        problem: VRP problem instance
        
    Returns:
        Comparison results
    """
    comparator = SolutionComparator(problem)
    return comparator.compare_methods(ga_solution, nn_solution)
