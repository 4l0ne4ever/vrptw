"""
Solution validator for VRP problems.
Validates solution feasibility and correctness.
"""

from typing import List, Dict, Tuple, Optional
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.data_processing.constraints import ConstraintHandler


class SolutionValidator:
    """Validates VRP solutions for correctness and feasibility."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize solution validator.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
        dataset_type = getattr(problem, 'dataset_type', None)
        self.constraint_handler = ConstraintHandler(
            problem.vehicle_capacity, 
            problem.num_vehicles,
            dataset_type=dataset_type
        )
    
    def validate_solution(self, individual: Individual) -> Dict:
        """
        Validate a VRP solution comprehensively.
        
        Args:
            individual: Solution to validate
            
        Returns:
            Validation results dictionary
        """
        if individual.is_empty():
            return {
                'is_valid': False,
                'is_feasible': False,
                'errors': ['Empty solution'],
                'warnings': [],
                'constraint_violations': {},
                'validation_score': 0.0
            }
        
        errors = []
        warnings = []
        
        # Validate chromosome
        chromosome_errors = self._validate_chromosome(individual.chromosome)
        errors.extend(chromosome_errors)
        
        # Validate routes
        route_errors, route_warnings = self._validate_routes(individual.routes)
        errors.extend(route_errors)
        warnings.extend(route_warnings)
        
        # Validate constraints
        constraint_results = self._validate_constraints(individual)
        
        # Calculate validation score
        validation_score = self._calculate_validation_score(
            errors, warnings, constraint_results
        )
        
        return {
            'is_valid': len(errors) == 0,
            'is_feasible': constraint_results['is_valid'],
            'errors': errors,
            'warnings': warnings,
            'constraint_violations': constraint_results['violations'],
            'constraint_penalty': constraint_results['total_penalty'],
            'validation_score': validation_score
        }
    
    def _validate_chromosome(self, chromosome: List[int]) -> List[str]:
        """Validate chromosome structure and content."""
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
        
        # Check for depot in chromosome (should not be there)
        if 0 in chromosome:
            errors.append("Depot (0) should not be in chromosome")
        
        return errors
    
    def _validate_routes(self, routes: List[List[int]]) -> Tuple[List[str], List[str]]:
        """Validate route structure and constraints."""
        errors = []
        warnings = []
        
        if not routes:
            errors.append("No routes provided")
            return errors, warnings
        
        # Check number of routes
        if len(routes) > self.problem.num_vehicles:
            errors.append(f"Too many routes: {len(routes)} > {self.problem.num_vehicles}")
        
        # Validate each route
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
            
            # Check for very long routes
            if len(customers) > len(self.problem.customers) // 2:
                warnings.append(f"Route {i} is very long ({len(customers)} customers)")
        
        # Check if all customers are visited exactly once
        all_visited_customers = set()
        for route in routes:
            for customer_id in route:
                if customer_id != 0:
                    if customer_id in all_visited_customers:
                        errors.append(f"Customer {customer_id} visited multiple times")
                    all_visited_customers.add(customer_id)
        
        # Check for missing customers
        expected_customers = set(c.id for c in self.problem.customers)
        missing_customers = expected_customers - all_visited_customers
        if missing_customers:
            errors.append(f"Missing customers: {missing_customers}")
        
        return errors, warnings
    
    def _validate_constraints(self, individual: Individual) -> Dict:
        """Validate constraint satisfaction."""
        demands = self.problem.get_demands()
        num_customers = len(self.problem.customers)
        
        validation_results = self.constraint_handler.validate_all_constraints(
            individual.routes, demands, self.problem.customers
        )
        
        return validation_results
    
    def _calculate_validation_score(self, errors: List[str], warnings: List[str], 
                                  constraint_results: Dict) -> float:
        """
        Calculate validation score (0-1, higher is better).
        
        Args:
            errors: List of errors
            warnings: List of warnings
            constraint_results: Constraint validation results
            
        Returns:
            Validation score
        """
        if errors:
            return 0.0
        
        score = 1.0
        
        # Deduct for warnings
        score -= len(warnings) * 0.1
        
        # Deduct for constraint violations
        if constraint_results['total_penalty'] > 0:
            score -= min(0.5, constraint_results['total_penalty'] / 1000)
        
        return max(0.0, score)
    
    def validate_population(self, population: List[Individual]) -> Dict:
        """
        Validate entire population.
        
        Args:
            population: List of individuals to validate
            
        Returns:
            Population validation results
        """
        if not population:
            return {
                'num_solutions': 0,
                'valid_solutions': 0,
                'feasible_solutions': 0,
                'avg_validation_score': 0.0,
                'validation_summary': {}
            }
        
        validation_results = []
        valid_count = 0
        feasible_count = 0
        total_score = 0.0
        
        for individual in population:
            result = self.validate_solution(individual)
            validation_results.append(result)
            
            if result['is_valid']:
                valid_count += 1
            
            if result['is_feasible']:
                feasible_count += 1
            
            total_score += result['validation_score']
        
        return {
            'num_solutions': len(population),
            'valid_solutions': valid_count,
            'feasible_solutions': feasible_count,
            'valid_percentage': (valid_count / len(population)) * 100,
            'feasible_percentage': (feasible_count / len(population)) * 100,
            'avg_validation_score': total_score / len(population),
            'validation_results': validation_results
        }
    
    def get_feasible_solutions(self, population: List[Individual]) -> List[Individual]:
        """
        Get all feasible solutions from population.
        
        Args:
            population: List of individuals
            
        Returns:
            List of feasible individuals
        """
        feasible_solutions = []
        
        for individual in population:
            validation_result = self.validate_solution(individual)
            if validation_result['is_feasible']:
                feasible_solutions.append(individual)
        
        return feasible_solutions
    
    def get_best_feasible_solution(self, population: List[Individual]) -> Optional[Individual]:
        """
        Get best feasible solution from population.
        
        Args:
            population: List of individuals
            
        Returns:
            Best feasible individual or None
        """
        feasible_solutions = self.get_feasible_solutions(population)
        
        if not feasible_solutions:
            return None
        
        return max(feasible_solutions, key=lambda ind: ind.fitness)
    
    def repair_solution(self, individual: Individual) -> Individual:
        """
        Attempt to repair an invalid solution.
        
        Args:
            individual: Individual to repair
            
        Returns:
            Repaired individual
        """
        if individual.is_empty():
            return individual
        
        repaired_individual = individual.copy()
        
        # Repair chromosome
        repaired_chromosome = self._repair_chromosome(individual.chromosome)
        repaired_individual.chromosome = repaired_chromosome
        
        # Re-decode routes
        from src.algorithms.decoder import RouteDecoder
        decoder = RouteDecoder(self.problem)
        repaired_routes = decoder.decode_chromosome(repaired_chromosome)
        repaired_individual.routes = repaired_routes
        
        return repaired_individual
    
    def _repair_chromosome(self, chromosome: List[int]) -> List[int]:
        """Repair chromosome by fixing common issues."""
        if not chromosome:
            return []
        
        # Remove duplicates while preserving order
        seen = set()
        repaired = []
        for customer_id in chromosome:
            if customer_id not in seen and customer_id != 0:
                repaired.append(customer_id)
                seen.add(customer_id)
        
        # Add missing customers
        expected_customers = set(c.id for c in self.problem.customers)
        missing_customers = expected_customers - set(repaired)
        repaired.extend(list(missing_customers))
        
        return repaired
    
    def validate_solution_quality(self, individual: Individual) -> Dict:
        """
        Validate solution quality beyond basic feasibility.
        
        Args:
            individual: Solution to validate
            
        Returns:
            Quality assessment
        """
        if individual.is_empty():
            return {
                'quality_score': 0.0,
                'quality_issues': ['Empty solution'],
                'recommendations': []
            }
        
        issues = []
        recommendations = []
        score = 1.0
        
        # Check route efficiency
        if individual.routes:
            route_lengths = [len([c for c in route if c != 0]) for route in individual.routes if route]
            
            if len(route_lengths) > 1:
                length_variance = sum((length - sum(route_lengths)/len(route_lengths))**2 for length in route_lengths) / len(route_lengths)
                
                if length_variance > 5:  # High variance in route lengths
                    issues.append("Uneven route lengths")
                    recommendations.append("Consider load balancing")
                    score -= 0.2
        
        # Check load utilization
        route_loads = []
        for route in individual.routes:
            if route:
                route_load = sum(
                    self.problem.get_customer_by_id(c).demand 
                    for c in route if c != 0
                )
                route_loads.append(route_load)
        
        if route_loads:
            avg_utilization = sum(route_loads) / (len(route_loads) * self.problem.vehicle_capacity)
            
            if avg_utilization < 0.5:
                issues.append("Low vehicle utilization")
                recommendations.append("Consider reducing number of vehicles")
                score -= 0.3
            elif avg_utilization > 0.9:
                issues.append("Very high vehicle utilization")
                recommendations.append("Consider adding more vehicles for robustness")
                score -= 0.1
        
        # Check for very short routes
        short_routes = [route for route in individual.routes if route and len([c for c in route if c != 0]) == 1]
        if len(short_routes) > len(individual.routes) * 0.3:
            issues.append("Many single-customer routes")
            recommendations.append("Consider merging short routes")
            score -= 0.2
        
        return {
            'quality_score': max(0.0, score),
            'quality_issues': issues,
            'recommendations': recommendations
        }


def validate_solution(individual: Individual, problem: VRPProblem) -> Dict:
    """
    Convenience function to validate a solution.
    
    Args:
        individual: Solution to validate
        problem: VRP problem instance
        
    Returns:
        Validation results
    """
    validator = SolutionValidator(problem)
    return validator.validate_solution(individual)
