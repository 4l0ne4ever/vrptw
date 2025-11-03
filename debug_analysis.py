#!/usr/bin/env python3
"""
Deep debugging and analysis script for VRP-GA system.
Analyzes each layer, function, and file to identify issues.
"""

import sys
import os
import json
import traceback
from typing import List, Dict, Tuple, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.vrp_model import VRPProblem, Customer, Depot
from src.data_processing.loader import SolomonLoader
from src.algorithms.decoder import RouteDecoder
from src.algorithms.fitness import FitnessEvaluator
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.nearest_neighbor import NearestNeighborHeuristic
from src.data_processing.constraints import ConstraintHandler
from src.evaluation.metrics import KPICalculator
from config import GA_CONFIG, VRP_CONFIG


class DeepDebugger:
    """Deep debugging and analysis for VRP-GA system."""
    
    def __init__(self):
        self.issues = []
        self.findings = []
        
    def log_issue(self, component: str, severity: str, description: str, details: str = ""):
        """Log an issue found during debugging."""
        self.issues.append({
            'component': component,
            'severity': severity,
            'description': description,
            'details': details
        })
        print(f"\n{'='*80}")
        print(f"[{severity}] {component}: {description}")
        if details:
            print(f"Details: {details}")
        print(f"{'='*80}")
    
    def log_finding(self, component: str, finding: str):
        """Log a finding (positive or negative)."""
        self.findings.append({
            'component': component,
            'finding': finding
        })
        print(f"[FINDING] {component}: {finding}")
    
    def analyze_decoder(self, problem: VRPProblem, test_chromosome: Optional[List[int]] = None):
        """Analyze decoder logic and behavior."""
        print("\n" + "="*80)
        print("ANALYZING DECODER")
        print("="*80)
        
        decoder = RouteDecoder(problem, use_split_algorithm=False)
        
        # Check if Split Algorithm is enabled
        if decoder.use_split:
            self.log_finding("Decoder", "Split Algorithm is ENABLED")
        else:
            self.log_issue("Decoder", "MEDIUM", 
                          "Split Algorithm is DISABLED",
                          "Using greedy decoder which may not be optimal for C-series")
        
        # Test decoding with a sample chromosome
        if test_chromosome is None:
            customer_ids = [c.id for c in problem.customers]
            test_chromosome = customer_ids.copy()
            import random
            random.shuffle(test_chromosome)
        
        try:
            routes = decoder.decode_chromosome(test_chromosome)
            
            # Analyze routes
            num_routes = len(routes)
            total_customers = sum(len([c for c in route if c != 0]) for route in routes)
            expected_customers = len(problem.customers)
            
            self.log_finding("Decoder", f"Decoded {num_routes} routes")
            self.log_finding("Decoder", f"Total customers in routes: {total_customers}/{expected_customers}")
            
            # Check capacity constraints
            constraint_handler = ConstraintHandler(problem.vehicle_capacity, problem.num_vehicles)
            demands = [c.demand for c in problem.customers]
            max_customer_id = max((max(route) for route in routes if route), default=0)
            while len(demands) < max_customer_id:
                demands.append(0.0)
            
            cap_valid, cap_penalty = constraint_handler.validate_capacity_constraint(routes, demands)
            
            if not cap_valid:
                self.log_issue("Decoder", "HIGH",
                              "Decoded routes violate capacity constraints",
                              f"Penalty: {cap_penalty}")
            else:
                self.log_finding("Decoder", "Routes satisfy capacity constraints")
            
            # Analyze route loads
            route_loads = []
            for route in routes:
                route_load = sum(
                    problem.get_customer_by_id(c).demand 
                    for c in route if c != 0
                )
                route_loads.append(route_load)
            
            avg_load = sum(route_loads) / len(route_loads) if route_loads else 0
            max_load = max(route_loads) if route_loads else 0
            min_load = min(route_loads) if route_loads else 0
            
            self.log_finding("Decoder", f"Route loads - Avg: {avg_load:.2f}, Max: {max_load:.2f}, Min: {min_load:.2f}")
            self.log_finding("Decoder", f"Capacity utilization - Avg: {avg_load/problem.vehicle_capacity*100:.1f}%")
            
            # Check if decoder is creating too many routes
            if num_routes > problem.num_vehicles:
                self.log_issue("Decoder", "HIGH",
                              f"Decoder creates {num_routes} routes, exceeding limit of {problem.num_vehicles}",
                              "This may cause infeasible solutions")
            
            return routes, num_routes, route_loads
            
        except Exception as e:
            self.log_issue("Decoder", "CRITICAL",
                          f"Decoder failed with error: {str(e)}",
                          traceback.format_exc())
            return None, 0, []
    
    def analyze_fitness(self, problem: VRPProblem, routes: List[List[int]]):
        """Analyze fitness calculation logic."""
        print("\n" + "="*80)
        print("ANALYZING FITNESS FUNCTION")
        print("="*80)
        
        fitness_evaluator = FitnessEvaluator(problem, penalty_weight=VRP_CONFIG['penalty_weight'])
        
        # Create a test individual
        from src.models.solution import Individual
        from src.algorithms.decoder import RouteDecoder
        decoder = RouteDecoder(problem)
        
        # Reconstruct chromosome from routes
        chromosome = decoder.merge_routes(routes) if routes else [c.id for c in problem.customers]
        
        individual = Individual(chromosome=chromosome)
        
        try:
            fitness = fitness_evaluator.evaluate_fitness(individual)
            
            self.log_finding("Fitness", f"Fitness value: {fitness:.6f}")
            self.log_finding("Fitness", f"Total distance: {individual.total_distance:.2f}")
            self.log_finding("Fitness", f"Penalty: {individual.penalty:.2f}")
            self.log_finding("Fitness", f"Is valid: {individual.is_valid}")
            
            # Analyze penalty calculation
            if individual.penalty > 0:
                self.log_issue("Fitness", "MEDIUM",
                              f"Solution has penalty: {individual.penalty}",
                              "This indicates constraint violations")
            
            # Check penalty weight
            penalty_weight = fitness_evaluator.penalty_weight
            self.log_finding("Fitness", f"Penalty weight: {penalty_weight}")
            
            if penalty_weight < 100:
                self.log_issue("Fitness", "MEDIUM",
                              f"Penalty weight may be too low: {penalty_weight}",
                              "Infeasible solutions may not be penalized enough")
            
            return fitness, individual
            
        except Exception as e:
            self.log_issue("Fitness", "CRITICAL",
                          f"Fitness evaluation failed: {str(e)}",
                          traceback.format_exc())
            return None, None
    
    def analyze_constraints(self, problem: VRPProblem, routes: List[List[int]]):
        """Analyze constraint handling."""
        print("\n" + "="*80)
        print("ANALYZING CONSTRAINT HANDLING")
        print("="*80)
        
        constraint_handler = ConstraintHandler(problem.vehicle_capacity, problem.num_vehicles)
        
        demands = [c.demand for c in problem.customers]
        max_customer_id = max((max(route) for route in routes if route), default=0)
        while len(demands) < max_customer_id:
            demands.append(0.0)
        
        # Validate all constraints
        all_violations = constraint_handler.validate_all_constraints(
            routes, demands, problem.customers
        )
        
        self.log_finding("Constraints", f"Total violations: {all_violations['total_violations']}")
        self.log_finding("Constraints", f"Capacity violations: {all_violations['capacity_violations']}")
        self.log_finding("Constraints", f"Vehicle count violations: {all_violations['vehicle_count_violations']}")
        self.log_finding("Constraints", f"Customer visit violations: {all_violations['customer_visit_violations']}")
        
        if all_violations['total_violations'] > 0:
            self.log_issue("Constraints", "HIGH",
                          f"Solution has {all_violations['total_violations']} constraint violations",
                          json.dumps(all_violations, indent=2))
        
        # Test repair mechanism
        try:
            repaired_routes = constraint_handler.repair_capacity_violations(routes, demands)
            
            repaired_violations = constraint_handler.validate_all_constraints(
                repaired_routes, demands, problem.customers
            )
            
            if repaired_violations['total_violations'] == 0:
                self.log_finding("Constraints", "Repair mechanism works correctly")
            else:
                self.log_issue("Constraints", "HIGH",
                              "Repair mechanism failed to fix all violations",
                              f"Remaining violations: {repaired_violations['total_violations']}")
                
        except Exception as e:
            self.log_issue("Constraints", "CRITICAL",
                          f"Repair mechanism failed: {str(e)}",
                          traceback.format_exc())
    
    def analyze_ga_convergence(self, problem: VRPProblem, max_generations: int = 50):
        """Analyze GA convergence behavior."""
        print("\n" + "="*80)
        print("ANALYZING GA CONVERGENCE")
        print("="*80)
        
        ga_config = GA_CONFIG.copy()
        ga_config['generations'] = max_generations
        ga_config['population_size'] = 50  # Smaller for testing
        
        try:
            ga = GeneticAlgorithm(problem, ga_config)
            
            # Run for a few generations
            best_solution, evolution_data = ga.evolve()
            
            if not evolution_data:
                self.log_issue("GA Convergence", "CRITICAL",
                              "No evolution data collected",
                              "GA may have failed to run")
                return
            
            initial_distance = evolution_data[0]['best_distance']
            final_distance = evolution_data[-1]['best_distance']
            improvement = ((initial_distance - final_distance) / initial_distance) * 100
            
            self.log_finding("GA Convergence", f"Initial distance: {initial_distance:.2f}")
            self.log_finding("GA Convergence", f"Final distance: {final_distance:.2f}")
            self.log_finding("GA Convergence", f"Improvement: {improvement:.2f}%")
            
            if improvement < 1.0:
                self.log_issue("GA Convergence", "MEDIUM",
                              f"Improvement is too low: {improvement:.2f}%",
                              "GA may be converging too early or not exploring enough")
            
            # Check diversity
            initial_diversity = evolution_data[0]['diversity']
            final_diversity = evolution_data[-1]['diversity']
            diversity_loss = ((initial_diversity - final_diversity) / initial_diversity) * 100
            
            self.log_finding("GA Convergence", f"Diversity loss: {diversity_loss:.2f}%")
            
            if diversity_loss > 50:
                self.log_issue("GA Convergence", "MEDIUM",
                              f"Diversity loss is high: {diversity_loss:.2f}%",
                              "Population may be converging too quickly")
            
            # Check if best solution improved
            best_improved = False
            for i in range(1, len(evolution_data)):
                if evolution_data[i]['best_distance'] < evolution_data[i-1]['best_distance']:
                    best_improved = True
                    break
            
            if not best_improved:
                self.log_issue("GA Convergence", "HIGH",
                              "Best solution did not improve during evolution",
                              "GA may be stuck in local optimum")
            
        except Exception as e:
            self.log_issue("GA Convergence", "CRITICAL",
                          f"GA analysis failed: {str(e)}",
                          traceback.format_exc())
    
    def analyze_c_series_issue(self, problem: VRPProblem):
        """Analyze why C-series has poor performance."""
        print("\n" + "="*80)
        print("ANALYZING C-SERIES ISSUE")
        print("="*80)
        
        # Check if this is a C-series problem (clustered customers)
        customer_coords = [(c.x, c.y) for c in problem.customers]
        depot_coord = (problem.depot.x, problem.depot.y)
        
        # Calculate distances from depot
        distances_from_depot = [
            ((c.x - depot_coord[0])**2 + (c.y - depot_coord[1])**2)**0.5
            for c in problem.customers
        ]
        
        avg_distance = sum(distances_from_depot) / len(distances_from_depot)
        std_distance = (sum((d - avg_distance)**2 for d in distances_from_depot) / len(distances_from_depot))**0.5
        
        # If low std, customers are clustered
        is_clustered = std_distance < avg_distance * 0.5
        
        if is_clustered:
            self.log_finding("C-series", "Detected clustered customer distribution")
            self.log_issue("C-series", "MEDIUM",
                          "Clustered customers detected",
                          "Greedy decoder may not be optimal for clustered problems")
        else:
            self.log_finding("C-series", "Random customer distribution detected")
        
        # Check decoder behavior with clustered customers
        decoder = RouteDecoder(problem, use_split_algorithm=False)
        customer_ids = [c.id for c in problem.customers]
        import random
        random.shuffle(customer_ids)
        
        routes = decoder.decode_chromosome(customer_ids)
        num_routes = len(routes)
        
        self.log_finding("C-series", f"Greedy decoder creates {num_routes} routes")
        
        # Try with Split Algorithm if available
        if hasattr(decoder, 'use_split') and not decoder.use_split:
            try:
                decoder_split = RouteDecoder(problem, use_split_algorithm=True)
                routes_split = decoder_split.decode_chromosome(customer_ids)
                num_routes_split = len(routes_split)
                
                self.log_finding("C-series", f"Split Algorithm creates {num_routes_split} routes")
                
                if num_routes_split < num_routes:
                    self.log_issue("C-series", "HIGH",
                                  f"Split Algorithm uses {num_routes_split} routes vs {num_routes} for greedy",
                                  "Split Algorithm should be enabled for better performance")
                
            except Exception as e:
                self.log_issue("C-series", "MEDIUM",
                              f"Split Algorithm not available or failed: {str(e)}",
                              "This may explain poor C-series performance")
    
    def analyze_mockup_vs_nn(self, problem: VRPProblem):
        """Analyze why GA performs worse than NN for mockup dataset."""
        print("\n" + "="*80)
        print("ANALYZING MOCKUP GA vs NN")
        print("="*80)
        
        # Run NN
        try:
            nn_heuristic = NearestNeighborHeuristic(problem)
            nn_solution = nn_heuristic.solve()
            
            self.log_finding("Mockup vs NN", f"NN Distance: {nn_solution.total_distance:.2f}")
            self.log_finding("Mockup vs NN", f"NN Routes: {len(nn_solution.routes)}")
            
            # Run GA with limited generations (simulate test scenario)
            ga_config = GA_CONFIG.copy()
            ga_config['generations'] = 50  # Same as test scenario
            ga_config['population_size'] = 30  # Same as test scenario
            
            ga = GeneticAlgorithm(problem, ga_config)
            ga_solution, evolution_data = ga.evolve()
            
            self.log_finding("Mockup vs NN", f"GA Distance: {ga_solution.total_distance:.2f}")
            self.log_finding("Mockup vs NN", f"GA Routes: {len(ga_solution.routes)}")
            
            if ga_solution.total_distance > nn_solution.total_distance:
                improvement = ((ga_solution.total_distance - nn_solution.total_distance) / 
                              nn_solution.total_distance) * 100
                self.log_issue("Mockup vs NN", "HIGH",
                              f"GA is {improvement:.2f}% WORSE than NN",
                              "GA should always be better than or equal to NN")
                
                # Analyze why
                self.log_finding("Mockup vs NN", f"GA Generations: {ga_config['generations']}")
                self.log_finding("Mockup vs NN", f"GA Population: {ga_config['population_size']}")
                
                if ga_config['generations'] < 100:
                    self.log_issue("Mockup vs NN", "MEDIUM",
                                  f"Generations too low: {ga_config['generations']}",
                                  "GA needs more generations to improve over NN")
                
                if ga_config['population_size'] < 50:
                    self.log_issue("Mockup vs NN", "MEDIUM",
                                  f"Population too small: {ga_config['population_size']}",
                                  "GA needs larger population for better diversity")
                
                # Check if GA found any improvement
                if evolution_data:
                    initial_distance = evolution_data[0]['best_distance']
                    improvement_achieved = ((initial_distance - ga_solution.total_distance) / 
                                           initial_distance) * 100
                    
                    if improvement_achieved < 1.0:
                        self.log_issue("Mockup vs NN", "MEDIUM",
                                      f"GA improvement too low: {improvement_achieved:.2f}%",
                                      "GA may not be exploring solution space effectively")
            
        except Exception as e:
            self.log_issue("Mockup vs NN", "CRITICAL",
                          f"Analysis failed: {str(e)}",
                          traceback.format_exc())
    
    def generate_report(self) -> Dict:
        """Generate final debug report."""
        print("\n" + "="*80)
        print("FINAL DEBUG REPORT")
        print("="*80)
        
        # Count issues by severity
        critical = [i for i in self.issues if i['severity'] == 'CRITICAL']
        high = [i for i in self.issues if i['severity'] == 'HIGH']
        medium = [i for i in self.issues if i['severity'] == 'MEDIUM']
        
        print(f"\nIssues Found:")
        print(f"  CRITICAL: {len(critical)}")
        print(f"  HIGH: {len(high)}")
        print(f"  MEDIUM: {len(medium)}")
        print(f"  Total: {len(self.issues)}")
        
        print(f"\nFindings: {len(self.findings)}")
        
        report = {
            'issues': self.issues,
            'findings': self.findings,
            'summary': {
                'total_issues': len(self.issues),
                'critical_issues': len(critical),
                'high_issues': len(high),
                'medium_issues': len(medium),
                'total_findings': len(self.findings)
            }
        }
        
        return report


def main():
    """Main debug analysis function."""
    print("="*80)
    print("VRP-GA SYSTEM DEEP DEBUG ANALYSIS")
    print("="*80)
    
    debugger = DeepDebugger()
    
    # Load a sample problem (C-series for testing)
    try:
        loader = SolomonLoader('data/solomon_dataset')
        problem = loader.load_instance('C101')
        
        print(f"\nLoaded problem: C101")
        print(f"  Customers: {len(problem.customers)}")
        print(f"  Vehicle Capacity: {problem.vehicle_capacity}")
        print(f"  Num Vehicles: {problem.num_vehicles}")
        
        # Analyze decoder
        routes, num_routes, route_loads = debugger.analyze_decoder(problem)
        
        if routes:
            # Analyze fitness
            fitness, individual = debugger.analyze_fitness(problem, routes)
            
            # Analyze constraints
            debugger.analyze_constraints(problem, routes)
            
            # Analyze C-series issue
            debugger.analyze_c_series_issue(problem)
        
        # Analyze GA convergence (with limited generations)
        debugger.analyze_ga_convergence(problem, max_generations=20)
        
        # Generate report
        report = debugger.generate_report()
        
        # Save report
        with open('debug_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Debug report saved to debug_report.json")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    main()

