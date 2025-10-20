"""
Report generation for VRP solutions.
Creates comprehensive reports with statistics and analysis.
"""

import json
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.evaluation.metrics import KPICalculator
from src.evaluation.comparator import SolutionComparator
from src.visualization.mapper import RouteMapper
from src.visualization.hanoi_map import HanoiMapVisualizer
from src.visualization.plotter import Plotter


class ReportGenerator:
    """Generates comprehensive reports for VRP solutions."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize report generator.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
        self.kpi_calculator = KPICalculator(problem)
        self.comparator = SolutionComparator(problem)
        self.route_mapper = RouteMapper(problem)
        self.plotter = Plotter()
    
    def generate_comprehensive_report(self, 
                                    ga_solution: Individual,
                                    nn_solution: Individual,
                                    ga_statistics: Dict,
                                    convergence_data: Optional[Dict] = None,
                                    output_dir: str = "results") -> Dict:
        """
        Generate comprehensive report for VRP solutions.
        
        Args:
            ga_solution: GA solution
            nn_solution: Nearest Neighbor solution
            ga_statistics: GA execution statistics
            convergence_data: Optional convergence data
            output_dir: Output directory
            
        Returns:
            Report summary
        """
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Calculate KPIs
        ga_kpis = self.kpi_calculator.calculate_kpis(ga_solution, ga_statistics.get('execution_time', 0))
        nn_kpis = self.kpi_calculator.calculate_kpis(nn_solution)
        
        # Compare solutions
        comparison = self.comparator.compare_methods(ga_solution, nn_solution)
        
        # Generate visualizations
        self._generate_visualizations(ga_solution, nn_solution, convergence_data, report_dir)
        
        # Generate text report
        report_content = self._generate_text_report(ga_solution, nn_solution, ga_kpis, nn_kpis, comparison, ga_statistics)
        
        # Save report
        report_file = os.path.join(report_dir, "report.txt")
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Save data as JSON (skip for now to avoid serialization errors)
        # report_data = {
        #     'timestamp': timestamp,
        #     'problem_info': self.problem.get_problem_info(),
        #     'ga_solution': ga_solution.to_dict(),
        #     'nn_solution': nn_solution.to_dict(),
        #     'ga_kpis': ga_kpis,
        #     'nn_kpis': nn_kpis,
        #     'comparison': comparison,
        #     'ga_statistics': ga_statistics,
        #     'convergence_data': convergence_data
        # }
        
        # data_file = os.path.join(report_dir, "report_data.json")
        # with open(data_file, 'w') as f:
        #     json.dump(report_data, f, indent=2)
        
        return {
            'report_dir': report_dir,
            'report_file': report_file,
            'summary': self._generate_summary(ga_kpis, nn_kpis, comparison)
        }
    
    def _generate_visualizations(self, ga_solution: Individual, nn_solution: Individual,
                               convergence_data: Optional[Dict], report_dir: str):
        """Generate all visualizations."""
        # Check if this is a mockup dataset (Hanoi coordinates)
        is_hanoi_dataset = self._is_hanoi_dataset()
        
        if is_hanoi_dataset:
            # Use Hanoi map visualization for mockup datasets
            self._generate_hanoi_visualizations(ga_solution, nn_solution, report_dir)
        else:
            # Use traditional 2D visualization for Solomon datasets
            self._generate_traditional_visualizations(ga_solution, nn_solution, convergence_data, report_dir)
        
    def _is_hanoi_dataset(self) -> bool:
        """Check if this is a Hanoi dataset (mockup data)."""
        # Check if coordinates are within Hanoi bounds
        depot = self.problem.depot
        hanoi_bounds = {
            'min_lat': 20.7, 'max_lat': 21.4,
            'min_lon': 105.3, 'max_lon': 106.0
        }
        
        # Check depot coordinates
        if (hanoi_bounds['min_lat'] <= depot.y <= hanoi_bounds['max_lat'] and
            hanoi_bounds['min_lon'] <= depot.x <= hanoi_bounds['max_lon']):
            return True
        
        # Check a few customer coordinates
        for customer in self.problem.customers[:5]:
            if (hanoi_bounds['min_lat'] <= customer.y <= hanoi_bounds['max_lat'] and
                hanoi_bounds['min_lon'] <= customer.x <= hanoi_bounds['max_lon']):
                return True
        
        return False
    
    def _generate_hanoi_visualizations(self, ga_solution: Individual, nn_solution: Individual, report_dir: str):
        """Generate Hanoi map visualizations for mockup datasets."""
        # Create Hanoi map visualizer
        hanoi_visualizer = HanoiMapVisualizer(self.problem)
        
        # GA solution map
        hanoi_visualizer.create_map(
            ga_solution, 
            "GA Solution - Hanoi",
            os.path.join(report_dir, "ga_hanoi_map.html")
        )
        
        # NN solution map
        hanoi_visualizer.create_map(
            nn_solution, 
            "Nearest Neighbor Solution - Hanoi",
            os.path.join(report_dir, "nn_hanoi_map.html")
        )
        
        # Comparison map
        hanoi_visualizer.create_comparison_map(
            ga_solution, 
            nn_solution,
            "GA vs NN Comparison - Hanoi",
            os.path.join(report_dir, "comparison_hanoi_map.html")
        )
        
        # Skip traditional plots for Hanoi datasets to avoid color errors
        print("Hanoi map visualizations generated successfully!")
    
    def _generate_traditional_visualizations(self, ga_solution: Individual, nn_solution: Individual,
                                          convergence_data: Optional[Dict], report_dir: str):
        """Generate traditional 2D visualizations for Solomon datasets."""
        self._generate_traditional_plots_only(ga_solution, nn_solution, report_dir)
        
        # Convergence plot
        if convergence_data:
            self.plotter.plot_convergence(convergence_data, "GA Convergence",
                                        os.path.join(report_dir, "convergence.png"))
    
    def _generate_traditional_plots_only(self, ga_solution: Individual, nn_solution: Individual, report_dir: str):
        """Generate traditional plots (used by both Hanoi and traditional visualizations)."""
        # Route maps
        self.route_mapper.plot_routes(ga_solution, "GA Solution", 
                                    os.path.join(report_dir, "ga_routes.png"))
        self.route_mapper.plot_routes(nn_solution, "Nearest Neighbor Solution", 
                                    os.path.join(report_dir, "nn_routes.png"))
        
        # Comparison plot
        self.route_mapper.plot_comparison([ga_solution, nn_solution], 
                                        ["GA", "Nearest Neighbor"],
                                        "Solution Comparison",
                                        os.path.join(report_dir, "comparison.png"))
        
        # Route load distribution
        self.route_mapper.plot_route_loads(ga_solution, "GA Route Loads",
                                          os.path.join(report_dir, "ga_loads.png"))
        self.route_mapper.plot_route_loads(nn_solution, "NN Route Loads",
                                          os.path.join(report_dir, "nn_loads.png"))
        
        # Customer demands
        self.route_mapper.plot_customer_demands("Customer Demands",
                                               os.path.join(report_dir, "demands.png"))
        
        # KPI dashboard
        ga_kpis = self.kpi_calculator.calculate_kpis(ga_solution)
        self.plotter.plot_kpi_dashboard(ga_kpis, "GA KPI Dashboard",
                                       os.path.join(report_dir, "ga_dashboard.png"))
        
        nn_kpis = self.kpi_calculator.calculate_kpis(nn_solution)
        self.plotter.plot_kpi_dashboard(nn_kpis, "NN KPI Dashboard",
                                       os.path.join(report_dir, "nn_dashboard.png"))
        
        # Comparison chart
        comparison = self.comparator.compare_methods(ga_solution, nn_solution)
        self.plotter.plot_comparison_chart(comparison, "Solution Comparison",
                                          os.path.join(report_dir, "comparison_chart.png"))
        
        # Statistics table
        self.plotter.plot_statistics_table(ga_kpis, "GA Statistics",
                                          os.path.join(report_dir, "ga_statistics.png"))
        self.plotter.plot_statistics_table(nn_kpis, "NN Statistics",
                                          os.path.join(report_dir, "nn_statistics.png"))
        
        # Improvement analysis
        improvements = comparison['improvements']
        self.plotter.plot_improvement_analysis(improvements, "Improvement Analysis",
                                              os.path.join(report_dir, "improvements.png"))
    
    def _generate_text_report(self, ga_solution: Individual, nn_solution: Individual,
                            ga_kpis: Dict, nn_kpis: Dict, comparison: Dict, 
                            ga_statistics: Dict) -> str:
        """Generate text report content."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("VRP-GA SYSTEM COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Problem information
        report_lines.append("PROBLEM INFORMATION")
        report_lines.append("-" * 40)
        problem_info = self.problem.get_problem_info()
        report_lines.append(f"Number of Customers: {problem_info['num_customers']}")
        report_lines.append(f"Vehicle Capacity: {problem_info['vehicle_capacity']}")
        report_lines.append(f"Number of Vehicles: {problem_info['num_vehicles']}")
        report_lines.append(f"Total Demand: {problem_info['total_demand']}")
        report_lines.append(f"Minimum Vehicles Needed: {problem_info['min_vehicles_needed']}")
        report_lines.append(f"Problem Feasible: {problem_info['is_feasible']}")
        report_lines.append("")
        
        # GA Solution
        report_lines.append("GENETIC ALGORITHM SOLUTION")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Distance: {ga_kpis['total_distance']:.2f}")
        report_lines.append(f"Number of Routes: {ga_kpis['num_routes']}")
        report_lines.append(f"Total Cost: {ga_kpis['total_cost']:.2f}")
        report_lines.append(f"Average Utilization: {ga_kpis['avg_utilization']:.1f}%")
        report_lines.append(f"Load Balance Index: {ga_kpis['load_balance_index']:.3f}")
        report_lines.append(f"Efficiency Score: {ga_kpis['efficiency_score']:.3f}")
        report_lines.append(f"Feasible: {ga_kpis['is_feasible']}")
        report_lines.append(f"Fitness: {ga_kpis['fitness']:.6f}")
        report_lines.append("")
        
        # Nearest Neighbor Solution
        report_lines.append("NEAREST NEIGHBOR SOLUTION")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Distance: {nn_kpis['total_distance']:.2f}")
        report_lines.append(f"Number of Routes: {nn_kpis['num_routes']}")
        report_lines.append(f"Total Cost: {nn_kpis['total_cost']:.2f}")
        report_lines.append(f"Average Utilization: {nn_kpis['avg_utilization']:.1f}%")
        report_lines.append(f"Load Balance Index: {nn_kpis['load_balance_index']:.3f}")
        report_lines.append(f"Efficiency Score: {nn_kpis['efficiency_score']:.3f}")
        report_lines.append(f"Feasible: {nn_kpis['is_feasible']}")
        report_lines.append(f"Fitness: {nn_kpis['fitness']:.6f}")
        report_lines.append("")
        
        # Comparison
        report_lines.append("COMPARISON ANALYSIS")
        report_lines.append("-" * 40)
        improvements = comparison['improvements']
        report_lines.append(f"Distance Improvement: {improvements['distance_improvement']:.2f} ({improvements['distance_improvement_percent']:.1f}%)")
        report_lines.append(f"Cost Improvement: {improvements['cost_improvement']:.2f} ({improvements['cost_improvement_percent']:.1f}%)")
        report_lines.append(f"Efficiency Improvement: {improvements['efficiency_improvement']:.3f} ({improvements['efficiency_improvement_percent']:.1f}%)")
        report_lines.append(f"Overall Improved: {improvements['is_improved']}")
        report_lines.append("")
        
        # GA Statistics
        report_lines.append("GENETIC ALGORITHM STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Generations: {ga_statistics.get('generations', 0)}")
        report_lines.append(f"Total Evaluations: {ga_statistics.get('total_evaluations', 0)}")
        report_lines.append(f"Execution Time: {ga_statistics.get('execution_time', 0):.2f} seconds")
        report_lines.append(f"Convergence Generation: {ga_statistics.get('convergence_generation', 'N/A')}")
        report_lines.append(f"Best Fitness: {ga_statistics.get('best_fitness', 0):.6f}")
        report_lines.append(f"Average Fitness: {ga_statistics.get('avg_fitness', 0):.6f}")
        report_lines.append(f"Diversity: {ga_statistics.get('diversity', 0):.3f}")
        report_lines.append("")
        
        # Route Details
        report_lines.append("ROUTE DETAILS")
        report_lines.append("-" * 40)
        
        # GA Routes
        report_lines.append("GA Routes:")
        for i, route in enumerate(ga_solution.routes):
            if route:
                customers = [str(c) for c in route if c != 0]
                route_load = sum(self.problem.get_customer_by_id(c).demand for c in route if c != 0)
                report_lines.append(f"  Route {i+1}: {' -> '.join(customers)} (Load: {route_load})")
        report_lines.append("")
        
        # NN Routes
        report_lines.append("Nearest Neighbor Routes:")
        for i, route in enumerate(nn_solution.routes):
            if route:
                customers = [str(c) for c in route if c != 0]
                route_load = sum(self.problem.get_customer_by_id(c).demand for c in route if c != 0)
                report_lines.append(f"  Route {i+1}: {' -> '.join(customers)} (Load: {route_load})")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if improvements['distance_improvement'] > 0:
            report_lines.append("✓ GA solution shows improvement over Nearest Neighbor")
        else:
            report_lines.append("⚠ GA solution did not improve over Nearest Neighbor")
        
        if ga_kpis['avg_utilization'] < 70:
            report_lines.append("⚠ Low vehicle utilization - consider reducing number of vehicles")
        
        if ga_kpis['load_balance_index'] < 0.5:
            report_lines.append("⚠ Poor load balance - consider improving route distribution")
        
        if not ga_kpis['is_feasible']:
            report_lines.append("⚠ GA solution has constraint violations")
        
        if not nn_kpis['is_feasible']:
            report_lines.append("⚠ Nearest Neighbor solution has constraint violations")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _generate_summary(self, ga_kpis: Dict, nn_kpis: Dict, comparison: Dict) -> Dict:
        """Generate report summary."""
        improvements = comparison['improvements']
        
        return {
            'ga_distance': ga_kpis['total_distance'],
            'nn_distance': nn_kpis['total_distance'],
            'distance_improvement_percent': improvements['distance_improvement_percent'],
            'ga_feasible': ga_kpis['is_feasible'],
            'nn_feasible': nn_kpis['is_feasible'],
            'ga_routes': ga_kpis['num_routes'],
            'nn_routes': nn_kpis['num_routes'],
            'overall_improved': improvements['is_improved']
        }
    
    def generate_quick_report(self, solution: Individual, 
                            solution_name: str = "Solution",
                            execution_time: Optional[float] = None) -> str:
        """
        Generate quick report for a single solution.
        
        Args:
            solution: Solution to report on
            solution_name: Name of the solution
            execution_time: Optional execution time
            
        Returns:
            Quick report string
        """
        kpis = self.kpi_calculator.calculate_kpis(solution, execution_time)
        
        report_lines = []
        report_lines.append(f"{solution_name} Quick Report")
        report_lines.append("=" * 40)
        report_lines.append(f"Total Distance: {kpis['total_distance']:.2f}")
        report_lines.append(f"Number of Routes: {kpis['num_routes']}")
        report_lines.append(f"Total Cost: {kpis['total_cost']:.2f}")
        report_lines.append(f"Average Utilization: {kpis['avg_utilization']:.1f}%")
        report_lines.append(f"Load Balance Index: {kpis['load_balance_index']:.3f}")
        report_lines.append(f"Efficiency Score: {kpis['efficiency_score']:.3f}")
        report_lines.append(f"Feasible: {kpis['is_feasible']}")
        report_lines.append(f"Fitness: {kpis['fitness']:.6f}")
        
        if execution_time:
            report_lines.append(f"Execution Time: {execution_time:.2f} seconds")
        
        return "\n".join(report_lines)
    
    def save_solution_data(self, solution: Individual, 
                         solution_name: str,
                         output_dir: str = "results") -> str:
        """
        Save solution data to file.
        
        Args:
            solution: Solution to save
            solution_name: Name for the solution
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{solution_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        solution_data = {
            'solution_name': solution_name,
            'timestamp': timestamp,
            'problem_info': self.problem.get_problem_info(),
            'solution': solution.to_dict(),
            'kpis': self.kpi_calculator.calculate_kpis(solution)
        }
        
        with open(filepath, 'w') as f:
            json.dump(solution_data, f, indent=2)
        
        return filepath


def generate_report(ga_solution: Individual, nn_solution: Individual,
                   problem: VRPProblem, ga_statistics: Dict,
                   convergence_data: Optional[Dict] = None,
                   output_dir: str = "results") -> Dict:
    """
    Convenience function to generate comprehensive report.
    
    Args:
        ga_solution: GA solution
        nn_solution: Nearest Neighbor solution
        problem: VRP problem instance
        ga_statistics: GA execution statistics
        convergence_data: Optional convergence data
        output_dir: Output directory
        
    Returns:
        Report summary
    """
    generator = ReportGenerator(problem)
    return generator.generate_comprehensive_report(
        ga_solution, nn_solution, ga_statistics, convergence_data, output_dir
    )
