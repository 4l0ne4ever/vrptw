"""
Result export module for VRP-GA system.
Exports detailed results in various formats for analysis.
"""

import csv
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem


class ResultExporter:
    """Exports VRP-GA results in various formats."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize result exporter.
        
        Args:
            output_dir: Output directory for results
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def export_evolution_data(self, evolution_data: List[Dict], 
                            filename: Optional[str] = None) -> str:
        """
        Export GA evolution data to CSV.
        
        Args:
            evolution_data: List of generation data
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"evolution_data_{self.timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data for CSV
        csv_data = []
        for gen_data in evolution_data:
            csv_data.append({
                'generation': gen_data.get('generation', 0),
                'evaluated_individuals': gen_data.get('evaluated_individuals', 0),
                'min_fitness': gen_data.get('min_fitness', 0),
                'max_fitness': gen_data.get('max_fitness', 0),
                'avg_fitness': gen_data.get('avg_fitness', 0),
                'std_fitness': gen_data.get('std_fitness', 0),
                'best_distance': gen_data.get('best_distance', 0),
                'avg_distance': gen_data.get('avg_distance', 0),
                'diversity': gen_data.get('diversity', 0)
            })
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        print(f"Evolution data exported to: {filepath}")
        return filepath
    
    def export_optimal_routes(self, individual: Individual, problem: VRPProblem,
                            filename: Optional[str] = None) -> str:
        """
        Export optimal routes to text file.
        
        Args:
            individual: Best GA solution
            problem: VRP problem instance
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"optimal_routes_{self.timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Decode routes
        from src.algorithms.decoder import RouteDecoder
        decoder = RouteDecoder(problem)
        routes = decoder.decode_chromosome(individual.chromosome)
        
        # Calculate route details
        route_details = []
        total_distance = 0
        total_shipping_cost = 0
        
        # Calculate shipping cost for all routes
        from src.evaluation.shipping_cost import ShippingCostCalculator
        shipping_calculator = ShippingCostCalculator(cost_model="ahamove")
        
        # Generate order values and waiting times
        order_values = shipping_calculator.generate_order_values(problem.customers)
        waiting_times = shipping_calculator.generate_waiting_times(problem.customers)
        
        # Calculate shipping cost for entire solution
        shipping_cost_data = shipping_calculator.calculate_solution_cost(
            routes, problem, service_type="express", 
            order_values=order_values, waiting_times=waiting_times
        )
        
        for i, route in enumerate(routes):
            if not route:
                continue
                
            route_distance = 0
            route_load = 0
            route_customers = []
            
            # Calculate route metrics
            for j in range(len(route) - 1):
                current_id = route[j]
                next_id = route[j + 1]
                
                if current_id == 0:  # Depot
                    route_customers.append("Depot")
                else:
                    customer = problem.get_customer_by_id(current_id)
                    route_load += customer.demand
                    route_customers.append(f"KH_{current_id}")
                
                # Calculate distance
                if current_id == 0:
                    current_coords = (problem.depot.x, problem.depot.y)
                else:
                    current_customer = problem.get_customer_by_id(current_id)
                    current_coords = (current_customer.x, current_customer.y)
                
                if next_id == 0:
                    next_coords = (problem.depot.x, problem.depot.y)
                else:
                    next_customer = problem.get_customer_by_id(next_id)
                    next_coords = (next_customer.x, next_customer.y)
                
                distance = problem.get_distance(current_id, next_id)
                route_distance += distance
            
            # Add final depot
            route_customers.append("Depot")
            
            # Get shipping cost for this route
            route_shipping_cost = 0
            if i < len(shipping_cost_data['route_costs']):
                route_shipping_cost = shipping_cost_data['route_costs'][i]['cost_breakdown']['total_cost']
            
            route_details.append({
                'vehicle_id': i + 1,
                'customers': route_customers,
                'distance': route_distance,
                'load': route_load,
                'capacity': problem.vehicle_capacity,
                'utilization': (route_load / problem.vehicle_capacity) * 100,
                'shipping_cost': route_shipping_cost
            })
            
            total_distance += route_distance
            total_shipping_cost += route_shipping_cost
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=== LỘ TRÌNH TỐI ƯU GA ===\n\n")
            f.write(f"Tổng số xe sử dụng: {len(route_details)}\n")
            f.write(f"Tổng quãng đường: {total_distance:.2f} km\n")
            f.write(f"Tổng chi phí vận chuyển: {total_distance:.2f}\n")
            f.write(f"Tổng phí giao hàng: {total_shipping_cost:,.0f} VND\n")
            f.write(f"Dịch vụ: {shipping_cost_data['service_type']}\n\n")
            
            for route in route_details:
                f.write(f"Xe {route['vehicle_id']}:\n")
                f.write(f"  Lộ trình: {' -> '.join(route['customers'])}\n")
                f.write(f"  Quãng đường: {route['distance']:.2f} km\n")
                f.write(f"  Tải trọng: {route['load']:.1f}/{route['capacity']} ({route['utilization']:.1f}%)\n")
                f.write(f"  Phí giao hàng: {route['shipping_cost']:,.0f} VND\n\n")
        
        print(f"Optimal routes exported to: {filepath}")
        return filepath
    
    def export_kpi_comparison(self, ga_solution: Individual, nn_solution: Individual,
                            problem: VRPProblem, ga_statistics: Dict, nn_statistics: Dict,
                            filename: Optional[str] = None) -> str:
        """
        Export KPI comparison between GA and NN.
        
        Args:
            ga_solution: GA solution
            nn_solution: NN solution
            ga_statistics: GA execution statistics
            nn_statistics: NN execution statistics
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"kpi_comparison_{self.timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Calculate KPIs
        from src.evaluation.metrics import KPICalculator
        kpi_calculator = KPICalculator(problem)
        
        ga_kpis = kpi_calculator.calculate_kpis(ga_solution, ga_statistics.get('execution_time', 0))
        nn_kpis = kpi_calculator.calculate_kpis(nn_solution, nn_statistics.get('execution_time', 0))
        
        # Calculate improvements
        improvements = {}
        for key in ga_kpis:
            if key in nn_kpis and nn_kpis[key] != 0:
                # Only calculate improvement for numeric values
                if isinstance(ga_kpis[key], (int, float)) and isinstance(nn_kpis[key], (int, float)):
                    if key in ['total_distance', 'total_cost', 'execution_time']:
                        # Lower is better
                        improvement = ((nn_kpis[key] - ga_kpis[key]) / nn_kpis[key]) * 100
                    else:
                        # Higher is better
                        improvement = ((ga_kpis[key] - nn_kpis[key]) / nn_kpis[key]) * 100
                    improvements[key] = improvement
        
        # Prepare CSV data
        csv_data = []
        
        # Add GA data
        ga_row = {'method': 'GA'}
        for key, value in ga_kpis.items():
            ga_row[key] = value
        csv_data.append(ga_row)
        
        # Add NN data
        nn_row = {'method': 'Nearest Neighbor'}
        for key, value in nn_kpis.items():
            nn_row[key] = value
        csv_data.append(nn_row)
        
        # Add improvement data
        improvement_row = {'method': 'Improvement (%)'}
        for key, value in improvements.items():
            improvement_row[key] = f"{value:.2f}%"
        csv_data.append(improvement_row)
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        print(f"KPI comparison exported to: {filepath}")
        return filepath
    
    def export_sensitivity_analysis(self, sensitivity_data: List[Dict],
                                   filename: Optional[str] = None) -> str:
        """
        Export sensitivity analysis results.
        
        Args:
            sensitivity_data: List of sensitivity test results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"sensitivity_analysis_{self.timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if sensitivity_data:
                writer = csv.DictWriter(f, fieldnames=sensitivity_data[0].keys())
                writer.writeheader()
                writer.writerows(sensitivity_data)
        
        print(f"Sensitivity analysis exported to: {filepath}")
        return filepath
    
    def export_solomon_summary(self, solomon_results: List[Dict],
                             filename: Optional[str] = None) -> str:
        """
        Export Solomon dataset summary comparison.
        
        Args:
            solomon_results: List of results from different Solomon datasets
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"solomon_summary_{self.timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if solomon_results:
                writer = csv.DictWriter(f, fieldnames=solomon_results[0].keys())
                writer.writeheader()
                writer.writerows(solomon_results)
        
        print(f"Solomon summary exported to: {filepath}")
        return filepath


def export_all_results(ga_solution: Individual, nn_solution: Individual,
                      problem: VRPProblem, ga_statistics: Dict,
                      nn_statistics: Dict, evolution_data: List[Dict],
                      output_dir: str = "results") -> Dict[str, str]:
    """
    Export all result files.
    
    Args:
        ga_solution: GA solution
        nn_solution: NN solution
        problem: VRP problem instance
        ga_statistics: GA execution statistics
        nn_statistics: NN execution statistics
        evolution_data: GA evolution data
        output_dir: Output directory
        
    Returns:
        Dictionary of exported file paths
    """
    exporter = ResultExporter(output_dir)
    
    exported_files = {}
    
    # Export evolution data
    exported_files['evolution'] = exporter.export_evolution_data(evolution_data)
    
    # Export optimal routes
    exported_files['routes'] = exporter.export_optimal_routes(ga_solution, problem)
    
    # Export KPI comparison
    exported_files['kpi'] = exporter.export_kpi_comparison(
        ga_solution, nn_solution, problem, ga_statistics, nn_statistics
    )
    
    return exported_files
