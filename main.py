"""
Main application entry point for VRP-GA System.
Provides CLI interface for both Solomon dataset and mockup data modes.
"""

import argparse
import sys
import os
import time
from typing import Dict, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing.json_loader import JSONDatasetLoader
from src.data_processing.dataset_converter import DatasetConverter
from src.data_processing.generator import generate_mockup_data
from src.data_processing.distance import DistanceCalculator
from src.models.vrp_model import create_vrp_problem_from_dict
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.local_search import TwoOptOptimizer
from src.algorithms.nearest_neighbor import NearestNeighborHeuristic
from src.evaluation.metrics import KPICalculator
from src.evaluation.comparator import SolutionComparator
from src.visualization.reporter import ReportGenerator
from config import GA_CONFIG, VRP_CONFIG, MOCKUP_CONFIG


def main():
    """Main application entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Handle dataset management commands
        if args.list_datasets:
            list_datasets("all")
            return
        elif args.list_solomon:
            list_datasets("solomon")
            return
        elif args.list_mockup:
            list_datasets("mockup")
            return
        elif args.convert_solomon:
            convert_solomon_datasets()
            return
        elif args.create_samples:
            create_sample_datasets()
            return
        
        # Run appropriate mode
        if args.dataset:
            run_dataset_mode(args, dataset_type=None)
        elif args.solomon_dataset:
            run_dataset_mode(args, dataset_type="solomon")
        elif args.mockup_dataset:
            run_dataset_mode(args, dataset_type="mockup")
        elif args.generate:
            run_mockup_mode(args)
        else:
            # Check if any dataset management command was used
            if (args.list_datasets or args.list_solomon or args.list_mockup or 
                args.convert_solomon or args.create_samples):
                pass  # Already handled above
            else:
                parser.print_help()
                sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="VRP-GA System: Vehicle Routing Problem solver using Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all datasets
  python main.py --list-datasets
  
  # List Solomon datasets only
  python main.py --list-solomon
  
  # List mockup datasets only
  python main.py --list-mockup
  
  # Convert Solomon datasets to JSON
  python main.py --convert-solomon
  
  # Create sample datasets
  python main.py --create-samples
  
  # Solve Solomon dataset
  python main.py --solomon-dataset C101
  
  # Solve mockup dataset
  python main.py --mockup-dataset small_random
  
  # Auto-detect dataset type
  python main.py --dataset C101
  
  # Generate and solve mockup data
  python main.py --generate --customers 50 --capacity 200
  
  # Custom GA parameters
  python main.py --solomon-dataset C101 --generations 2000 --population 150
        """
    )
    
    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument('--dataset', type=str, 
                           help='Name of JSON dataset to load')
    data_group.add_argument('--solomon-dataset', type=str, 
                           help='Name of Solomon JSON dataset to load')
    data_group.add_argument('--mockup-dataset', type=str, 
                           help='Name of mockup JSON dataset to load')
    data_group.add_argument('--generate', action='store_true',
                           help='Generate mockup data')
    
    # Mockup generation options
    parser.add_argument('--customers', type=int, default=50,
                       help='Number of customers for mockup data (default: 50)')
    parser.add_argument('--capacity', type=float, default=200,
                       help='Vehicle capacity (default: 200)')
    parser.add_argument('--clustering', type=str, default='kmeans',
                       choices=['random', 'kmeans', 'radial'],
                       help='Clustering method for mockup data (default: kmeans)')
    parser.add_argument('--area', type=str, default='hanoi',
                       help='Area name for mockup data (default: hanoi)')
    
    # GA parameters
    parser.add_argument('--generations', type=int, default=1000,
                       help='Number of GA generations (default: 1000)')
    parser.add_argument('--population', type=int, default=100,
                       help='Population size (default: 100)')
    parser.add_argument('--crossover-prob', type=float, default=0.9,
                       help='Crossover probability (default: 0.9)')
    parser.add_argument('--mutation-prob', type=float, default=0.15,
                       help='Mutation probability (default: 0.15)')
    parser.add_argument('--tournament-size', type=int, default=5,
                       help='Tournament size (default: 5)')
    parser.add_argument('--elitism-rate', type=float, default=0.15,
                       help='Elitism rate (default: 0.15)')
    
    # VRP parameters
    parser.add_argument('--vehicles', type=int,
                       help='Number of vehicles (auto-determined if not specified)')
    parser.add_argument('--traffic-factor', type=float, default=1.0,
                       help='Traffic factor for distance calculation (default: 1.0)')
    
    # Dataset management options
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all available datasets')
    parser.add_argument('--list-solomon', action='store_true',
                       help='List Solomon datasets only')
    parser.add_argument('--list-mockup', action='store_true',
                       help='List mockup datasets only')
    parser.add_argument('--convert-solomon', action='store_true',
                       help='Convert all Solomon datasets to JSON format')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create sample mockup datasets')
    
    # Output options
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip generating report')
    parser.add_argument('--save-solution', action='store_true',
                       help='Save solution data to JSON')
    
    # Algorithm options
    parser.add_argument('--no-local-search', action='store_true',
                       help='Skip 2-opt local search optimization')
    parser.add_argument('--no-baseline', action='store_true',
                       help='Skip Nearest Neighbor baseline')
    
    # Debug options
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    
    return parser


def run_dataset_mode(args, dataset_type: str = None):
    """Run VRP solver with JSON dataset."""
    print("=" * 60)
    print("VRP-GA System - JSON Dataset Mode")
    print("=" * 60)
    
    # Determine dataset name
    if args.dataset:
        dataset_name = args.dataset
    elif args.solomon_dataset:
        dataset_name = args.solomon_dataset
    elif args.mockup_dataset:
        dataset_name = args.mockup_dataset
    else:
        raise ValueError("No dataset specified")
    
    # Load JSON dataset
    print(f"Loading JSON dataset: {dataset_name}")
    
    loader = JSONDatasetLoader()
    data, distance_matrix = loader.load_dataset_with_distance_matrix(
        dataset_name, args.traffic_factor, dataset_type
    )
    
    # Create VRP problem
    problem = create_vrp_problem_from_dict(data, distance_matrix)
    
    print(f"Problem loaded: {len(problem.customers)} customers, "
          f"capacity {problem.vehicle_capacity}, {problem.num_vehicles} vehicles")
    
    # Run optimization
    run_optimization(problem, args, data['metadata']['name'])


def run_mockup_mode(args):
    """Run VRP solver with generated mockup data."""
    print("=" * 60)
    print("VRP-GA System - Mockup Data Mode")
    print("=" * 60)
    
    # Generate mockup data
    print(f"Generating mockup data: {args.customers} customers, "
          f"capacity {args.capacity}, clustering {args.clustering}")
    
    # Set random seed if specified
    if args.seed:
        import numpy as np
        np.random.seed(args.seed)
    
    # Generate data
    data = generate_mockup_data(
        n_customers=args.customers,
        vehicle_capacity=args.capacity,
        clustering=args.clustering
    )
    
    # Override number of vehicles if specified
    if args.vehicles:
        data['num_vehicles'] = args.vehicles
    
    # Save generated data
    output_file = os.path.join(args.output, f"mockup_{args.customers}_customers.csv")
    os.makedirs(args.output, exist_ok=True)
    
    from src.data_processing.generator import MockupDataGenerator
    temp_generator = MockupDataGenerator()
    temp_generator.customers = data['customers']  # Keep as dictionaries
    temp_generator.depot = data['depot']  # Keep as dictionary
    temp_generator.vehicle_capacity = data['vehicle_capacity']
    temp_generator.num_vehicles = data['num_vehicles']
    temp_generator.export_to_csv(output_file)
    
    print(f"Generated data saved to: {output_file}")
    
    # Calculate distance matrix
    print("Calculating distance matrix...")
    distance_calculator = DistanceCalculator(args.traffic_factor)
    coordinates = [(data['depot']['x'], data['depot']['y'])]
    coordinates.extend([(c['x'], c['y']) for c in data['customers']])
    distance_matrix = distance_calculator.calculate_distance_matrix(coordinates)
    
    # Create VRP problem
    problem = create_vrp_problem_from_dict(data, distance_matrix)
    
    print(f"Problem created: {problem.get_problem_info()['num_customers']} customers, "
          f"capacity {problem.vehicle_capacity}, {problem.num_vehicles} vehicles")
    
    # Run optimization
    run_optimization(problem, args, "Mockup")


def create_customer_from_dict(customer_dict):
    """Create Customer object from dictionary."""
    from src.models.vrp_model import Customer
    return Customer(
        id=customer_dict['id'],
        x=customer_dict['x'],
        y=customer_dict['y'],
        demand=customer_dict['demand'],
        ready_time=customer_dict['ready_time'],
        due_date=customer_dict['due_date'],
        service_time=customer_dict['service_time']
    )


def create_depot_from_dict(depot_dict):
    """Create Depot object from dictionary."""
    from src.models.vrp_model import Depot
    return Depot(
        id=depot_dict['id'],
        x=depot_dict['x'],
        y=depot_dict['y'],
        demand=depot_dict['demand'],
        ready_time=depot_dict['ready_time'],
        due_date=depot_dict['due_date'],
        service_time=depot_dict['service_time']
    )


def run_optimization(problem, args, mode_name):
    """Run the optimization process."""
    # Set random seed if specified
    if args.seed:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Create GA configuration
    ga_config = GA_CONFIG.copy()
    ga_config.update({
        'generations': args.generations,
        'population_size': args.population,
        'crossover_prob': args.crossover_prob,
        'mutation_prob': args.mutation_prob,
        'tournament_size': args.tournament_size,
        'elitism_rate': args.elitism_rate
    })
    
    print(f"\nGA Configuration:")
    print(f"  Generations: {ga_config['generations']}")
    print(f"  Population Size: {ga_config['population_size']}")
    print(f"  Crossover Probability: {ga_config['crossover_prob']}")
    print(f"  Mutation Probability: {ga_config['mutation_prob']}")
    print(f"  Tournament Size: {ga_config['tournament_size']}")
    print(f"  Elitism Rate: {ga_config['elitism_rate']}")
    
    # Initialize GA
    print(f"\nInitializing Genetic Algorithm...")
    ga = GeneticAlgorithm(problem, ga_config)
    
    # Run GA
    print(f"Running GA for {ga_config['generations']} generations...")
    start_time = time.time()
    
    ga_solution = ga.evolve()
    ga_execution_time = time.time() - start_time
    
    print(f"GA completed in {ga_execution_time:.2f} seconds")
    
    # Apply 2-opt local search if not disabled
    if not args.no_local_search:
        print("Applying 2-opt local search optimization...")
        optimizer = TwoOptOptimizer(problem)
        ga_solution = optimizer.optimize_individual(ga_solution)
        print("2-opt optimization completed")
    
    # Run Nearest Neighbor baseline if not disabled
    nn_solution = None
    if not args.no_baseline:
        print("Running Nearest Neighbor baseline...")
        nn_heuristic = NearestNeighborHeuristic(problem)
        nn_solution = nn_heuristic.solve()
        print("Nearest Neighbor baseline completed")
    
    # Calculate KPIs
    print("Calculating KPIs...")
    kpi_calculator = KPICalculator(problem)
    ga_kpis = kpi_calculator.calculate_kpis(ga_solution, ga_execution_time)
    
    print(f"\nGA Solution Results:")
    print(f"  Total Distance: {ga_kpis['total_distance']:.2f}")
    print(f"  Number of Routes: {ga_kpis['num_routes']}")
    print(f"  Total Cost: {ga_kpis['total_cost']:.2f}")
    print(f"  Average Utilization: {ga_kpis['avg_utilization']:.1f}%")
    print(f"  Load Balance Index: {ga_kpis['load_balance_index']:.3f}")
    print(f"  Efficiency Score: {ga_kpis['efficiency_score']:.3f}")
    print(f"  Feasible: {ga_kpis['is_feasible']}")
    print(f"  Fitness: {ga_kpis['fitness']:.6f}")
    
    if nn_solution:
        nn_kpis = kpi_calculator.calculate_kpis(nn_solution)
        print(f"\nNearest Neighbor Results:")
        print(f"  Total Distance: {nn_kpis['total_distance']:.2f}")
        print(f"  Number of Routes: {nn_kpis['num_routes']}")
        print(f"  Total Cost: {nn_kpis['total_cost']:.2f}")
        print(f"  Average Utilization: {nn_kpis['avg_utilization']:.1f}%")
        print(f"  Load Balance Index: {nn_kpis['load_balance_index']:.3f}")
        print(f"  Efficiency Score: {nn_kpis['efficiency_score']:.3f}")
        print(f"  Feasible: {nn_kpis['is_feasible']}")
        print(f"  Fitness: {nn_kpis['fitness']:.6f}")
        
        # Calculate improvement
        distance_improvement = nn_kpis['total_distance'] - ga_kpis['total_distance']
        distance_improvement_percent = (distance_improvement / nn_kpis['total_distance']) * 100
        
        print(f"\nImprovement Analysis:")
        print(f"  Distance Improvement: {distance_improvement:.2f} ({distance_improvement_percent:.1f}%)")
        print(f"  Cost Improvement: {nn_kpis['total_cost'] - ga_kpis['total_cost']:.2f}")
        print(f"  Efficiency Improvement: {ga_kpis['efficiency_score'] - nn_kpis['efficiency_score']:.3f}")
    
    # Generate report and visualizations
    if not args.no_report or not args.no_plots:
        print(f"\nGenerating report and visualizations...")
        
        # Get GA statistics
        ga_statistics = ga.get_statistics()
        ga_statistics['execution_time'] = ga_execution_time
        
        # Get convergence data
        convergence_data = ga.get_convergence_data()
        
        # Generate report
        report_generator = ReportGenerator(problem)
        
        if nn_solution:
            report_summary = report_generator.generate_comprehensive_report(
                ga_solution, nn_solution, ga_statistics, convergence_data, args.output
            )
        else:
            # Generate report for GA solution only
            report_summary = report_generator.generate_quick_report(
                ga_solution, "GA Solution", ga_execution_time
            )
            print(f"Quick report generated")
        
        print(f"Report generated in: {report_summary.get('report_dir', 'N/A')}")
    
    # Save solution data if requested
    if args.save_solution:
        print("Saving solution data...")
        report_generator = ReportGenerator(problem)
        ga_file = report_generator.save_solution_data(ga_solution, "ga_solution", args.output)
        print(f"GA solution saved to: {ga_file}")
        
        if nn_solution:
            nn_file = report_generator.save_solution_data(nn_solution, "nn_solution", args.output)
            print(f"NN solution saved to: {nn_file}")
    
    print(f"\nOptimization completed successfully!")
    print(f"Results saved in: {args.output}")


def list_datasets(dataset_type: str = "all"):
    """List all available datasets."""
    print("=" * 60)
    print(f"AVAILABLE DATASETS ({dataset_type.upper()})")
    print("=" * 60)
    
    loader = JSONDatasetLoader()
    datasets = loader.list_available_datasets(dataset_type)
    
    if not datasets:
        print(f"No {dataset_type} datasets found. Run --convert-solomon or --create-samples first.")
        return
    
    print(f"Found {len(datasets)} datasets:\n")
    
    # Group by type for better display
    solomon_datasets = [d for d in datasets if d['type'] == 'solomon']
    mockup_datasets = [d for d in datasets if d['type'] == 'mockup']
    
    if solomon_datasets:
        print("SOLOMON DATASETS:")
        for dataset in solomon_datasets:
            print(f"  {dataset['name']}")
            print(f"    Description: {dataset['metadata']['description']}")
            print(f"    Customers: {dataset['num_customers']}")
            print(f"    Capacity: {dataset['vehicle_capacity']}")
            print(f"    Vehicles: {dataset['num_vehicles']}")
            print()
    
    if mockup_datasets:
        print("MOCKUP DATASETS:")
        for dataset in mockup_datasets:
            print(f"  {dataset['name']}")
            print(f"    Description: {dataset['metadata']['description']}")
            print(f"    Customers: {dataset['num_customers']}")
            print(f"    Capacity: {dataset['vehicle_capacity']}")
            print(f"    Vehicles: {dataset['num_vehicles']}")
            print()


def convert_solomon_datasets():
    """Convert all Solomon datasets to JSON format."""
    print("=" * 60)
    print("CONVERTING SOLOMON DATASETS TO JSON")
    print("=" * 60)
    
    converter = DatasetConverter()
    converter.convert_all_solomon_datasets()
    converter.create_dataset_catalog()
    
    print("\nConversion completed!")


def create_sample_datasets():
    """Create sample mockup datasets."""
    print("=" * 60)
    print("CREATING SAMPLE DATASETS")
    print("=" * 60)
    
    converter = DatasetConverter()
    
    # Create various sample datasets
    datasets = [
        (10, 50, 'random', 'small_random'),
        (20, 100, 'kmeans', 'medium_kmeans'),
        (30, 150, 'radial', 'medium_radial'),
        (50, 200, 'kmeans', 'large_kmeans'),
        (100, 300, 'kmeans', 'xlarge_kmeans')
    ]
    
    for n_customers, capacity, clustering, name in datasets:
        converter.create_mockup_dataset(
            n_customers=n_customers,
            vehicle_capacity=capacity,
            clustering=clustering,
            name=name
        )
    
    converter.create_dataset_catalog()
    print("\nSample datasets created!")


if __name__ == "__main__":
    main()
