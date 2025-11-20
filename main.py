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
from src.algorithms.decoder import RouteDecoder
from src.evaluation.comparator import SolutionComparator
from src.evaluation.result_exporter import ResultExporter
from src.evaluation.bks_validator import BKSValidator
from src.visualization.reporter import ReportGenerator
from src.data_processing.constraints import ConstraintHandler
from src.core.logger import setup_logger
from src.core.exceptions import (
    VRPException, DatasetNotFoundError, InvalidConfigurationError,
    InfeasibleSolutionError, CapacityViolationError
)
from config import GA_CONFIG, VRP_CONFIG, MOCKUP_CONFIG


def main():
    """Main application entry point."""
    # Setup logger
    logger = setup_logger('vrp_ga', log_dir='logs')
    logger.info("="*60)
    logger.info("VRP-GA System Starting")
    logger.info("="*60)
    
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
        elif args.solomon_batch:
            run_solomon_batch(args)
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
        logger.info("Operation cancelled by user")
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except (DatasetNotFoundError, InvalidConfigurationError, InfeasibleSolutionError) as e:
        logger.error(f"VRP Error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
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
  
  # Run all Solomon datasets in batch
  python main.py --solomon-batch --generations 100 --population 50
  
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
    
    # Batch processing
    parser.add_argument('--solomon-batch', action='store_true',
                       help='Run all Solomon datasets in batch mode')
    
    # Output options
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--debug-constraints', action='store_true',
                       help='Write detailed constraint analysis to results')
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
    logger = setup_logger('vrp_ga.dataset')
    logger.info("=" * 60)
    logger.info("VRP-GA System - JSON Dataset Mode")
    logger.info("=" * 60)
    
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
    logger.info(f"Loading JSON dataset: {dataset_name}")
    print(f"Loading JSON dataset: {dataset_name}")
    
    try:
        loader = JSONDatasetLoader()
        data, distance_matrix = loader.load_dataset_with_distance_matrix(
            dataset_name, args.traffic_factor, dataset_type
        )
    except FileNotFoundError as e:
        raise DatasetNotFoundError(dataset_name, dataset_type) from e
    
    # Create VRP problem
    problem = create_vrp_problem_from_dict(data, distance_matrix)
    
    logger.info(f"Problem loaded: {len(problem.customers)} customers, "
               f"capacity {problem.vehicle_capacity}, {problem.num_vehicles} vehicles")
    print(f"Problem loaded: {len(problem.customers)} customers, "
          f"capacity {problem.vehicle_capacity}, {problem.num_vehicles} vehicles")
    
    # Run optimization
    run_optimization(problem, args, data['metadata']['name'])


def run_mockup_mode(args):
    """Run VRP solver with generated mockup data."""
    logger = setup_logger('vrp_ga.mockup')
    logger.info("=" * 60)
    logger.info("VRP-GA System - Mockup Data Mode")
    logger.info("=" * 60)
    
    print("=" * 60)
    print("VRP-GA System - Mockup Data Mode")
    print("=" * 60)
    
    # Generate mockup data
    logger.info(f"Generating mockup data: {args.customers} customers, "
               f"capacity {args.capacity}, clustering {args.clustering}")
    print(f"Generating mockup data: {args.customers} customers, "
          f"capacity {args.capacity}, clustering {args.clustering}")
    
    # Set random seed if specified
    if args.seed:
        import numpy as np
        np.random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
    
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
    
    logger.info(f"Generated data saved to: {output_file}")
    print(f"Generated data saved to: {output_file}")
    
    # Calculate distance matrix
    logger.info("Calculating distance matrix...")
    print("Calculating distance matrix...")
    distance_calculator = DistanceCalculator(args.traffic_factor)
    coordinates = [(data['depot']['x'], data['depot']['y'])]
    coordinates.extend([(c['x'], c['y']) for c in data['customers']])
    distance_matrix = distance_calculator.calculate_distance_matrix(coordinates)
    
    # Create VRP problem
    problem = create_vrp_problem_from_dict(data, distance_matrix)
    
    logger.info(f"Problem created: {problem.get_problem_info()['num_customers']} customers, "
               f"capacity {problem.vehicle_capacity}, {problem.num_vehicles} vehicles")
    print(f"Problem created: {problem.get_problem_info()['num_customers']} customers, "
          f"capacity {problem.vehicle_capacity}, {problem.num_vehicles} vehicles")
    
    # Run optimization
    run_optimization(problem, args, "Mockup")


def run_solomon_batch(args):
    """Run VRP solver on all Solomon datasets individually."""
    logger = setup_logger('vrp_ga.batch')
    logger.info("=" * 60)
    logger.info("VRP-GA System - Solomon Batch Mode")
    logger.info("=" * 60)
    
    print("=" * 60)
    print("VRP-GA System - Solomon Batch Mode")
    print("=" * 60)
    
    # Initialize BKS validator
    bks_validator = BKSValidator('data/solomon_bks.json')
    num_instances = len(bks_validator.bks_data)
    logger.info(f"BKS validator initialized with {num_instances} instances")
    
    # Load dataset catalog
    loader = JSONDatasetLoader()
    solomon_datasets = loader.list_available_datasets('solomon')
    
    if not solomon_datasets:
        logger.warning("No Solomon datasets found!")
        print("No Solomon datasets found!")
        return
    
    logger.info(f"Found {len(solomon_datasets)} Solomon datasets")
    print(f"Found {len(solomon_datasets)} Solomon datasets")
    
    # Results storage
    batch_results = []
    
    # Run each dataset
    for i, dataset_info in enumerate(solomon_datasets):
        dataset_name = dataset_info['name']
        logger.info(f"Processing dataset {i+1}/{len(solomon_datasets)}: {dataset_name}")
        print(f"\n{'='*50}")
        print(f"Running dataset {i+1}/{len(solomon_datasets)}: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Load dataset
            data, distance_matrix = loader.load_dataset_with_distance_matrix(
                dataset_name, args.traffic_factor, 'solomon'
            )
            
            # Create VRP problem
            problem = create_vrp_problem_from_dict(data, distance_matrix)
            
            logger.info(f"Problem: {len(problem.customers)} customers, "
                       f"capacity {problem.vehicle_capacity}, {problem.num_vehicles} vehicles")
            print(f"Problem: {len(problem.customers)} customers, "
                  f"capacity {problem.vehicle_capacity}, {problem.num_vehicles} vehicles")
            
            # Run optimization
            ga_solution, evolution_data = run_single_optimization(problem, args)
            
            # Calculate KPIs
            kpi_calculator = KPICalculator(problem)
            ga_kpis = kpi_calculator.calculate_kpis(ga_solution)
            
            # BKS Validation
            bks_validation = bks_validator.validate_solution(dataset_name, ga_solution)
            
            # Store results
            result = {
                'dataset': dataset_name,
                'customers': len(problem.customers),
                'capacity': problem.vehicle_capacity,
                'vehicles': problem.num_vehicles,
                'ga_distance': ga_kpis['total_distance'],
                'ga_cost': ga_kpis['total_cost'],
                'ga_routes': ga_kpis['num_routes'],
                'ga_utilization': ga_kpis['avg_utilization'],
                'ga_efficiency': ga_kpis['efficiency_score'],
                'ga_feasible': ga_kpis['is_feasible'],
                'generations': args.generations,
                'population': args.population
            }
            
            # Add BKS data if available
            if bks_validation['has_bks']:
                result['bks_distance'] = bks_validation['bks_distance']
                result['bks_vehicles'] = bks_validation['bks_vehicles']
                result['gap_percent'] = bks_validation['gap_percent']
                result['vehicle_diff'] = bks_validation['vehicle_diff']
                result['quality'] = bks_validation['quality']
                
                logger.info(f"BKS Validation - Gap: {bks_validation['gap_percent']:.2f}%, "
                          f"Quality: {bks_validation['quality']}")
            
            batch_results.append(result)
            
            # Print results with BKS info if available
            if bks_validation['has_bks']:
                print(f"Completed: Distance={ga_kpis['total_distance']:.2f} "
                      f"(BKS: {bks_validation['bks_distance']:.2f}, "
                      f"Gap: {bks_validation['gap_percent']:.2f}%), "
                      f"Routes={ga_kpis['num_routes']} "
                      f"(BKS: {bks_validation['bks_vehicles']}), "
                      f"Quality: {bks_validation['quality']}, "
                      f"Utilization={ga_kpis['avg_utilization']:.1f}%")
            else:
                print(f"Completed: Distance={ga_kpis['total_distance']:.2f}, "
                      f"Routes={ga_kpis['num_routes']}, "
                      f"Utilization={ga_kpis['avg_utilization']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}", exc_info=True)
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    # Export batch summary
    if batch_results:
        exporter = ResultExporter(args.output)
        summary_file = exporter.export_solomon_summary(batch_results)
        logger.info(f"Batch summary exported to: {summary_file}")
        print(f"\nBatch summary exported to: {summary_file}")
        
        # Print summary statistics
        print(f"\nBatch Summary:")
        print(f"  Total datasets processed: {len(batch_results)}")
        
        # Calculate average distance
        avg_distance = sum(r['ga_distance'] for r in batch_results) / len(batch_results)
        print(f"  Average distance: {avg_distance:.2f}")
        
        # Calculate average routes
        avg_routes = sum(r['ga_routes'] for r in batch_results) / len(batch_results)
        print(f"  Average routes: {avg_routes:.1f}")
        
        # Calculate average utilization
        avg_utilization = sum(r['ga_utilization'] for r in batch_results) / len(batch_results)
        print(f"  Average utilization: {avg_utilization:.1f}%")
        
        # BKS statistics if available
        bks_results = [r for r in batch_results if 'gap_percent' in r and r['gap_percent'] is not None]
        if bks_results:
            avg_gap = sum(r['gap_percent'] for r in bks_results) / len(bks_results)
            print(f"  BKS Comparison (for {len(bks_results)} instances with BKS):")
            print(f"    Average gap from BKS: {avg_gap:.2f}%")
            
            # Quality distribution
            quality_counts = {}
            for r in bks_results:
                quality = r.get('quality', 'UNKNOWN')
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            print(f"    Quality distribution:")
            for quality, count in sorted(quality_counts.items()):
                print(f"      {quality}: {count}")
        
        logger.info(f"Batch processing completed: {len(batch_results)} datasets processed")


def run_single_optimization(problem, args):
    """Run optimization for a single problem (without full reporting)."""
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
    
    # Initialize and run GA
    ga = GeneticAlgorithm(problem, ga_config)
    ga_solution, evolution_data = ga.evolve()
    
    # Apply 2-opt if not disabled
    if not args.no_local_search:
        optimizer = TwoOptOptimizer(problem)
        ga_solution = optimizer.optimize_individual(ga_solution)
    
    return ga_solution, evolution_data


def run_optimization(problem, args, mode_name):
    """Run the optimization process."""
    logger = setup_logger('vrp_ga.optimization')
    logger.info(f"Starting optimization for: {mode_name}")
    
    # Set random seed if specified
    if args.seed:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
    
    # Validate GA configuration
    from src.core.validators import ConfigValidator
    ga_config = GA_CONFIG.copy()
    ga_config.update({
        'generations': args.generations,
        'population_size': args.population,
        'crossover_prob': args.crossover_prob,
        'mutation_prob': args.mutation_prob,
        'tournament_size': args.tournament_size,
        'elitism_rate': args.elitism_rate
    })
    
    try:
        ConfigValidator.validate_ga_config(ga_config)
    except InvalidConfigurationError as e:
        logger.error(f"Invalid GA configuration: {e}")
        raise
    
    logger.info(f"GA Configuration: generations={ga_config['generations']}, "
               f"population={ga_config['population_size']}, "
               f"crossover_prob={ga_config['crossover_prob']}, "
               f"mutation_prob={ga_config['mutation_prob']}")
    print(f"\nGA Configuration:")
    print(f"  Generations: {ga_config['generations']}")
    print(f"  Population Size: {ga_config['population_size']}")
    print(f"  Crossover Probability: {ga_config['crossover_prob']}")
    print(f"  Mutation Probability: {ga_config['mutation_prob']}")
    print(f"  Tournament Size: {ga_config['tournament_size']}")
    print(f"  Elitism Rate: {ga_config['elitism_rate']}")
    
    # Initialize GA
    logger.info("Initializing Genetic Algorithm...")
    print(f"\nInitializing Genetic Algorithm...")
    ga = GeneticAlgorithm(problem, ga_config)
    
    # Run GA
    logger.info(f"Running GA for {ga_config['generations']} generations...")
    print(f"Running GA for {ga_config['generations']} generations...")
    start_time = time.time()
    
    ga_solution, evolution_data = ga.evolve()
    ga_execution_time = time.time() - start_time
    
    logger.info(f"GA completed in {ga_execution_time:.2f} seconds")
    print(f"GA completed in {ga_execution_time:.2f} seconds")
    
    # Apply final repair to ensure solution is feasible
    # Use Split Algorithm to match GA behavior
    decoder = RouteDecoder(problem, use_split_algorithm=GA_CONFIG.get('use_split_algorithm', False))
    # Use routes from solution if available, otherwise decode from chromosome
    if ga_solution.routes:
        routes = ga_solution.routes
    else:
        routes = decoder.decode_chromosome(ga_solution.chromosome)
    demands = [c.demand for c in problem.customers]
    # Ensure demands array covers all possible customer IDs (1 to N)
    max_customer_id = max(max(route) for route in routes) if routes else 0
    while len(demands) < max_customer_id:
        demands.append(0.0)  # Add zero demand for missing customers
    
    constraint_handler = ConstraintHandler(problem.vehicle_capacity, problem.num_vehicles)
    cap_valid, _ = constraint_handler.validate_capacity_constraint(routes, demands)
    if not cap_valid:
        print("Final repair: Capacity violations detected, repairing...")
        routes = constraint_handler.repair_capacity_violations(routes, demands)
        ga_solution.routes = routes
        ga_solution.chromosome = decoder.encode_routes(routes)
        # Recalculate total distance after repair
        total_distance = 0.0
        for route in routes:
            if not route:
                continue
            for i in range(len(route) - 1):
                total_distance += problem.get_distance(route[i], route[i + 1])
        ga_solution.total_distance = total_distance
        print("Final repair: Completed")
    else:
        print("Final repair: No capacity violations found")
    
    # Optional TW repair on final solution (post-processing)
    tw_cfg = GA_CONFIG.get('tw_repair', {})
    if (
        tw_cfg.get('enabled', False)
        and tw_cfg.get('apply_on_final_solution', False)
        and ga_solution.routes
    ):
        try:
            from src.algorithms.tw_repair import TWRepairOperator
            tw_repair = TWRepairOperator(
                problem,
                max_iterations=tw_cfg.get('max_iterations', 50),
                violation_weight=tw_cfg.get('violation_weight', 50.0),
                max_relocations_per_route=tw_cfg.get('max_relocations_per_route', 2),
                max_routes_to_try=tw_cfg.get('max_routes_to_try', None),
                max_positions_to_try=tw_cfg.get('max_positions_to_try', None),
                max_iterations_soft=tw_cfg.get('max_iterations_soft'),
                max_routes_soft_limit=tw_cfg.get('max_routes_soft_limit'),
                max_positions_soft_limit=tw_cfg.get('max_positions_soft_limit'),
                lateness_soft_threshold=tw_cfg.get('lateness_soft_threshold'),
                lateness_skip_threshold=tw_cfg.get('lateness_skip_threshold'),
            )
            repaired_routes = tw_repair.repair_routes(ga_solution.routes)
            ga_solution.routes = repaired_routes
            ga_solution.chromosome = decoder.encode_routes(repaired_routes)
        except Exception as tw_err:
            logger.warning(f"Final TW repair skipped: {tw_err}")
    
    # Apply 2-opt local search if not disabled
    if not args.no_local_search:
        print("Applying 2-opt local search optimization...")
        optimizer = TwoOptOptimizer(problem)
        ga_solution = optimizer.optimize_individual(ga_solution)
        
        # Repair any capacity violations caused by 2-opt
        # Use Split Algorithm to match GA behavior
        decoder = RouteDecoder(problem, use_split_algorithm=GA_CONFIG.get('use_split_algorithm', False))
        # Use routes from solution if available, otherwise decode from chromosome
        if ga_solution.routes:
            routes = ga_solution.routes
        else:
            routes = decoder.decode_chromosome(ga_solution.chromosome)
        demands = [c.demand for c in problem.customers]
        # Ensure demands array covers all possible customer IDs (1 to N)
        max_customer_id = max(max(route) for route in routes) if routes else 0
        while len(demands) < max_customer_id:
            demands.append(0.0)  # Add zero demand for missing customers
        
        constraint_handler = ConstraintHandler(problem.vehicle_capacity, problem.num_vehicles)
        cap_valid, _ = constraint_handler.validate_capacity_constraint(routes, demands)
        if not cap_valid:
            print("Post-2opt repair: Capacity violations detected, repairing...")
            routes = constraint_handler.repair_capacity_violations(routes, demands)
            ga_solution.routes = routes
            ga_solution.chromosome = decoder.encode_routes(routes)
            # Recalculate total distance after repair
            total_distance = 0.0
            for route in routes:
                if not route:
                    continue
                for i in range(len(route) - 1):
                    total_distance += problem.get_distance(route[i], route[i + 1])
            ga_solution.total_distance = total_distance
            print("Post-2opt repair: Completed")
        else:
            print("Post-2opt repair: No capacity violations found")
        
        print("2-opt optimization completed")

    # Capacity repair is handled early in FitnessEvaluator; skip here to avoid double-repair
    
    # Run Nearest Neighbor baseline if not disabled
    nn_solution = None
    nn_execution_time = None
    if not args.no_baseline:
        print("Running Nearest Neighbor baseline...")
        nn_heuristic = NearestNeighborHeuristic(problem)
        nn_start_time = time.time()
        nn_solution = nn_heuristic.solve()
        nn_execution_time = time.time() - nn_start_time
        print("Nearest Neighbor baseline completed")
    
    # Calculate KPIs
    logger.info("Calculating KPIs...")
    print("Calculating KPIs...")
    kpi_calculator = KPICalculator(problem)
    ga_kpis = kpi_calculator.calculate_kpis(ga_solution, ga_execution_time)
    
    # BKS Validation for Solomon datasets
    if mode_name and 'solomon' in mode_name.lower():
        bks_validator = BKSValidator('data/solomon_bks.json')
        # Try to extract instance name from mode_name or dataset
        instance_name = getattr(args, 'solomon_dataset', None) or mode_name
        if instance_name:
            bks_validation = bks_validator.validate_solution(instance_name, ga_solution)
            if bks_validation['has_bks']:
                logger.info(f"BKS Validation - Instance: {bks_validation['instance']}, "
                          f"Gap: {bks_validation['gap_percent']:.2f}%, "
                          f"Quality: {bks_validation['quality']}")
                print(f"\nBKS Comparison:")
                print(f"  Instance: {bks_validation['instance']}")
                print(f"  Solution Distance: {bks_validation['solution_distance']:.2f}")
                print(f"  BKS Distance: {bks_validation['bks_distance']:.2f}")
                print(f"  Gap: {bks_validation['gap_percent']:.2f}%")
                print(f"  Quality: {bks_validation['quality']}")
                if bks_validation['vehicle_diff'] is not None:
                    print(f"  Vehicle Difference: {bks_validation['vehicle_diff']} "
                          f"(Solution: {bks_validation['solution_vehicles']}, "
                          f"BKS: {bks_validation['bks_vehicles']})")
    
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
        nn_kpis = kpi_calculator.calculate_kpis(nn_solution, nn_execution_time)
        print(f"\nNearest Neighbor Results:")
        print(f"  Total Distance: {nn_kpis['total_distance']:.2f}")
        print(f"  Number of Routes: {nn_kpis['num_routes']}")
        print(f"  Total Cost: {nn_kpis['total_cost']:.2f}")
        print(f"  Average Utilization: {nn_kpis['avg_utilization']:.1f}%")
        print(f"  Load Balance Index: {nn_kpis['load_balance_index']:.3f}")
        print(f"  Efficiency Score: {nn_kpis['efficiency_score']:.3f}")
        print(f"  Feasible: {nn_kpis['is_feasible']}")
        print(f"  Fitness: {nn_kpis['fitness']:.6f}")
        print(f"  Execution Time: {nn_kpis['execution_time']:.3f}s")
        
        # Calculate improvement
        distance_improvement = nn_kpis['total_distance'] - ga_kpis['total_distance']
        distance_improvement_percent = (distance_improvement / nn_kpis['total_distance']) * 100
        
        print(f"\nImprovement Analysis:")
        print(f"  Distance Improvement: {distance_improvement:.2f} ({distance_improvement_percent:.1f}%)")
        print(f"  Cost Improvement: {nn_kpis['total_cost'] - ga_kpis['total_cost']:.2f}")
        print(f"  Efficiency Improvement: {ga_kpis['efficiency_score'] - nn_kpis['efficiency_score']:.3f}")
    
    # Export detailed results
    logger.info("Exporting detailed results...")
    print(f"\nExporting detailed results...")
    
    # Create result exporter
    exporter = ResultExporter(args.output)
    
    # Export evolution data
    evolution_file = exporter.export_evolution_data(evolution_data)
    
    # Export optimal routes
    routes_file = exporter.export_optimal_routes(ga_solution, problem)
    
    # Export KPI comparison if NN solution exists
    if nn_solution:
        ga_stats = ga.get_statistics()
        ga_stats['execution_time'] = ga_execution_time
        
        nn_stats = {'execution_time': nn_execution_time or 0.0}
        
        # Add BKS data if available for Solomon datasets
        if mode_name and 'solomon' in mode_name.lower():
            instance_name = getattr(args, 'solomon_dataset', None) or mode_name
            if instance_name:
                bks_validator = BKSValidator('data/solomon_bks.json')
                bks_validation = bks_validator.validate_solution(instance_name, ga_solution)
                if bks_validation['has_bks']:
                    ga_stats['bks_distance'] = bks_validation['bks_distance']
                    ga_stats['bks_gap_percent'] = bks_validation['gap_percent']
                    ga_stats['bks_quality'] = bks_validation['quality']
        
        kpi_file = exporter.export_kpi_comparison(
            ga_solution, nn_solution, problem, ga_stats, nn_stats
        )
        
        logger.info(f"Results exported: evolution={evolution_file}, routes={routes_file}, kpi={kpi_file}")
        print(f"Results exported:")
        print(f"  Evolution data: {evolution_file}")
        print(f"  Optimal routes: {routes_file}")
        print(f"  KPI comparison: {kpi_file}")
    else:
        logger.info(f"Results exported: evolution={evolution_file}, routes={routes_file}")
        print(f"Results exported:")
        print(f"  Evolution data: {evolution_file}")
        print(f"  Optimal routes: {routes_file}")
    
    # Constraint debug output (optional)
    if args.debug_constraints:
        try:
            os.makedirs(args.output, exist_ok=True)
            ch = ConstraintHandler(problem.vehicle_capacity, problem.num_vehicles)
            decoder = RouteDecoder(problem)
            # Use the repaired routes from the solution, not decode from chromosome
            ga_routes = ga_solution.routes if ga_solution.routes else decoder.decode_chromosome(ga_solution.chromosome)
            # sanitize routes to remove any stray ids beyond [0..N] (N = total nodes including depot)
            max_valid = len(problem.customers) + 1  # +1 for depot
            ga_routes = [[node for node in r if 0 <= int(node) <= max_valid] for r in ga_routes]
            ga_analysis = ch.analyze_routes(problem, ga_routes)
            # Build full validation inputs sized by max node id observed in routes
            demands = [c.demand for c in problem.customers]
            num_customers = len(problem.customers)
            # If running Solomon CVRP, skip TW validation
            skip_tw = bool(getattr(args, 'solomon_dataset', None))
            if skip_tw:
                time_windows = None
                service_times = None
                distance_matrix = None
            else:
                # determine max node id present in routes
                max_node_id = 0
                for r in ga_routes:
                    if r:
                        max_node_id = max(max_node_id, max(r))
                # time windows and service times sized up to max_node_id
                time_windows = []
                service_times = []
                for i in range(max_node_id + 1):
                    if i == 0:
                        time_windows.append((0.0, 1e9))
                        service_times.append(0.0)
                    else:
                        idx = i - 1
                        if 0 <= idx < num_customers:
                            c = problem.customers[idx]
                            time_windows.append((float(getattr(c, 'ready_time', 0.0)), float(getattr(c, 'due_date', 1e9))))
                            service_times.append(float(getattr(c, 'service_time', 0.0)))
                        else:
                            # fill defaults for any stray ids
                            time_windows.append((0.0, 1e9))
                            service_times.append(0.0)
                # Build a full distance matrix via problem.get_distance to ensure indexing alignment
                try:
                    import numpy as np
                    n = max_node_id + 1
                    dm_full = np.zeros((n, n), dtype=float)
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                dm_full[i, j] = 0.0
                            else:
                                dm_full[i, j] = problem.get_distance(i, j)
                    distance_matrix = dm_full
                except Exception:
                    distance_matrix = getattr(problem, 'distance_matrix', None)
            full_validation = ch.validate_all_constraints(
                ga_routes, demands, problem.customers, time_windows, service_times, distance_matrix
            )
            debug_path = os.path.join(args.output, f"constraint_debug_{int(time.time())}.txt")
            with open(debug_path, 'w', encoding='utf-8') as f:
                import json
                f.write(json.dumps({'ga': ga_analysis, 'full_validation': full_validation}, indent=2, ensure_ascii=False))
            print(f"Constraint analysis written to: {debug_path}")
        except Exception as e:
            print(f"Constraint analysis failed: {e}")

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
