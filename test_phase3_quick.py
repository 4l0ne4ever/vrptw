"""
Quick test for Phase 3 LNS integration
Tests LNS on a feasible solution (0 violations)
"""

import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing.json_loader import JSONDatasetLoader
from src.data_processing.distance import DistanceCalculator
from src.models.vrp_model import create_vrp_problem_from_dict
from src.optimization.matrix_preprocessor import MatrixPreprocessor
from src.optimization.neighbor_lists import NeighborListBuilder
from src.optimization.vidal_evaluator import VidalEvaluator
from src.optimization.strong_repair import StrongRepair

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phase3():
    """Quick test: Run Strong Repair with Phase 3 on C201 dataset."""
    
    logger.info("="*60)
    logger.info("Quick Test: Phase 3 LNS Integration")
    logger.info("="*60)
    
    # Load dataset
    dataset_name = "C201"
    loader = JSONDatasetLoader()
    dataset_dict = loader.load_dataset(dataset_name)
    
    # Calculate distance matrix
    distance_calculator = DistanceCalculator(dataset_type="solomon")
    coordinates = [(dataset_dict['depot']['x'], dataset_dict['depot']['y'])]
    coordinates.extend([(c['x'], c['y']) for c in dataset_dict['customers']])
    distance_matrix = distance_calculator.calculate_distance_matrix(coordinates)
    
    # Create problem
    problem = create_vrp_problem_from_dict(dataset_dict, distance_matrix)
    
    # Preprocessing
    preprocessor = MatrixPreprocessor(problem)
    distance_matrix, time_matrix = preprocessor.normalize_matrices()
    
    neighbor_builder = NeighborListBuilder(time_matrix, problem, k=40)
    neighbor_lists = neighbor_builder.build_neighbor_lists()
    
    evaluator = VidalEvaluator(problem, distance_matrix, time_matrix)
    
    # Initialize Strong Repair
    strong_repair = StrongRepair(
        problem=problem,
        neighbor_lists=neighbor_lists,
        evaluator=evaluator,
        max_iterations=500,  # Reduced for quick test
        enable_swap=True,
        enable_restart=True
    )
    
    # Create initial solution with violations
    from src.algorithms.nearest_neighbor import NearestNeighborHeuristic
    nn_heuristic = NearestNeighborHeuristic(problem)
    initial_individual = nn_heuristic.solve()
    initial_routes = initial_individual.routes
    
    initial_distance = sum(
        sum(problem.get_distance(route[i], route[i+1])
            for i in range(len(route)-1))
        for route in initial_routes if len(route) > 1
    )
    
    logger.info(f"\nInitial solution: {initial_distance:.2f} km")
    
    # Run Strong Repair (Phase 1 + Phase 2 + Phase 3)
    logger.info("\n" + "="*60)
    logger.info("Running Full Pipeline (Phase 1 + Phase 2 + Phase 3)...")
    logger.info("="*60)
    
    import time
    start_time = time.time()
    final_routes = strong_repair.repair_routes(initial_routes)
    total_time = time.time() - start_time
    
    # Calculate final metrics
    final_distance = sum(
        sum(problem.get_distance(route[i], route[i+1])
            for i in range(len(route)-1))
        for route in final_routes if len(route) > 1
    )
    
    final_violations = strong_repair._count_violations(final_routes)
    
    # Results
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Initial distance: {initial_distance:.2f} km")
    logger.info(f"Final distance:   {final_distance:.2f} km")
    logger.info(f"Improvement:      {initial_distance - final_distance:+.2f} km")
    logger.info(f"Final violations: {final_violations}")
    logger.info(f"Total time:       {total_time:.1f} seconds")
    
    if final_violations == 0:
        logger.info("✅ SUCCESS: 0 violations maintained!")
        if final_distance < initial_distance:
            logger.info(f"✅ SUCCESS: Distance improved by {initial_distance - final_distance:.2f} km!")
        else:
            logger.info(f"⚠️  Distance increased by {final_distance - initial_distance:.2f} km (acceptable for feasibility)")
    else:
        logger.warning(f"⚠️  {final_violations} violations remain")
    
    logger.info("="*60)
    
    return final_violations == 0

if __name__ == "__main__":
    try:
        success = test_phase3()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)

