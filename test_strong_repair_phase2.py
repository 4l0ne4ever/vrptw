"""
Test script for Strong Repair Phase 2: Violation Repair
Verifies that Phase 2 correctly reduces violations from ~85 to 0
"""

import sys
import os
import logging
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing.json_loader import JSONDatasetLoader
from src.data_processing.distance import DistanceCalculator
from src.models.vrp_model import create_vrp_problem_from_dict
from src.optimization.matrix_preprocessor import MatrixPreprocessor
from src.optimization.neighbor_lists import NeighborListBuilder
from src.optimization.vidal_evaluator import VidalEvaluator
from src.optimization.strong_repair import StrongRepair
from src.optimization.lns_optimizer import LNSOptimizer
from src.evaluation.metrics import KPICalculator
from src.models.solution import Individual

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_route_distance(problem, routes):
    """Calculate total distance for given routes."""
    total = 0.0
    for route in routes:
        if len(route) <= 1:
            continue
        total += sum(
            problem.get_distance(route[i], route[i + 1])
            for i in range(len(route) - 1)
        )
    return total


def test_strong_repair_phase2():
    """Test Phase 2: Violation Repair on C207 dataset."""
    
    logger.info("="*60)
    logger.info("Testing Strong Repair Phase 2: Violation Repair")
    logger.info("="*60)
    
    # Load C207 dataset (or C201 if C207 not available)
    dataset_name = "C207"
    dataset_path = f"data/datasets/solomon/{dataset_name}.json"
    if not os.path.exists(dataset_path):
        # Try C201 as fallback
        dataset_name = "C201"
        dataset_path = f"data/datasets/solomon/{dataset_name}.json"
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found: {dataset_path}")
            logger.info("Please run: python main.py --convert-solomon first")
            return False
        logger.info(f"C207 not found, using {dataset_name} instead")
    
    # Load dataset and compute distance matrix
    loader = JSONDatasetLoader()
    dataset_dict = loader.load_dataset(dataset_name)
    
    # Calculate distance matrix
    distance_calculator = DistanceCalculator(dataset_type="solomon")
    coordinates = [(dataset_dict['depot']['x'], dataset_dict['depot']['y'])]
    coordinates.extend([(c['x'], c['y']) for c in dataset_dict['customers']])
    distance_matrix = distance_calculator.calculate_distance_matrix(coordinates)
    
    # Create problem with distance matrix
    problem = create_vrp_problem_from_dict(dataset_dict, distance_matrix)
    
    num_customers = len(problem.customers)
    logger.info(f"Dataset loaded: {num_customers} customers, {problem.num_vehicles} vehicles")
    logger.info(f"Vehicle capacity: {problem.vehicle_capacity}, Total demand: {sum(c.demand for c in problem.customers)}")
    
    # Preprocessing
    logger.info("Preprocessing: Computing distance matrix and neighbor lists...")
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
        max_iterations=1000,  # Increased for thorough repair
        enable_swap=True,
        enable_restart=True
    )
    
    # Create a test solution with violations (simulating Phase 1 output)
    # This simulates the relaxed construction result: 100/100 routed, ~85 violations
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Creating test solution with violations...")
    logger.info("="*60)
    
    # Use a simple greedy construction to create initial routes with violations
    # This simulates what Phase 1 (Relaxed Construction) would produce
    from src.algorithms.nearest_neighbor import NearestNeighborHeuristic
    nn_heuristic = NearestNeighborHeuristic(problem)
    initial_individual = nn_heuristic.solve()
    initial_routes = initial_individual.routes
    initial_distance = initial_individual.total_distance
    
    # Count initial violations
    kpi_calc = KPICalculator(problem)
    initial_kpis = kpi_calc.calculate_kpis(initial_individual, execution_time=0)
    initial_violations = initial_kpis.get('constraint_violations', {}).get('time_window_violations', 0)
    
    logger.info(f"Initial solution: {len(initial_routes)} routes")
    logger.info(f"Initial violations: {initial_violations}")
    
    # Verify all customers are routed
    routed_customers = set()
    for route in initial_routes:
        for cust_id in route:
            if cust_id != 0:  # Skip depot
                routed_customers.add(cust_id)
    
    total_customers = num_customers
    logger.info(f"Routed customers: {len(routed_customers)}/{total_customers}")
    
    if len(routed_customers) != total_customers:
        logger.warning(f"⚠️  Not all customers routed! Missing: {total_customers - len(routed_customers)}")
        # This is OK for testing - we just need some violations
    
    logger.info(f"Initial distance: {initial_distance:.2f} km")
    
    # Run Strong Repair (includes both Phase 1 and Phase 2)
    logger.info("\n" + "="*60)
    logger.info("Running Strong Repair (Phase 1 + Phase 2)...")
    logger.info("="*60)
    
    start_time = time.time()
    repaired_routes = strong_repair.repair_routes(initial_routes)
    repair_time = time.time() - start_time
    
    # =============================================================================
    # PHASE 3: LNS POST-OPTIMIZATION (Distance Reduction)
    # =============================================================================
    violations_after_phase2 = strong_repair._count_violations(repaired_routes)
    
    if violations_after_phase2 == 0:
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: LNS POST-OPTIMIZATION (Distance Reduction)")
        logger.info("="*60)
        
        lns = LNSOptimizer(
            problem=problem,
            evaluator=evaluator,
            max_iterations=2000,
            time_limit=300,
            initial_temperature=50.0
        )
        
        distance_before_lns = calculate_route_distance(problem, repaired_routes)
        logger.info(f"🚀 Starting LNS at {distance_before_lns:.2f} km...")
        
        optimized_routes = lns.optimize(
            initial_routes=repaired_routes,
            require_feasible=True
        )
        
        optimized_distance = calculate_route_distance(problem, optimized_routes)
        optimized_violations = strong_repair._count_violations(optimized_routes)
        gap_vs_bks = ((optimized_distance - 588.29) / 588.29) * 100.0
        logger.info(f"🎯 LNS result: {optimized_distance:.2f} km ({gap_vs_bks:+.2f}% vs BKS)")
        logger.info(f"   Status: {'✅ FEASIBLE' if optimized_violations == 0 else '❌ INFEASIBLE'}")
        
        if optimized_violations == 0:
            repaired_routes = optimized_routes
        else:
            logger.warning("⚠️ LNS introduced violations, keeping Phase 2 solution.")
    else:
        logger.warning("⚠️ Skipping LNS because Phase 2 did not reach 0 violations.")
    
    # Calculate final distance
    final_distance = calculate_route_distance(problem, repaired_routes)
    
    # Create Individual object for final KPI calculation
    final_individual = Individual(chromosome=[], routes=repaired_routes)
    final_individual.total_distance = final_distance
    
    # Count final violations
    final_kpis = kpi_calc.calculate_kpis(final_individual, execution_time=repair_time)
    final_violations = final_kpis.get('constraint_violations', {}).get('time_window_violations', 0)
    
    # Verify all customers still routed
    final_routed_customers = set()
    for route in repaired_routes:
        for cust_id in route:
            if cust_id != 0:
                final_routed_customers.add(cust_id)
    
    # Results
    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Initial violations: {initial_violations}")
    logger.info(f"Final violations: {final_violations}")
    logger.info(f"Violations reduced: {initial_violations - final_violations}")
    logger.info(f"Initial distance: {initial_distance:.2f} km")
    logger.info(f"Final distance: {final_distance:.2f} km")
    logger.info(f"Distance change: {final_distance - initial_distance:+.2f} km")
    logger.info(f"Repair time: {repair_time:.2f} seconds")
    logger.info(f"Initial routed: {len(routed_customers)}/{total_customers}")
    logger.info(f"Final routed: {len(final_routed_customers)}/{total_customers}")
    
    # Validation
    success = True
    if len(final_routed_customers) < total_customers:
        logger.error(f"❌ FAIL: Customers lost during repair! {len(final_routed_customers)}/{total_customers}")
        success = False
    else:
        logger.info(f"✅ PASS: All {total_customers} customers still routed")
    
    if final_violations > initial_violations:
        logger.error(f"❌ FAIL: Violations increased! {initial_violations} → {final_violations}")
        success = False
    elif final_violations == 0:
        logger.info(f"✅ PASS: Achieved 0 violations!")
    elif final_violations < initial_violations:
        logger.info(f"✅ PARTIAL: Violations reduced from {initial_violations} to {final_violations}")
        logger.info(f"   (Phase 2 is working, but may need more iterations or different strategy)")
    else:
        logger.warning(f"⚠️  No improvement: violations remain at {final_violations}")
    
    if repair_time > 300:  # 5 minutes
        logger.warning(f"⚠️  Repair took {repair_time:.1f}s (>5 min) - may need optimization")
    else:
        logger.info(f"✅ Repair time acceptable: {repair_time:.1f}s")
    
    logger.info("="*60)
    if success and final_violations == 0:
        logger.info("🎉 TEST PASSED: Phase 2 successfully reduced violations to 0!")
    elif success:
        logger.info("✅ TEST PARTIAL: Phase 2 reduced violations but not to 0")
    else:
        logger.error("❌ TEST FAILED: Phase 2 did not work as expected")
    logger.info("="*60)
    
    return success and final_violations == 0


if __name__ == "__main__":
    try:
        success = test_strong_repair_phase2()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        sys.exit(1)

