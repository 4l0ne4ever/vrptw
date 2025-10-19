"""
Unit tests for VRP-GA System core components.
Tests data processing, models, algorithms, and evaluation modules.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.vrp_model import Customer, Depot, VRPProblem, create_vrp_problem_from_dict
from src.models.solution import Individual, Population
from src.data_processing.loader import SolomonLoader, load_solomon_dataset
from src.data_processing.generator import MockupDataGenerator, generate_mockup_data
from src.data_processing.distance import DistanceCalculator
from src.data_processing.constraints import ConstraintHandler
from src.algorithms.operators import SelectionOperator, CrossoverOperator, MutationOperator
from src.algorithms.fitness import FitnessEvaluator
from src.algorithms.decoder import RouteDecoder
from src.algorithms.local_search import TwoOptOptimizer
from src.algorithms.nearest_neighbor import NearestNeighborHeuristic
from src.evaluation.metrics import KPICalculator
from src.evaluation.comparator import SolutionComparator
from src.evaluation.validator import SolutionValidator


class TestVRPModel(unittest.TestCase):
    """Test VRP model components."""
    
    def setUp(self):
        """Set up test data."""
        self.customers = [
            Customer(1, 10, 20, 5, 0, 100, 10),
            Customer(2, 30, 40, 8, 0, 100, 10),
            Customer(3, 50, 60, 12, 0, 100, 10)
        ]
        self.depot = Depot(0, 0, 0)
        self.distance_matrix = np.array([
            [0, 22.36, 50, 78.10],
            [22.36, 0, 28.28, 56.57],
            [50, 28.28, 0, 28.28],
            [78.10, 56.57, 28.28, 0]
        ])
    
    def test_customer_creation(self):
        """Test customer creation and validation."""
        customer = Customer(1, 10, 20, 5, 0, 100, 10)
        self.assertEqual(customer.id, 1)
        self.assertEqual(customer.x, 10)
        self.assertEqual(customer.y, 20)
        self.assertEqual(customer.demand, 5)
    
    def test_customer_validation(self):
        """Test customer validation."""
        # Valid customer
        customer = Customer(1, 10, 20, 5, 0, 100, 10)
        self.assertIsNotNone(customer)
        
        # Invalid customer (negative demand)
        with self.assertRaises(ValueError):
            Customer(1, 10, 20, -5, 0, 100, 10)
        
        # Invalid customer (invalid time window)
        with self.assertRaises(ValueError):
            Customer(1, 10, 20, 5, 100, 50, 10)
    
    def test_vrp_problem_creation(self):
        """Test VRP problem creation."""
        problem = VRPProblem(
            self.customers, self.depot, 20, 2, self.distance_matrix
        )
        
        self.assertEqual(len(problem.customers), 3)
        self.assertEqual(problem.vehicle_capacity, 20)
        self.assertEqual(problem.num_vehicles, 2)
    
    def test_vrp_problem_validation(self):
        """Test VRP problem validation."""
        # Valid problem
        problem = VRPProblem(
            self.customers, self.depot, 20, 2, self.distance_matrix
        )
        self.assertTrue(problem.is_feasible())
        
        # Invalid problem (no customers)
        with self.assertRaises(ValueError):
            VRPProblem([], self.depot, 20, 2, self.distance_matrix)
        
        # Invalid problem (negative capacity)
        with self.assertRaises(ValueError):
            VRPProblem(self.customers, self.depot, -20, 2, self.distance_matrix)
    
    def test_problem_info(self):
        """Test problem information extraction."""
        problem = VRPProblem(
            self.customers, self.depot, 20, 2, self.distance_matrix
        )
        
        info = problem.get_problem_info()
        self.assertEqual(info['num_customers'], 3)
        self.assertEqual(info['vehicle_capacity'], 20)
        self.assertEqual(info['num_vehicles'], 2)
        self.assertEqual(info['total_demand'], 25)  # 5 + 8 + 12


class TestSolutionModel(unittest.TestCase):
    """Test solution model components."""
    
    def setUp(self):
        """Set up test data."""
        self.chromosome = [1, 2, 3]
        self.routes = [[0, 1, 2, 0], [0, 3, 0]]
    
    def test_individual_creation(self):
        """Test individual creation."""
        individual = Individual(
            chromosome=self.chromosome,
            fitness=0.5,
            routes=self.routes
        )
        
        self.assertEqual(individual.chromosome, self.chromosome)
        self.assertEqual(individual.fitness, 0.5)
        self.assertEqual(individual.routes, self.routes)
    
    def test_individual_copy(self):
        """Test individual copying."""
        individual = Individual(
            chromosome=self.chromosome,
            fitness=0.5,
            routes=self.routes
        )
        
        copied = individual.copy()
        self.assertEqual(copied.chromosome, individual.chromosome)
        self.assertEqual(copied.fitness, individual.fitness)
        self.assertEqual(copied.routes, individual.routes)
        
        # Modify copy and ensure original is unchanged
        copied.chromosome.append(4)
        self.assertNotEqual(copied.chromosome, individual.chromosome)
    
    def test_population_creation(self):
        """Test population creation."""
        individuals = [
            Individual(chromosome=[1, 2], fitness=0.5),
            Individual(chromosome=[2, 1], fitness=0.3),
            Individual(chromosome=[1, 3, 2], fitness=0.7)
        ]
        
        population = Population(individuals)
        self.assertEqual(population.get_size(), 3)
        self.assertEqual(population.get_best_fitness(), 0.7)
        self.assertEqual(population.get_worst_fitness(), 0.3)
    
    def test_population_statistics(self):
        """Test population statistics."""
        individuals = [
            Individual(chromosome=[1, 2], fitness=0.5),
            Individual(chromosome=[2, 1], fitness=0.3),
            Individual(chromosome=[1, 3, 2], fitness=0.7)
        ]
        
        population = Population(individuals)
        stats = population.get_statistics()
        
        self.assertEqual(stats['size'], 3)
        self.assertEqual(stats['best_fitness'], 0.7)
        self.assertEqual(stats['avg_fitness'], 0.5)  # (0.5 + 0.3 + 0.7) / 3


class TestDataProcessing(unittest.TestCase):
    """Test data processing modules."""
    
    def test_distance_calculator(self):
        """Test distance calculation."""
        coordinates = [(0, 0), (3, 4), (0, 5)]
        calculator = DistanceCalculator()
        
        distance_matrix = calculator.calculate_distance_matrix(coordinates)
        
        # Check depot to first customer distance (3,4) = 5
        self.assertAlmostEqual(distance_matrix[0, 1], 5.0, places=2)
        
        # Check depot to second customer distance (0,5) = 5
        self.assertAlmostEqual(distance_matrix[0, 2], 5.0, places=2)
        
        # Check distance between customers
        self.assertAlmostEqual(distance_matrix[1, 2], 5.0, places=2)
    
    def test_constraint_handler(self):
        """Test constraint handling."""
        handler = ConstraintHandler(vehicle_capacity=20, num_vehicles=2)
        
        # Valid routes
        routes = [[0, 1, 2, 0], [0, 3, 0]]
        demands = [5, 8, 12]  # Total demand per customer
        
        is_valid, penalty = handler.validate_capacity_constraint(routes, demands)
        self.assertTrue(is_valid)
        self.assertEqual(penalty, 0.0)
        
        # Invalid routes (capacity exceeded)
        invalid_routes = [[0, 1, 2, 3, 0]]  # All customers in one route
        is_valid, penalty = handler.validate_capacity_constraint(invalid_routes, demands)
        self.assertFalse(is_valid)
        self.assertGreater(penalty, 0.0)
    
    def test_mockup_data_generator(self):
        """Test mockup data generation."""
        generator = MockupDataGenerator({
            'n_customers': 10,
            'demand_lambda': 5,
            'area_bounds': (0, 100),
            'clustering': 'random',
            'seed': 42
        })
        
        customers = generator.generate_customers()
        depot = generator.generate_depot()
        
        self.assertEqual(len(customers), 10)
        self.assertIsNotNone(depot)
        self.assertEqual(depot['id'], 0)
        
        # Check that all customers have valid data
        for customer in customers:
            self.assertGreaterEqual(customer['demand'], 0)
            self.assertGreaterEqual(customer['x'], 0)
            self.assertGreaterEqual(customer['y'], 0)


class TestAlgorithms(unittest.TestCase):
    """Test algorithm components."""
    
    def setUp(self):
        """Set up test data."""
        self.customers = [
            Customer(1, 10, 20, 5, 0, 100, 10),
            Customer(2, 30, 40, 8, 0, 100, 10),
            Customer(3, 50, 60, 12, 0, 100, 10)
        ]
        self.depot = Depot(0, 0, 0)
        self.distance_matrix = np.array([
            [0, 22.36, 50, 78.10],
            [22.36, 0, 28.28, 56.57],
            [50, 28.28, 0, 28.28],
            [78.10, 56.57, 28.28, 0]
        ])
        self.problem = VRPProblem(
            self.customers, self.depot, 20, 2, self.distance_matrix
        )
    
    def test_selection_operators(self):
        """Test selection operators."""
        individuals = [
            Individual(chromosome=[1, 2], fitness=0.5),
            Individual(chromosome=[2, 1], fitness=0.3),
            Individual(chromosome=[1, 3, 2], fitness=0.7)
        ]
        
        # Tournament selection
        parents = SelectionOperator.tournament_selection(individuals, tournament_size=2, num_parents=2)
        self.assertEqual(len(parents), 2)
        
        # Roulette wheel selection
        parents = SelectionOperator.roulette_wheel_selection(individuals, num_parents=2)
        self.assertEqual(len(parents), 2)
    
    def test_crossover_operators(self):
        """Test crossover operators."""
        parent1 = Individual(chromosome=[1, 2, 3, 4, 5])
        parent2 = Individual(chromosome=[5, 4, 3, 2, 1])
        
        # Order crossover
        child1, child2 = CrossoverOperator.order_crossover(parent1, parent2)
        self.assertEqual(len(child1.chromosome), 5)
        self.assertEqual(len(child2.chromosome), 5)
        
        # Partially mapped crossover
        child1, child2 = CrossoverOperator.partially_mapped_crossover(parent1, parent2)
        self.assertEqual(len(child1.chromosome), 5)
        self.assertEqual(len(child2.chromosome), 5)
    
    def test_mutation_operators(self):
        """Test mutation operators."""
        individual = Individual(chromosome=[1, 2, 3, 4, 5])
        
        # Swap mutation
        mutated = MutationOperator.swap_mutation(individual, mutation_rate=1.0)
        self.assertEqual(len(mutated.chromosome), 5)
        
        # Inversion mutation
        mutated = MutationOperator.inversion_mutation(individual, mutation_rate=1.0)
        self.assertEqual(len(mutated.chromosome), 5)
        
        # Insertion mutation
        mutated = MutationOperator.insertion_mutation(individual, mutation_rate=1.0)
        self.assertEqual(len(mutated.chromosome), 5)
    
    def test_fitness_evaluator(self):
        """Test fitness evaluation."""
        evaluator = FitnessEvaluator(self.problem)
        
        individual = Individual(chromosome=[1, 2, 3])
        fitness = evaluator.evaluate_fitness(individual)
        
        self.assertGreater(fitness, 0)
        self.assertIsNotNone(individual.routes)
        self.assertGreater(individual.total_distance, 0)
    
    def test_route_decoder(self):
        """Test route decoding."""
        decoder = RouteDecoder(self.problem)
        
        chromosome = [1, 2, 3]
        routes = decoder.decode_chromosome(chromosome)
        
        self.assertIsInstance(routes, list)
        self.assertGreater(len(routes), 0)
        
        # Check that all routes start and end at depot
        for route in routes:
            if route:
                self.assertEqual(route[0], 0)  # Start at depot
                self.assertEqual(route[-1], 0)  # End at depot
    
    def test_nearest_neighbor_heuristic(self):
        """Test nearest neighbor heuristic."""
        heuristic = NearestNeighborHeuristic(self.problem)
        solution = heuristic.solve()
        
        self.assertIsInstance(solution, Individual)
        self.assertGreater(solution.fitness, 0)
        self.assertIsNotNone(solution.routes)
    
    def test_two_opt_optimizer(self):
        """Test 2-opt optimizer."""
        optimizer = TwoOptOptimizer(self.problem)
        
        # Create a simple solution
        individual = Individual(chromosome=[1, 2, 3])
        evaluator = FitnessEvaluator(self.problem)
        evaluator.evaluate_fitness(individual)
        
        # Optimize
        optimized = optimizer.optimize_individual(individual)
        
        self.assertIsInstance(optimized, Individual)
        self.assertIsNotNone(optimized.routes)


class TestEvaluation(unittest.TestCase):
    """Test evaluation modules."""
    
    def setUp(self):
        """Set up test data."""
        self.customers = [
            Customer(1, 10, 20, 5, 0, 100, 10),
            Customer(2, 30, 40, 8, 0, 100, 10),
            Customer(3, 50, 60, 12, 0, 100, 10)
        ]
        self.depot = Depot(0, 0, 0)
        self.distance_matrix = np.array([
            [0, 22.36, 50, 78.10],
            [22.36, 0, 28.28, 56.57],
            [50, 28.28, 0, 28.28],
            [78.10, 56.57, 28.28, 0]
        ])
        self.problem = VRPProblem(
            self.customers, self.depot, 20, 2, self.distance_matrix
        )
    
    def test_kpi_calculator(self):
        """Test KPI calculation."""
        calculator = KPICalculator(self.problem)
        
        individual = Individual(chromosome=[1, 2, 3])
        evaluator = FitnessEvaluator(self.problem)
        evaluator.evaluate_fitness(individual)
        
        kpis = calculator.calculate_kpis(individual)
        
        self.assertIn('total_distance', kpis)
        self.assertIn('num_routes', kpis)
        self.assertIn('total_cost', kpis)
        self.assertIn('avg_utilization', kpis)
        self.assertIn('efficiency_score', kpis)
        self.assertIn('is_feasible', kpis)
    
    def test_solution_comparator(self):
        """Test solution comparison."""
        comparator = SolutionComparator(self.problem)
        
        # Create two solutions
        solution1 = Individual(chromosome=[1, 2, 3])
        solution2 = Individual(chromosome=[3, 2, 1])
        
        evaluator = FitnessEvaluator(self.problem)
        evaluator.evaluate_fitness(solution1)
        evaluator.evaluate_fitness(solution2)
        
        comparison = comparator.compare_methods(solution1, solution2)
        
        self.assertIn('solution1', comparison)
        self.assertIn('solution2', comparison)
        self.assertIn('improvements', comparison)
    
    def test_solution_validator(self):
        """Test solution validation."""
        validator = SolutionValidator(self.problem)
        
        individual = Individual(chromosome=[1, 2, 3])
        evaluator = FitnessEvaluator(self.problem)
        evaluator.evaluate_fitness(individual)
        
        validation_result = validator.validate_solution(individual)
        
        self.assertIn('is_valid', validation_result)
        self.assertIn('is_feasible', validation_result)
        self.assertIn('errors', validation_result)
        self.assertIn('warnings', validation_result)
        self.assertIn('validation_score', validation_result)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data to solution."""
        # Generate mockup data
        data = generate_mockup_data(
            n_customers=5,
            vehicle_capacity=20,
            clustering='random',
            seed=42
        )
        
        # Calculate distance matrix
        distance_calculator = DistanceCalculator()
        coordinates = [(data['depot']['x'], data['depot']['y'])]
        coordinates.extend([(c['x'], c['y']) for c in data['customers']])
        distance_matrix = distance_calculator.calculate_distance_matrix(coordinates)
        
        # Create VRP problem
        problem = create_vrp_problem_from_dict(data, distance_matrix)
        
        # Solve with Nearest Neighbor
        nn_heuristic = NearestNeighborHeuristic(problem)
        nn_solution = nn_heuristic.solve()
        
        # Evaluate solution
        kpi_calculator = KPICalculator(problem)
        nn_kpis = kpi_calculator.calculate_kpis(nn_solution)
        
        # Validate solution
        validator = SolutionValidator(problem)
        validation_result = validator.validate_solution(nn_solution)
        
        # Check that solution is valid
        self.assertGreater(nn_solution.fitness, 0)
        self.assertGreater(nn_kpis['total_distance'], 0)
        self.assertGreaterEqual(validation_result['validation_score'], 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestVRPModel,
        TestSolutionModel,
        TestDataProcessing,
        TestAlgorithms,
        TestEvaluation,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
