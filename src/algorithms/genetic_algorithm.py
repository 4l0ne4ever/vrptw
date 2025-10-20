"""
Main Genetic Algorithm engine for VRP.
Implements the complete GA workflow with population management.
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from src.models.solution import Individual, Population
from src.models.vrp_model import VRPProblem
from src.algorithms.operators import (
    SelectionOperator, CrossoverOperator, MutationOperator, AdaptiveMutationOperator
)
from src.algorithms.fitness import FitnessEvaluator
from config import GA_CONFIG


class GeneticAlgorithm:
    """Main Genetic Algorithm engine for VRP optimization."""
    
    def __init__(self, problem: VRPProblem, config: Optional[Dict] = None):
        """
        Initialize GA engine.
        
        Args:
            problem: VRP problem instance
            config: GA configuration parameters
        """
        self.problem = problem
        self.config = config or GA_CONFIG.copy()
        
        # Initialize components
        self.fitness_evaluator = FitnessEvaluator(problem)
        self.adaptive_mutation = AdaptiveMutationOperator(
            base_mutation_rate=self.config['mutation_prob'],
            diversity_threshold=0.1
        )
        
        # GA state
        self.population = Population()
        self.generation = 0
        self.best_solution = None
        self.convergence_history = []
        self.execution_time = 0.0
        
        # Statistics
        self.stats = {
            'generations': 0,
            'total_evaluations': 0,
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'diversity_history': [],
            'convergence_generation': None
        }
    
    def initialize_population(self, population_size: Optional[int] = None) -> Population:
        """
        Initialize population with diverse individuals.
        
        Args:
            population_size: Size of population to create
            
        Returns:
            Initialized population
        """
        population_size = population_size or self.config['population_size']
        self.population = Population()
        
        # Create diverse initial population
        for i in range(population_size):
            individual = self._create_individual(i, population_size)
            self.fitness_evaluator.evaluate_fitness(individual)
            self.population.add_individual(individual)
        
        self.population._update_statistics()
        self.stats['total_evaluations'] += population_size
        
        return self.population
    
    def _create_individual(self, index: int, population_size: int) -> Individual:
        """
        Create an individual using different initialization strategies.
        
        Args:
            index: Individual index
            population_size: Total population size
            
        Returns:
            Created individual
        """
        if not self.problem.customers:
            raise ValueError("No customers in problem")
        
        customer_ids = [c.id for c in self.problem.customers]
        
        # Use different initialization strategies
        strategy_ratio = index / population_size
        
        if strategy_ratio < 0.6:
            # Random initialization (60%)
            chromosome = customer_ids.copy()
            random.shuffle(chromosome)
        elif strategy_ratio < 0.8:
            # Greedy initialization (20%)
            chromosome = self._greedy_initialization(customer_ids)
        elif strategy_ratio < 0.95:
            # Cluster-first initialization (15%)
            chromosome = self._cluster_first_initialization(customer_ids)
        else:
            # Savings-based initialization (5%)
            chromosome = self._savings_initialization(customer_ids)
        
        return Individual(chromosome=chromosome)
    
    def _greedy_initialization(self, customer_ids: List[int]) -> List[int]:
        """Create chromosome using greedy nearest neighbor approach."""
        if not customer_ids:
            return []
        
        chromosome = []
        unvisited = customer_ids.copy()
        current = 0  # Start at depot
        
        while unvisited:
            # Find nearest unvisited customer
            nearest = min(unvisited, key=lambda c: self.problem.get_distance(current, c))
            chromosome.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return chromosome
    
    def _cluster_first_initialization(self, customer_ids: List[int]) -> List[int]:
        """Create chromosome using cluster-first approach."""
        if len(customer_ids) < 3:
            return self._greedy_initialization(customer_ids)
        
        # Simple clustering based on coordinates
        customer_coords = [(c.x, c.y) for c in self.problem.customers]
        
        # Use k-means-like clustering (simplified)
        n_clusters = min(3, len(customer_ids) // 3)
        if n_clusters < 2:
            return self._greedy_initialization(customer_ids)
        
        # Assign customers to clusters
        clusters = [[] for _ in range(n_clusters)]
        for i, customer_id in enumerate(customer_ids):
            cluster_idx = i % n_clusters
            clusters[cluster_idx].append(customer_id)
        
        # Build chromosome by visiting clusters
        chromosome = []
        for cluster in clusters:
            # Within each cluster, use greedy approach
            cluster_chromosome = self._greedy_initialization(cluster)
            chromosome.extend(cluster_chromosome)
        
        return chromosome
    
    def _savings_initialization(self, customer_ids: List[int]) -> List[int]:
        """Create chromosome using savings-based approach."""
        if len(customer_ids) < 3:
            return self._greedy_initialization(customer_ids)
        
        # Calculate savings for all customer pairs
        savings = []
        for i, c1 in enumerate(customer_ids):
            for j, c2 in enumerate(customer_ids):
                if i != j:
                    # Savings = distance(depot, c1) + distance(depot, c2) - distance(c1, c2)
                    saving = (self.problem.get_distance(0, c1) + 
                            self.problem.get_distance(0, c2) - 
                            self.problem.get_distance(c1, c2))
                    savings.append((saving, c1, c2))
        
        # Sort by savings (descending)
        savings.sort(key=lambda x: x[0], reverse=True)
        
        # Build chromosome using savings
        chromosome = []
        used_customers = set()
        
        for saving, c1, c2 in savings:
            if c1 not in used_customers and c2 not in used_customers:
                chromosome.extend([c1, c2])
                used_customers.update([c1, c2])
            elif c1 not in used_customers:
                chromosome.append(c1)
                used_customers.add(c1)
            elif c2 not in used_customers:
                chromosome.append(c2)
                used_customers.add(c2)
        
        # Add any remaining customers
        remaining = [c for c in customer_ids if c not in used_customers]
        chromosome.extend(remaining)
        
        return chromosome
    
    def evolve(self, max_generations: Optional[int] = None) -> Tuple[Individual, List[Dict]]:
        """
        Run GA evolution process.
        
        Args:
            max_generations: Maximum number of generations
            
        Returns:
            Tuple of (best_solution, evolution_data)
        """
        max_generations = max_generations or self.config['generations']
        start_time = time.time()
        
        # Initialize population if not done
        if self.population.is_empty():
            self.initialize_population()
        
        # Evolution data collection
        evolution_data = []
        
        # Evolution loop
        for generation in range(max_generations):
            self.generation = generation
            
            # Create next generation
            self._create_next_generation()
            
            # Update statistics
            self._update_statistics()
            
            # Collect evolution data
            gen_data = {
                'generation': generation,
                'evaluated_individuals': len(self.population.individuals),
                'min_fitness': min(ind.fitness for ind in self.population.individuals),
                'max_fitness': max(ind.fitness for ind in self.population.individuals),
                'avg_fitness': np.mean([ind.fitness for ind in self.population.individuals]),
                'std_fitness': np.std([ind.fitness for ind in self.population.individuals]),
                'best_distance': self.population.get_best_individual().total_distance,
                'avg_distance': np.mean([ind.total_distance for ind in self.population.individuals]),
                'diversity': self._calculate_diversity()
            }
            evolution_data.append(gen_data)
            
            # Check convergence
            if self._check_convergence():
                self.stats['convergence_generation'] = generation
                break
        
        # Final evaluation
        self.execution_time = time.time() - start_time
        self.stats['generations'] = self.generation + 1
        
        # Get best solution
        self.best_solution = self.population.get_best_individual()
        
        return self.best_solution, evolution_data
        
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population.individuals) < 2:
            return 0.0
        
        # Calculate average pairwise distance between chromosomes
        distances = []
        individuals = self.population.individuals
        
        for i in range(len(individuals)):
            for j in range(i + 1, len(individuals)):
                # Calculate Hamming distance
                dist = sum(1 for a, b in zip(individuals[i].chromosome, individuals[j].chromosome) if a != b)
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _create_next_generation(self):
        """Create next generation using selection, crossover, and mutation."""
        new_population = []
        
        # Apply elitism
        elite_count = int(self.config['elitism_rate'] * len(self.population.individuals))
        elite = self.population.apply_elitism(elite_count)
        new_population.extend(elite)
        
        # Create offspring
        offspring_count = len(self.population.individuals) - elite_count
        
        for _ in range(offspring_count // 2):
            # Selection
            parents = SelectionOperator.tournament_selection(
                self.population.individuals,
                tournament_size=self.config['tournament_size'],
                num_parents=2
            )
            
            # Crossover
            if random.random() < self.config['crossover_prob']:
                offspring1, offspring2 = CrossoverOperator.order_crossover(
                    parents[0], parents[1]
                )
            else:
                offspring1, offspring2 = parents[0].copy(), parents[1].copy()
            
            # Mutation
            population_diversity = self.population.diversity
            offspring1 = self.adaptive_mutation.mutate(offspring1, population_diversity)
            offspring2 = self.adaptive_mutation.mutate(offspring2, population_diversity)
            
            # Evaluate fitness
            self.fitness_evaluator.evaluate_fitness(offspring1)
            self.fitness_evaluator.evaluate_fitness(offspring2)
            
            new_population.extend([offspring1, offspring2])
            self.stats['total_evaluations'] += 2
        
        # Replace population
        self.population.replace_individuals(new_population)
        self.population.next_generation()
    
    def _update_statistics(self):
        """Update GA statistics."""
        stats = self.population.get_statistics()
        
        self.stats['best_fitness_history'].append(stats['best_fitness'])
        self.stats['avg_fitness_history'].append(stats['avg_fitness'])
        self.stats['diversity_history'].append(stats['diversity'])
    
    def _check_convergence(self) -> bool:
        """
        Check if GA has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.stats['best_fitness_history']) < 50:
            return False
        
        # Check stagnation
        recent_fitness = self.stats['best_fitness_history'][-50:]
        fitness_std = np.std(recent_fitness)
        
        if fitness_std < self.config['convergence_threshold']:
            return True
        
        # Check stagnation limit
        if len(self.stats['best_fitness_history']) >= self.config['stagnation_limit']:
            recent_best = self.stats['best_fitness_history'][-self.config['stagnation_limit']:]
            if max(recent_best) - min(recent_best) < self.config['convergence_threshold']:
                return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get GA execution statistics."""
        return {
            'generations': self.stats['generations'],
            'total_evaluations': self.stats['total_evaluations'],
            'execution_time': self.execution_time,
            'convergence_generation': self.stats['convergence_generation'],
            'best_fitness': self.population.get_best_fitness(),
            'avg_fitness': self.population.get_avg_fitness(),
            'diversity': self.population.diversity,
            'population_size': self.population.get_size()
        }
    
    def get_convergence_data(self) -> Dict:
        """Get convergence data for visualization."""
        return {
            'generations': list(range(len(self.stats['best_fitness_history']))),
            'best_fitness': self.stats['best_fitness_history'],
            'avg_fitness': self.stats['avg_fitness_history'],
            'diversity': self.stats['diversity_history']
        }
    
    def save_best_solution(self, filepath: str):
        """Save best solution to file."""
        if self.best_solution is None:
            raise ValueError("No solution to save")
        
        import json
        
        solution_data = {
            'chromosome': self.best_solution.chromosome,
            'routes': self.best_solution.routes,
            'fitness': self.best_solution.fitness,
            'total_distance': self.best_solution.total_distance,
            'is_valid': self.best_solution.is_valid,
            'penalty': self.best_solution.penalty,
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(solution_data, f, indent=2)
    
    def load_solution(self, filepath: str) -> Individual:
        """Load solution from file."""
        import json
        
        with open(filepath, 'r') as f:
            solution_data = json.load(f)
        
        individual = Individual(
            chromosome=solution_data['chromosome'],
            fitness=solution_data['fitness'],
            routes=solution_data['routes'],
            total_distance=solution_data['total_distance'],
            is_valid=solution_data['is_valid'],
            penalty=solution_data['penalty']
        )
        
        return individual


def run_genetic_algorithm(problem: VRPProblem, 
                         config: Optional[Dict] = None,
                         max_generations: Optional[int] = None) -> Tuple[Individual, Dict, List[Dict]]:
    """
    Convenience function to run GA.
    
    Args:
        problem: VRP problem instance
        config: GA configuration
        max_generations: Maximum generations
        
    Returns:
        Tuple of (best_solution, statistics, evolution_data)
    """
    ga = GeneticAlgorithm(problem, config)
    best_solution, evolution_data = ga.evolve(max_generations)
    statistics = ga.get_statistics()
    
    return best_solution, statistics, evolution_data
