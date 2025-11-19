"""
Genetic Algorithm operators for VRP.
Implements selection, crossover, and mutation operators.
"""

import random
import numpy as np
from typing import List, Tuple, Optional
from src.models.solution import Individual


class SelectionOperator:
    """Selection operators for GA."""
    
    @staticmethod
    def tournament_selection(population: List[Individual], 
                           tournament_size: int = 3,
                           num_parents: int = 2) -> List[Individual]:
        """
        Tournament selection operator.
        
        Args:
            population: List of individuals
            tournament_size: Size of tournament
            num_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        parents = []
        
        for _ in range(num_parents):
            # Randomly select tournament participants
            tournament = random.sample(population, min(tournament_size, len(population)))
            
            # Select best individual from tournament
            winner = max(tournament, key=lambda ind: ind.fitness)
            parents.append(winner)
        
        return parents
    
    @staticmethod
    def roulette_wheel_selection(population: List[Individual], 
                               num_parents: int = 2) -> List[Individual]:
        """
        Roulette wheel selection operator.
        
        Args:
            population: List of individuals
            num_parents: Number of parents to select
            
        Returns:
            List of selected parents
        """
        if not population:
            return []
        
        # Calculate fitness values
        fitness_values = [ind.fitness for ind in population]
        
        # Handle negative fitness values
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1 for f in fitness_values]
        
        # Calculate selection probabilities
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            # If all fitness values are 0, select randomly
            return random.sample(population, min(num_parents, len(population)))
        
        probabilities = [f / total_fitness for f in fitness_values]
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            # Roulette wheel selection
            r = random.random()
            cumulative_prob = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    parents.append(population[i])
                    break
        
        return parents


class CrossoverOperator:
    """Crossover operators for VRP."""
    
    @staticmethod
    def order_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Order Crossover (OX) operator.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        if len(parent1.chromosome) != len(parent2.chromosome):
            raise ValueError("Parents must have same chromosome length")
        
        n = len(parent1.chromosome)
        if n < 2:
            return parent1.copy(), parent2.copy()
        
        # Select crossover points
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, n - 1)
        
        # Create offspring
        child1 = CrossoverOperator._create_ox_child(parent1, parent2, start, end)
        child2 = CrossoverOperator._create_ox_child(parent2, parent1, start, end)
        
        return child1, child2
    
    @staticmethod
    def _create_ox_child(parent1: Individual, parent2: Individual, 
                        start: int, end: int) -> Individual:
        """Create a child using Order Crossover."""
        n = len(parent1.chromosome)
        child_chromosome = [-1] * n
        
        # Copy segment from parent1
        for i in range(start, end + 1):
            child_chromosome[i] = parent1.chromosome[i]
        
        # Fill remaining positions from parent2
        remaining_genes = []
        for gene in parent2.chromosome:
            if gene not in child_chromosome[start:end + 1]:
                remaining_genes.append(gene)
        
        # Place remaining genes
        idx = end + 1
        for gene in remaining_genes:
            child_chromosome[idx % n] = gene
            idx += 1
        
        # Create child individual
        child = Individual(chromosome=child_chromosome)
        return child
    
    @staticmethod
    def partially_mapped_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Partially Mapped Crossover (PMX) operator.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        if len(parent1.chromosome) != len(parent2.chromosome):
            raise ValueError("Parents must have same chromosome length")
        
        n = len(parent1.chromosome)
        if n < 2:
            return parent1.copy(), parent2.copy()
        
        # Select crossover points
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, n - 1)
        
        # Create offspring
        child1 = CrossoverOperator._create_pmx_child(parent1, parent2, start, end)
        child2 = CrossoverOperator._create_pmx_child(parent2, parent1, start, end)
        
        return child1, child2
    
    @staticmethod
    def _create_pmx_child(parent1: Individual, parent2: Individual, 
                         start: int, end: int) -> Individual:
        """
        Create a child using Partially Mapped Crossover.
        
        Correct PMX algorithm:
        1. Copy segment from parent1
        2. For each position outside segment, try to copy from parent2
        3. If gene from parent2 is already in child (in copied segment), 
           follow the mapping chain to find a valid replacement
        4. Prevent infinite loops with cycle detection
        """
        n = len(parent1.chromosome)
        child_chromosome = [-1] * n
        
        # Step 1: Copy segment from parent1
        for i in range(start, end + 1):
            child_chromosome[i] = parent1.chromosome[i]
        
        # Step 2: Create mapping from genes in segment
        mapping = {}
        for i in range(start, end + 1):
            gene1 = parent1.chromosome[i]
            gene2 = parent2.chromosome[i]
            mapping[gene2] = gene1
        
        # Step 3: Fill remaining positions
        for i in range(n):
            if child_chromosome[i] == -1:  # Position not filled yet
                gene = parent2.chromosome[i]
                visited = set()
                
                # Follow mapping chain until we find a gene not in child
                # Use visited set to prevent infinite loops from cycles
                while gene in child_chromosome and gene in mapping and gene not in visited:
                    visited.add(gene)
                    gene = mapping[gene]
                
                child_chromosome[i] = gene
        
        # Create child individual
        child = Individual(chromosome=child_chromosome)
        return child


class MutationOperator:
    """Mutation operators for VRP."""
    
    @staticmethod
    def swap_mutation(individual: Individual, mutation_rate: float = 0.1) -> Individual:
        """
        Swap mutation operator.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated individual
        """
        if random.random() > mutation_rate or len(individual.chromosome) < 2:
            return individual.copy()
        
        # Select two random positions
        pos1, pos2 = random.sample(range(len(individual.chromosome)), 2)
        
        # Create mutated chromosome
        mutated_chromosome = individual.chromosome.copy()
        mutated_chromosome[pos1], mutated_chromosome[pos2] = mutated_chromosome[pos2], mutated_chromosome[pos1]
        
        # Create mutated individual
        mutated = Individual(chromosome=mutated_chromosome)
        return mutated
    
    @staticmethod
    def inversion_mutation(individual: Individual, mutation_rate: float = 0.1) -> Individual:
        """
        Inversion mutation operator.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated individual
        """
        if random.random() > mutation_rate or len(individual.chromosome) < 2:
            return individual.copy()
        
        # Select two random positions
        start = random.randint(0, len(individual.chromosome) - 2)
        end = random.randint(start + 1, len(individual.chromosome) - 1)
        
        # Create mutated chromosome
        mutated_chromosome = individual.chromosome.copy()
        mutated_chromosome[start:end + 1] = reversed(mutated_chromosome[start:end + 1])
        
        # Create mutated individual
        mutated = Individual(chromosome=mutated_chromosome)
        return mutated
    
    @staticmethod
    def insertion_mutation(individual: Individual, mutation_rate: float = 0.1) -> Individual:
        """
        Insertion mutation operator.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated individual
        """
        if random.random() > mutation_rate or len(individual.chromosome) < 2:
            return individual.copy()
        
        # Select random positions
        from_pos = random.randint(0, len(individual.chromosome) - 1)
        to_pos = random.randint(0, len(individual.chromosome) - 1)
        
        if from_pos == to_pos:
            return individual.copy()
        
        # Create mutated chromosome
        mutated_chromosome = individual.chromosome.copy()
        gene = mutated_chromosome.pop(from_pos)
        mutated_chromosome.insert(to_pos, gene)
        
        # Create mutated individual
        mutated = Individual(chromosome=mutated_chromosome)
        return mutated
    
    @staticmethod
    def scramble_mutation(individual: Individual, mutation_rate: float = 0.1) -> Individual:
        """
        Scramble mutation operator.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated individual
        """
        if random.random() > mutation_rate or len(individual.chromosome) < 2:
            return individual.copy()
        
        # Select random segment
        start = random.randint(0, len(individual.chromosome) - 2)
        end = random.randint(start + 1, len(individual.chromosome) - 1)
        
        # Create mutated chromosome
        mutated_chromosome = individual.chromosome.copy()
        segment = mutated_chromosome[start:end + 1]
        random.shuffle(segment)
        mutated_chromosome[start:end + 1] = segment
        
        # Create mutated individual
        mutated = Individual(chromosome=mutated_chromosome)
        return mutated


class AdaptiveMutationOperator:
    """Adaptive mutation operator that adjusts mutation rate based on diversity."""
    
    def __init__(self, base_mutation_rate: float = 0.1, 
                 diversity_threshold: float = 0.1,
                 max_mutation_rate: float = 0.3,
                 use_adaptive_selection: bool = True):
        """
        Initialize adaptive mutation operator.
        
        Args:
            base_mutation_rate: Base mutation rate
            diversity_threshold: Diversity threshold for rate adjustment
            max_mutation_rate: Maximum mutation rate
            use_adaptive_selection: Whether to use adaptive operator selection
        """
        self.base_mutation_rate = base_mutation_rate
        self.diversity_threshold = diversity_threshold
        self.max_mutation_rate = max_mutation_rate
        self.use_adaptive_selection = use_adaptive_selection
        
        # Track operator performance for adaptive selection
        self.operator_success = {'swap': 0, 'inversion': 0, 'insertion': 0, 'scramble': 0}
        self.operator_attempts = {'swap': 0, 'inversion': 0, 'insertion': 0, 'scramble': 0}
        self.last_fitness_before = {}  # Track fitness before mutation
    
    def get_mutation_rate(self, population_diversity: float) -> float:
        """
        Get adaptive mutation rate based on population diversity.
        
        Args:
            population_diversity: Current population diversity
            
        Returns:
            Adaptive mutation rate
        """
        if population_diversity < self.diversity_threshold:
            # Low diversity, increase mutation rate
            return min(self.max_mutation_rate, self.base_mutation_rate * 2)
        else:
            # High diversity, use base mutation rate
            return self.base_mutation_rate
    
    def mutate(self, individual: Individual, population_diversity: float, 
               fitness_before: Optional[float] = None) -> Individual:
        """
        Apply adaptive mutation to individual.
        
        Args:
            individual: Individual to mutate
            population_diversity: Current population diversity
            fitness_before: Fitness before mutation (for tracking operator success)
            
        Returns:
            Mutated individual
        """
        mutation_rate = self.get_mutation_rate(population_diversity)
        
        # Select mutation operator
        if self.use_adaptive_selection:
            operator_name = self._select_operator()
        else:
            operator_name = random.choice(['swap', 'inversion', 'insertion', 'scramble'])
        
        # Map operator name to function
        operator_map = {
            'swap': MutationOperator.swap_mutation,
            'inversion': MutationOperator.inversion_mutation,
            'insertion': MutationOperator.insertion_mutation,
            'scramble': MutationOperator.scramble_mutation
        }
        
        mutation_operator = operator_map[operator_name]
        mutated = mutation_operator(individual, mutation_rate)
        
        # Track operator attempt
        self.operator_attempts[operator_name] += 1
        
        # Note: Success tracking should be done by caller after fitness evaluation
        # Store operator name for later success tracking
        if not hasattr(mutated, '_mutation_operator'):
            mutated._mutation_operator = operator_name
            mutated._fitness_before_mutation = fitness_before
        
        return mutated
    
    def _select_operator(self) -> str:
        """
        Select mutation operator based on success rate.
        
        Returns:
            Operator name
        """
        operators = ['swap', 'inversion', 'insertion', 'scramble']
        
        # Calculate success rates
        success_rates = {}
        for op in operators:
            attempts = self.operator_attempts[op]
            if attempts > 0:
                success_rates[op] = self.operator_success[op] / attempts
            else:
                success_rates[op] = 0.5  # Default for unexplored operators
        
        # 70% exploitation (use best), 30% exploration (random)
        if random.random() < 0.7 and max(success_rates.values()) > 0:
            # Select operator with highest success rate
            return max(success_rates, key=success_rates.get)
        else:
            # Random selection for exploration
            return random.choice(operators)
    
    def record_success(self, individual: Individual, fitness_after: float):
        """
        Record success of mutation operator.
        
        Args:
            individual: Mutated individual
            fitness_after: Fitness after mutation
        """
        if not self.use_adaptive_selection:
            return
        
        if hasattr(individual, '_mutation_operator') and hasattr(individual, '_fitness_before_mutation'):
            operator_name = individual._mutation_operator
            fitness_before = individual._fitness_before_mutation
            
            if fitness_before is not None and fitness_after > fitness_before:
                # Mutation improved fitness
                self.operator_success[operator_name] += 1
