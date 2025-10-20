"""
Solution representation for VRP problems.
Defines Individual and Population classes for GA.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import copy


@dataclass
class Individual:
    """Represents a single solution (individual) in the GA population."""
    chromosome: List[int] = field(default_factory=list)
    fitness: float = 0.0
    routes: List[List[int]] = field(default_factory=list)
    total_distance: float = 0.0
    is_valid: bool = True
    penalty: float = 0.0
    
    def __post_init__(self):
        """Initialize empty lists if not provided."""
        if not self.chromosome:
            self.chromosome = []
        if not self.routes:
            self.routes = []
    
    def copy(self) -> 'Individual':
        """Create a deep copy of the individual."""
        return Individual(
            chromosome=self.chromosome.copy(),
            fitness=self.fitness,
            routes=[route.copy() for route in self.routes],
            total_distance=self.total_distance,
            is_valid=self.is_valid,
            penalty=self.penalty
        )
    
    def get_route_count(self) -> int:
        """Get number of routes in the solution."""
        return len([route for route in self.routes if route])
    
    def get_customer_count(self) -> int:
        """Get total number of customers in the solution."""
        return len(self.chromosome)
    
    def is_empty(self) -> bool:
        """Check if the individual is empty."""
        return len(self.chromosome) == 0
    
    def clear(self):
        """Clear all data."""
        self.chromosome.clear()
        self.routes.clear()
        self.fitness = 0.0
        self.total_distance = 0.0
        self.is_valid = True
        self.penalty = 0.0
    
    def to_dict(self) -> Dict:
        """Convert individual to dictionary."""
        return {
            'chromosome': self.chromosome,
            'fitness': float(self.fitness) if self.fitness is not None else None,
            'routes': self.routes,
            'total_distance': float(self.total_distance) if self.total_distance is not None else None,
            'is_valid': bool(self.is_valid),
            'penalty': float(self.penalty) if self.penalty is not None else None,
            'route_count': int(self.get_route_count()),
            'customer_count': int(self.get_customer_count())
        }


class Population:
    """Represents a population of individuals in the GA."""
    
    def __init__(self, individuals: Optional[List[Individual]] = None):
        """
        Initialize population.
        
        Args:
            individuals: List of individuals (empty if None)
        """
        self.individuals = individuals or []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.avg_fitness = 0.0
        self.diversity = 0.0
        self.fitness_history: List[float] = []
        self.best_fitness_history: List[float] = []
    
    def add_individual(self, individual: Individual):
        """Add an individual to the population."""
        self.individuals.append(individual)
        self._update_statistics()
    
    def remove_individual(self, index: int):
        """Remove individual by index."""
        if 0 <= index < len(self.individuals):
            del self.individuals[index]
            self._update_statistics()
    
    def get_individual(self, index: int) -> Optional[Individual]:
        """Get individual by index."""
        if 0 <= index < len(self.individuals):
            return self.individuals[index]
        return None
    
    def get_best_individual(self) -> Optional[Individual]:
        """Get the best individual in the population."""
        if not self.individuals:
            return None
        
        return max(self.individuals, key=lambda ind: ind.fitness)
    
    def get_worst_individual(self) -> Optional[Individual]:
        """Get the worst individual in the population."""
        if not self.individuals:
            return None
        
        return min(self.individuals, key=lambda ind: ind.fitness)
    
    def sort_by_fitness(self, reverse: bool = True):
        """Sort individuals by fitness."""
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=reverse)
        self._update_statistics()
    
    def get_fitness_values(self) -> List[float]:
        """Get list of fitness values."""
        return [ind.fitness for ind in self.individuals]
    
    def get_best_fitness(self) -> float:
        """Get best fitness value."""
        if not self.individuals:
            return 0.0
        return max(ind.fitness for ind in self.individuals)
    
    def get_avg_fitness(self) -> float:
        """Get average fitness value."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)
    
    def get_worst_fitness(self) -> float:
        """Get worst fitness value."""
        if not self.individuals:
            return 0.0
        return min(ind.fitness for ind in self.individuals)
    
    def calculate_diversity(self) -> float:
        """
        Calculate population diversity based on chromosome differences.
        
        Returns:
            Diversity measure (0-1, higher means more diverse)
        """
        if len(self.individuals) < 2:
            return 0.0
        
        total_differences = 0
        total_comparisons = 0
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                chrom1 = self.individuals[i].chromosome
                chrom2 = self.individuals[j].chromosome
                
                if len(chrom1) != len(chrom2):
                    continue
                
                # Count position differences
                differences = sum(1 for a, b in zip(chrom1, chrom2) if a != b)
                total_differences += differences
                total_comparisons += len(chrom1)
        
        if total_comparisons == 0:
            return 0.0
        
        return total_differences / total_comparisons
    
    def _update_statistics(self):
        """Update population statistics."""
        if not self.individuals:
            self.best_individual = None
            self.avg_fitness = 0.0
            self.diversity = 0.0
            return
        
        # Update best individual
        self.best_individual = self.get_best_individual()
        
        # Update average fitness
        self.avg_fitness = self.get_avg_fitness()
        
        # Update diversity
        self.diversity = self.calculate_diversity()
        
        # Update fitness history
        self.fitness_history.append(self.avg_fitness)
        self.best_fitness_history.append(self.get_best_fitness())
    
    def next_generation(self):
        """Move to next generation."""
        self.generation += 1
    
    def get_size(self) -> int:
        """Get population size."""
        return len(self.individuals)
    
    def is_empty(self) -> bool:
        """Check if population is empty."""
        return len(self.individuals) == 0
    
    def clear(self):
        """Clear all individuals."""
        self.individuals.clear()
        self.best_individual = None
        self.avg_fitness = 0.0
        self.diversity = 0.0
        self.generation = 0
        self.fitness_history.clear()
        self.best_fitness_history.clear()
    
    def get_statistics(self) -> Dict:
        """Get population statistics."""
        return {
            'size': self.get_size(),
            'generation': self.generation,
            'best_fitness': self.get_best_fitness(),
            'avg_fitness': self.get_avg_fitness(),
            'worst_fitness': self.get_worst_fitness(),
            'diversity': self.diversity,
            'fitness_std': np.std(self.get_fitness_values()) if self.individuals else 0.0
        }
    
    def copy(self) -> 'Population':
        """Create a deep copy of the population."""
        copied_individuals = [ind.copy() for ind in self.individuals]
        new_pop = Population(copied_individuals)
        new_pop.generation = self.generation
        new_pop.best_individual = self.best_individual.copy() if self.best_individual else None
        new_pop.avg_fitness = self.avg_fitness
        new_pop.diversity = self.diversity
        new_pop.fitness_history = self.fitness_history.copy()
        new_pop.best_fitness_history = self.best_fitness_history.copy()
        return new_pop
    
    def apply_elitism(self, elite_count: int) -> List[Individual]:
        """
        Select elite individuals for next generation.
        
        Args:
            elite_count: Number of elite individuals to select
            
        Returns:
            List of elite individuals
        """
        if not self.individuals:
            return []
        
        # Sort by fitness (descending)
        sorted_individuals = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        
        # Select top individuals
        elite_count = min(elite_count, len(sorted_individuals))
        elite = []
        
        for i in range(elite_count):
            elite.append(sorted_individuals[i].copy())
        
        return elite
    
    def replace_individuals(self, new_individuals: List[Individual]):
        """
        Replace current individuals with new ones.
        
        Args:
            new_individuals: List of new individuals
        """
        self.individuals = new_individuals
        self._update_statistics()
    
    def to_dict(self) -> Dict:
        """Convert population to dictionary."""
        return {
            'size': self.get_size(),
            'generation': self.generation,
            'statistics': self.get_statistics(),
            'individuals': [ind.to_dict() for ind in self.individuals]
        }
