"""
Optimization service for running GA optimization.
Wraps GeneticAlgorithm and provides clean interface for Streamlit.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms.genetic_algorithm import GeneticAlgorithm, run_genetic_algorithm
from src.models.vrp_model import VRPProblem
from src.models.solution import Individual
from app.core.logger import setup_app_logger
from app.core.exceptions import OptimizationError

logger = setup_app_logger()


class OptimizationService:
    """Service for running VRP optimizations."""
    
    def __init__(self):
        """Initialize optimization service."""
        self.current_ga: Optional[GeneticAlgorithm] = None
        self.is_running = False
        self.should_stop = False
    
    def run_optimization(
        self,
        problem: VRPProblem,
        config: Dict,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[Individual, Dict, list]:
        """
        Run GA optimization with progress tracking.
        
        Args:
            problem: VRP problem instance
            config: GA configuration parameters
            progress_callback: Optional callback function(generation, best_fitness, best_distance)
            
        Returns:
            Tuple of (best_solution, statistics, evolution_data)
        """
        try:
            self.is_running = True
            self.should_stop = False
            
            # Merge config with default GA_CONFIG to ensure all required keys are present
            from config import GA_CONFIG
            merged_config = GA_CONFIG.copy()
            merged_config.update(config)  # Override with user config
            
            # Create GA instance
            ga = GeneticAlgorithm(problem, merged_config)
            self.current_ga = ga
            
            # Use GA's evolve method but with progress tracking
            # We need to override evolve to add progress callbacks
            max_generations = config.get('generations', 1000)
            
            # Initialize population
            logger.info(f"Initializing population of size {merged_config.get('population_size', 100)}")
            ga.initialize_population()
            logger.info("Population initialized successfully")
            
            evolution_data = []
            start_time = time.time()
            actual_generations = 0
            logger.info(f"Starting evolution loop for {max_generations} generations")
            
            # Custom evolution loop with progress tracking
            for generation in range(max_generations):
                if self.should_stop:
                    logger.warning(f"Optimization stopped by user at generation {generation + 1}/{max_generations}")
                    break
                
                logger.debug(f"Starting generation {generation + 1}/{max_generations}")
                gen_start_time = time.time()
                
                try:
                    # Create next generation
                    logger.debug(f"Creating next generation for gen {generation + 1}")
                    ga._create_next_generation()
                    logger.debug(f"Next generation created in {time.time() - gen_start_time:.2f}s")
                    
                    # Update statistics
                    ga._update_statistics()
                    actual_generations = generation + 1
                    
                    # Get current best
                    best_individual = ga.population.get_best_individual()
                    best_fitness = best_individual.fitness
                    best_distance = best_individual.total_distance
                    
                    # Collect evolution data
                    gen_data = {
                        'generation': generation,
                        'best_fitness': best_fitness,
                        'best_distance': best_distance,
                        'avg_fitness': ga.stats.get('avg_fitness_history', [0])[-1] if ga.stats.get('avg_fitness_history') else 0,
                        'diversity': ga._calculate_diversity() if hasattr(ga, '_calculate_diversity') else 0
                    }
                    evolution_data.append(gen_data)
                    
                    # Call progress callback (non-blocking, catch all exceptions)
                    if progress_callback:
                        try:
                            progress_callback(generation + 1, max_generations, best_fitness, best_distance)
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}", exc_info=True)
                    
                    gen_duration = time.time() - gen_start_time
                    logger.info(f"Generation {generation + 1}/{max_generations} completed in {gen_duration:.2f}s - Best distance: {best_distance:.2f}")
                    
                    # Check convergence (only after minimum required generations)
                    if ga._check_convergence():
                        logger.info(f"Convergence reached at generation {generation + 1}/{max_generations}")
                        # Store convergence generation in stats
                        ga.stats['convergence_generation'] = generation + 1
                        break
                        
                except Exception as e:
                    logger.error(f"Error in generation {generation + 1}: {e}", exc_info=True)
                    # Continue to next generation instead of crashing
                    # But log warning if too many consecutive errors
                    if generation > 0 and generation % 50 == 0:
                        logger.warning(f"Multiple errors encountered. Continuing optimization...")
                    continue
            
            # Log completion status
            if actual_generations < max_generations:
                if self.should_stop:
                    logger.warning(f"Optimization stopped early by user at generation {actual_generations}/{max_generations}")
                elif actual_generations < max_generations * 0.9:
                    logger.warning(f"Optimization completed early at generation {actual_generations}/{max_generations} - may indicate issues")
            else:
                logger.info(f"Optimization completed all {max_generations} generations")
            
            # Finalize
            execution_time = time.time() - start_time
            ga.execution_time = execution_time
            ga.stats['generations'] = actual_generations
            ga.generation = actual_generations
            
            best_solution = ga.population.get_best_individual()
            statistics = ga.get_statistics()
            statistics['config_used'] = ga.config.copy()
            statistics['generations_run'] = actual_generations
            statistics['execution_time'] = execution_time
            
            self.is_running = False
            self.current_ga = None
            
            return best_solution, statistics, evolution_data
            
        except Exception as e:
            self.is_running = False
            self.current_ga = None
            logger.error(f"Optimization error: {e}")
            raise OptimizationError(f"Optimization failed: {str(e)}")
    
    def stop_optimization(self):
        """Stop current optimization gracefully."""
        if self.is_running:
            self.should_stop = True
            logger.info("Stop signal sent to optimization")
    
    def get_progress(self) -> Dict:
        """
        Get current optimization progress.
        
        Returns:
            Dictionary with progress information
        """
        if not self.is_running or not self.current_ga:
            return {
                'is_running': False,
                'generation': 0,
                'max_generations': 0,
                'best_fitness': 0,
                'best_distance': 0
            }
        
        best_individual = self.current_ga.population.get_best_individual()
        
        return {
            'is_running': True,
            'generation': self.current_ga.generation,
            'max_generations': self.current_ga.config.get('generations', 1000),
            'best_fitness': best_individual.fitness,
            'best_distance': best_individual.total_distance
        }

