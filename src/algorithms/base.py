"""
Abstract base classes for VRP algorithms.
Defines interfaces for algorithms and optimizers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from src.models.vrp_model import VRPProblem
from src.models.solution import Individual


class BaseAlgorithm(ABC):
    """Base class for all VRP algorithms."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize algorithm.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
    
    @abstractmethod
    def solve(self) -> Individual:
        """
        Solve VRP problem and return solution.
        
        Returns:
            Best solution found
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return algorithm statistics.
        
        Returns:
            Dictionary with algorithm statistics
        """
        pass


class BaseOptimizer(ABC):
    """Base class for local search optimizers."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize optimizer.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
    
    @abstractmethod
    def optimize(self, solution: Individual) -> Individual:
        """
        Optimize solution.
        
        Args:
            solution: Solution to optimize
            
        Returns:
            Optimized solution
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return optimizer statistics.
        
        Returns:
            Dictionary with optimizer statistics
        """
        pass


class BaseDecoder(ABC):
    """Base class for route decoders."""
    
    def __init__(self, problem: VRPProblem):
        """
        Initialize decoder.
        
        Args:
            problem: VRP problem instance
        """
        self.problem = problem
    
    @abstractmethod
    def decode_chromosome(self, chromosome: list) -> list:
        """
        Decode chromosome to routes.
        
        Args:
            chromosome: Chromosome representation
            
        Returns:
            List of routes
        """
        pass
    
    @abstractmethod
    def encode_routes(self, routes: list) -> list:
        """
        Encode routes to chromosome.
        
        Args:
            routes: List of routes
            
        Returns:
            Chromosome representation
        """
        pass

