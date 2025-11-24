"""
Advanced optimization modules for VRP.

This package contains high-performance optimization components:
- Matrix preprocessing and neighbor lists
- O(1) evaluation with Vidal's concatenation logic
- Strong repair operators
- Large Neighborhood Search (LNS)
"""

from .matrix_preprocessor import MatrixPreprocessor
from .neighbor_lists import NeighborListBuilder
from .vidal_evaluator import VidalEvaluator, VidalNode
from .strong_repair import StrongRepair
from .lns_optimizer import LNSOptimizer

__all__ = ['MatrixPreprocessor', 'NeighborListBuilder', 'VidalEvaluator', 'VidalNode', 'StrongRepair', 'LNSOptimizer']
