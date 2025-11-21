"""
Mode-specific configuration classes for Hanoi and Solomon modes.
Implements pragmatic hybrid strategy: same algorithm, different intensity.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HanoiConfig:
    """
    Configuration for real-world Hanoi routing.
    
    Philosophy:
    - Feasibility is important but not absolute (can negotiate)
    - Speed matters (users wait for results)
    - Soft time windows reflect business reality
    - Capacity is critical (must satisfy)
    """
    # Time window handling
    time_window_buffer: float = 15.0  # minutes flexibility
    allow_time_flexibility: bool = True
    time_window_penalty_hard: float = 10000.0  # High but not critical
    time_window_penalty_soft: float = 100.0  # Low - acceptable
    
    # Capacity focus
    capacity_penalty: float = 50000.0  # High - must satisfy
    
    # Performance priorities
    max_generations: int = 1000
    population_size: int = 150
    initialization_tw_aware_ratio: float = 0.5  # 50% TW-aware, 50% random for diversity
    
    # Repair strategy
    repair_intensity: str = "moderate"  # Don't over-optimize
    max_repair_iterations: int = 50
    repair_accept_near_feasible: bool = True  # Accept 1-2 violations if distance good
    
    # Acceptance criteria
    accept_near_feasible: bool = True  # Accept solutions with 1-2 soft violations
    feasibility_threshold: float = 0.98  # 98% compliance OK
    max_acceptable_violations: int = 2  # At most 2 violations acceptable
    max_acceptable_hard_violations: int = 0  # No hard violations acceptable


@dataclass
class SolomonConfig:
    """
    Configuration for academic Solomon benchmarking.
    
    Philosophy:
    - Strict time window handling (academic standards)
    - Must be perfectly feasible
    - More iterations for quality
    - TW-aware initialization for feasibility
    """
    # Time window handling
    time_window_buffer: float = 0.0  # No flexibility
    allow_time_flexibility: bool = False
    time_window_penalty_hard: float = 10000.0  # High enough to enforce, low enough for gradient
    time_window_penalty_soft: float = 5000.0  # High - minimize
    
    # Capacity focus
    capacity_penalty: float = 100000.0  # Very high
    
    # Performance priorities
    max_generations: int = 2500  # More iterations
    population_size: int = 200  # Larger population
    initialization_tw_aware_ratio: float = 1.0  # 100% TW-aware
    
    # Repair strategy
    repair_intensity: str = "aggressive"  # Pursue feasibility
    max_repair_iterations: int = 200
    repair_accept_near_feasible: bool = False  # Must be perfect
    
    # Acceptance criteria
    accept_near_feasible: bool = False  # Must be perfectly feasible
    feasibility_threshold: float = 1.0  # 100% compliance required
    max_acceptable_violations: int = 0  # Zero violations required
    max_acceptable_hard_violations: int = 0  # Zero hard violations required


def get_mode_config(dataset_type: str) -> HanoiConfig | SolomonConfig:
    """
    Get mode-specific configuration.
    
    Args:
        dataset_type: "hanoi" or "solomon"
        
    Returns:
        HanoiConfig or SolomonConfig instance
    """
    dataset_type = (dataset_type or "").strip().lower()
    
    if dataset_type.startswith("solomon"):
        return SolomonConfig()
    else:
        # Default to Hanoi for real-world routing
        return HanoiConfig()

