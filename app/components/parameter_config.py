"""
Parameter configuration components for GA optimization.
"""

import streamlit as st
from typing import Dict, Optional
import numpy as np

from src.data_processing.mode_profiles import ModeProfile, get_mode_profile


def render_parameter_config(
    dataset_size: int = 10,
    default_config: Optional[Dict] = None,
    dataset_type: str = "hanoi"
) -> Dict:
    """
    Render parameter configuration UI.
    
    Args:
        dataset_size: Number of customers in dataset
        default_config: Default configuration values
        
    Returns:
        Dictionary with GA configuration parameters
    """
    st.subheader("GA Parameters Configuration")
    mode_profile = get_mode_profile(dataset_type)
    st.caption(f"{mode_profile.label}: {mode_profile.description}")
    
    # Calculate sensible defaults based on dataset size
    if default_config is None:
        default_config = _calculate_defaults(dataset_size, dataset_type, mode_profile)
    
    # Preset selection
    st.markdown("### Preset Configuration")
    preset_options = ["Fast", "Standard", "Benchmark", "Custom"]
    try:
        default_preset_index = preset_options.index(mode_profile.default_ga_preset.capitalize())
    except ValueError:
        default_preset_index = 1  # Standard
    preset = st.radio(
        "Choose a preset:",
        preset_options,
        horizontal=True,
        index=default_preset_index,
        help="Select a preset configuration or customize your own"
    )
    
    if preset != "Custom":
        config = _get_preset_config(preset, dataset_size, dataset_type, mode_profile)
    else:
        config = default_config.copy()
    
    # Basic Parameters
    st.markdown("### Basic Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        population_size = st.slider(
            "Population Size",
            min_value=20,
            max_value=500,
            value=config.get('population_size', 100),
            step=10,
            help="Number of solutions in each generation. Larger values explore more but take longer."
        )
        
        generations = st.slider(
            "Number of Generations",
            min_value=50,
            max_value=5000,
            value=config.get('generations', 1000),
            step=50,
            help="Maximum number of generations. More generations may find better solutions."
        )
    
    with col2:
        num_vehicles = st.number_input(
            "Number of Vehicles",
            min_value=1,
            max_value=50,
            value=config.get('num_vehicles', min(5, max(1, int(np.ceil(dataset_size / 10))))),
            step=1,
            help="Maximum number of vehicles to use"
        )
        
        use_split_algorithm = st.checkbox(
            "Use Split Algorithm",
            value=config.get('use_split_algorithm', True),
            help="Use optimal route splitting (slower but better quality)"
        )
    
    # Advanced Parameters (collapsible)
    with st.expander("Advanced Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            crossover_prob = st.slider(
                "Crossover Probability",
                min_value=0.0,
                max_value=1.0,
                value=config.get('crossover_prob', 0.9),
                step=0.05,
                help="Probability of crossover between parents (0.8-0.95 recommended)"
            )
            
            mutation_prob = st.slider(
                "Mutation Probability",
                min_value=0.0,
                max_value=1.0,
                value=config.get('mutation_prob', 0.1),
                step=0.01,
                help="Probability of mutation (0.05-0.2 recommended)"
            )
        
        with col2:
            tournament_size = st.slider(
                "Tournament Size",
                min_value=2,
                max_value=10,
                value=config.get('tournament_size', 3),
                step=1,
                help="Size of tournament for selection (2-5 recommended)"
            )
            
            elitism_rate = st.slider(
                "Elitism Rate",
                min_value=0.0,
                max_value=0.5,
                value=config.get('elitism_rate', 0.1),
                step=0.05,
                help="Fraction of best solutions to preserve (0.05-0.2 recommended)"
            )
    
    # Build configuration dictionary
    # Penalty weight defaults by dataset type
    default_penalty = config.get('penalty_weight', mode_profile.penalty_weight)

    ga_config = {
        'population_size': population_size,
        'generations': generations,
        'num_vehicles': num_vehicles,
        'use_split_algorithm': use_split_algorithm,
        'crossover_prob': crossover_prob,
        'mutation_prob': mutation_prob,
        'tournament_size': tournament_size,
        'elitism_rate': elitism_rate,
        'penalty_weight': default_penalty,
        'convergence_threshold': 0.001,  # Required for convergence check
        'stagnation_limit': 50  # Required for convergence check
    }
    
    # Validation
    errors = _validate_config(ga_config)
    if errors:
        for error in errors:
            st.error(error)
        return None
    
    # Estimated time
    estimated_time = _estimate_runtime(ga_config, dataset_size)
    st.info(f"Estimated runtime: {estimated_time}")
    
    return ga_config


def _calculate_defaults(
    dataset_size: int,
    dataset_type: str = "hanoi",
    mode_profile: Optional[ModeProfile] = None,
) -> Dict:
    """Calculate sensible defaults based on dataset size."""
    mode_profile = mode_profile or get_mode_profile(dataset_type)
    if dataset_type == "solomon":
        # Solomon benchmarks benefit from larger populations and split algorithm
        if dataset_size <= 50:
            return {
                'population_size': 150,
                'generations': 1500,
                'num_vehicles': 13,  # Default for Solomon 100-customer instances
                'use_split_algorithm': True,
                'penalty_weight': mode_profile.penalty_weight
            }
        else:
            return {
                'population_size': 200,
                'generations': 2000,
                'num_vehicles': 13,  # Default for Solomon 100-customer instances
                'use_split_algorithm': True,
                'penalty_weight': mode_profile.penalty_weight
            }
    else:
        if dataset_size <= 10:
            return {
                'population_size': 50,
                'generations': 500,
                'num_vehicles': max(1, int(np.ceil(dataset_size / 5))),
                'use_split_algorithm': True,
                'penalty_weight': mode_profile.penalty_weight
            }
        elif dataset_size <= 50:
            return {
                'population_size': 100,
                'generations': 1000,
                'num_vehicles': max(1, int(np.ceil(dataset_size / 10))),
                'use_split_algorithm': True,
                'penalty_weight': mode_profile.penalty_weight
            }
        else:
            return {
                'population_size': 150,
                'generations': 1500,
                'num_vehicles': max(1, int(np.ceil(dataset_size / 10))),
                'use_split_algorithm': True,
                'penalty_weight': mode_profile.penalty_weight
            }


def _get_preset_config(
    preset: str,
    dataset_size: int,
    dataset_type: str = "hanoi",
    mode_profile: Optional[ModeProfile] = None,
) -> Dict:
    """Get preset configuration from GA_PRESETS."""
    from config import GA_PRESETS
    
    base = _calculate_defaults(dataset_size, dataset_type, mode_profile)
    
    preset_key = preset.lower()
    if preset_key in GA_PRESETS:
        preset_config = GA_PRESETS[preset_key].copy()
        # Merge with base to ensure num_vehicles and other dataset-specific params are included
        merged = {**base, **preset_config}
        # Ensure penalty_weight uses mode-specific default if None
        if merged.get('penalty_weight') is None:
            merged['penalty_weight'] = mode_profile.penalty_weight if mode_profile else 5000
        return merged
    
    return base


def _validate_config(config: Dict) -> list:
    """Validate configuration parameters."""
    errors = []
    
    if config['population_size'] < 10:
        errors.append("Population size must be at least 10")
    
    if config['generations'] < 10:
        errors.append("Generations must be at least 10")
    
    if not 0 <= config['crossover_prob'] <= 1:
        errors.append("Crossover probability must be between 0 and 1")
    
    if not 0 <= config['mutation_prob'] <= 1:
        errors.append("Mutation probability must be between 0 and 1")
    
    if config['tournament_size'] < 2:
        errors.append("Tournament size must be at least 2")
    
    if not 0 <= config['elitism_rate'] <= 1:
        errors.append("Elitism rate must be between 0 and 1")
    
    return errors


def _estimate_runtime(config: Dict, dataset_size: int) -> str:
    """Estimate runtime based on configuration and profiling data."""
    # Based on profiling: 4.74s per generation for pop=100, n=1000 customers
    # Scale linearly with pop size and quadratically with customer count (approximate)
    base_time_per_gen = 4.74  # seconds for pop=100, n=1000
    pop_factor = config['population_size'] / 100.0
    # Scale customer count: for n=1000, factor=1.0; approximate quadratic scaling
    customer_factor = (dataset_size / 1000.0) ** 1.5  # Slightly less than quadratic
    
    time_per_gen = base_time_per_gen * pop_factor * customer_factor
    total_seconds = time_per_gen * config['generations']
    
    if total_seconds < 60:
        return f"{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        minutes = int(total_seconds / 60)
        return f"{minutes} minutes"
    else:
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

