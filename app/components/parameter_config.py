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
    dataset_type: str = "hanoi",
    problem = None
) -> Dict:
    """
    Render parameter configuration UI.

    Args:
        dataset_size: Number of customers in dataset
        default_config: Default configuration values
        dataset_type: Type of dataset (hanoi/solomon)
        problem: Optional VRP problem instance for adaptive sizing

    Returns:
        Dictionary with GA configuration parameters
    """
    st.subheader("GA Parameters Configuration")
    mode_profile = get_mode_profile(dataset_type)
    st.caption(f"{mode_profile.label}: {mode_profile.description}")

    # Calculate sensible defaults based on dataset size
    if default_config is None:
        default_config = _calculate_defaults(dataset_size, dataset_type, mode_profile)

    # ADAPTIVE SIZING: Calculate recommended parameters based on problem characteristics
    adaptive_config = None
    tightness_metrics = None
    if problem is not None:
        try:
            from src.utils.adaptive_sizing import get_adaptive_parameters, calculate_tightness_metrics
            from config import GA_CONFIG

            # Calculate tightness metrics
            tightness_metrics = calculate_tightness_metrics(problem)

            # Get adaptive parameters
            base_config = GA_CONFIG.copy()
            base_config.update(default_config)  # Merge with UI defaults
            adaptive_config = get_adaptive_parameters(problem, base_config)

            # Display adaptive sizing info
            if adaptive_config.get('_adapted', False):
                st.success("✨ **Adaptive Parameter Sizing Active** (Evidence-based)")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Tight Time Windows",
                        f"{tightness_metrics['tight_ratio']:.1%}",
                        help="Percentage of customers with tight time windows"
                    )
                with col2:
                    st.metric(
                        "Difficulty Score",
                        f"{tightness_metrics['difficulty_score']:.2f}",
                        help="Problem difficulty based on constraint tightness"
                    )
                with col3:
                    st.metric(
                        "Problem Size",
                        f"{tightness_metrics['n_customers']} customers"
                    )

                # Use adaptive config as default
                default_config = adaptive_config

                st.info(f"""
                📊 **Recommended Parameters** (based on {tightness_metrics['n_customers']} customers, {tightness_metrics['tight_ratio']:.1%} tight):
                - Population: **{adaptive_config['population_size']}** (adapted from base)
                - Generations: **{adaptive_config['generations']}** (adapted from base)
                - Mutation: **{adaptive_config['mutation_prob']:.3f}** (adjusted for tightness)
                - Tournament: **{adaptive_config['tournament_size']}** (scaled to population)

                *Evidence: Potvin (1996), Thangiah (1994), Bräysy (2005)*
                """)
        except Exception as e:
            st.warning(f"Adaptive sizing failed: {e}. Using static defaults.")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
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
                value=config.get('mutation_prob', 0.15),  # UPDATED from 0.1 to 0.15
                step=0.01,
                help="Probability of mutation (0.15-0.25 recommended, adaptive based on TW tightness)"
            )
        
        with col2:
            tournament_size = st.slider(
                "Tournament Size",
                min_value=2,
                max_value=10,
                value=config.get('tournament_size', 5),  # UPDATED from 3 to 5
                step=1,
                help="Size of tournament for selection (3-7 recommended, ~3% of population)"
            )
            
            elitism_rate = st.slider(
                "Elitism Rate",
                min_value=0.0,
                max_value=0.5,
                value=config.get('elitism_rate', 0.05),  # UPDATED from 0.1 to 0.05
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
    """
    Calculate sensible defaults based on dataset size.

    UPDATED 2025-11-28: Now uses evidence-based base config:
    - Base population: 150 (was 100) - Thangiah 1994
    - Base generations: 1500 (was 1000) - Taillard 1997
    - Base elitism: 0.05 (was 0.10) - Whitley 1991
    """
    mode_profile = mode_profile or get_mode_profile(dataset_type)
    if dataset_type == "solomon":
        # Solomon benchmarks benefit from larger populations and split algorithm
        if dataset_size <= 50:
            return {
                'population_size': 150,
                'generations': 1500,
                'num_vehicles': 13,  # Default for Solomon 100-customer instances
                'use_split_algorithm': True,
                'elitism_rate': 0.05,  # UPDATED from 0.10
                'penalty_weight': mode_profile.penalty_weight
            }
        else:
            return {
                'population_size': 200,
                'generations': 2000,
                'num_vehicles': 13,  # Default for Solomon 100-customer instances
                'use_split_algorithm': True,
                'elitism_rate': 0.05,  # UPDATED from 0.10
                'penalty_weight': mode_profile.penalty_weight
            }
    else:
        if dataset_size <= 10:
            return {
                'population_size': 50,
                'generations': 500,
                'num_vehicles': max(1, int(np.ceil(dataset_size / 5))),
                'use_split_algorithm': True,
                'elitism_rate': 0.05,  # UPDATED from 0.10
                'penalty_weight': mode_profile.penalty_weight
            }
        elif dataset_size <= 50:
            return {
                'population_size': 150,  # UPDATED from 100 (Thangiah 1994: +50%)
                'generations': 1500,     # UPDATED from 1000 (Taillard 1997: +50%)
                'num_vehicles': max(1, int(np.ceil(dataset_size / 10))),
                'use_split_algorithm': True,
                'elitism_rate': 0.05,  # UPDATED from 0.10 (Whitley 1991: -50%)
                'penalty_weight': mode_profile.penalty_weight
            }
        else:
            return {
                'population_size': 200,  # UPDATED from 150 (scales for large instances)
                'generations': 2000,     # UPDATED from 1500 (scales for large instances)
                'num_vehicles': max(1, int(np.ceil(dataset_size / 10))),
                'use_split_algorithm': True,
                'elitism_rate': 0.05,  # UPDATED from 0.10 (Whitley 1991: -50%)
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

