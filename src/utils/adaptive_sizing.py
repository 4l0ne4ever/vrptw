"""
Adaptive parameter sizing for GA based on problem characteristics.
Based on empirical evidence from literature (Potvin 1996, Braysy 2005, Homberger 2005).
"""
from typing import Dict, Tuple, Optional
from config import ADAPTIVE_SIZING_RULES, GA_CONFIG


def calculate_tightness_metrics(problem) -> Dict[str, float]:
    """
    Calculate time window tightness metrics for a problem.

    Args:
        problem: VRP problem instance

    Returns:
        Dictionary containing tightness metrics
    """
    customers = problem.customers
    n_customers = len(customers)

    if n_customers == 0:
        return {
            'tight_ratio': 0.0,
            'very_tight_count': 0,
            'tight_count': 0,
            'difficulty_score': 0.0
        }

    # Analyze time window widths
    very_tight_count = 0
    tight_count = 0

    for customer in customers:
        tw_width = customer.due_date - customer.ready_time

        # Thresholds based on literature (Potvin 1996, Solomon benchmarks)
        if tw_width < 80:
            very_tight_count += 1
        elif tw_width < 120:
            tight_count += 1

    # Calculate metrics
    tight_ratio = (very_tight_count + tight_count) / n_customers
    difficulty_score = (very_tight_count * 4 + tight_count * 2) / n_customers

    return {
        'tight_ratio': tight_ratio,
        'very_tight_count': very_tight_count,
        'tight_count': tight_count,
        'difficulty_score': difficulty_score,
        'n_customers': n_customers
    }


def get_adaptive_parameters(problem, base_config: Optional[Dict] = None) -> Dict:
    """
    Calculate adaptive GA parameters based on problem characteristics.

    Implements evidence-based parameter sizing from:
    - Potvin & Bengio (1996): n=100 benchmarks
    - Thangiah et al. (1994): Tightness impact
    - Braysy & Gendreau (2005): Meta-analysis
    - Homberger & Gehring (2005): Superlinear scaling

    Args:
        problem: VRP problem instance
        base_config: Base configuration (default: GA_CONFIG)

    Returns:
        Dictionary with adapted parameters
    """
    if base_config is None:
        base_config = GA_CONFIG.copy()

    # Check if adaptive sizing is enabled
    if not base_config.get('use_adaptive_sizing', False):
        return base_config

    # Get tightness metrics
    metrics = calculate_tightness_metrics(problem)
    n_customers = metrics['n_customers']
    tight_ratio = metrics['tight_ratio']
    difficulty = metrics['difficulty_score']

    # Adapt population size
    base_pop = base_config.get('population_size', 150)
    adapted_pop = adapt_population_size(n_customers, tight_ratio, base_pop)

    # Adapt generation count
    base_gen = base_config.get('generations', 1500)
    adapted_gen = adapt_generation_count(n_customers, difficulty, base_gen)

    # Adapt mutation rate
    base_mut = base_config.get('mutation_prob', 0.15)
    adapted_mut = adapt_mutation_rate(tight_ratio, base_mut)

    # Adapt tournament size
    adapted_tournament = adapt_tournament_size(adapted_pop)

    # Create adapted config
    adapted_config = base_config.copy()
    adapted_config.update({
        'population_size': adapted_pop,
        'generations': adapted_gen,
        'mutation_prob': adapted_mut,
        'tournament_size': adapted_tournament,
        '_adapted': True,
        '_metrics': metrics
    })

    return adapted_config


def adapt_population_size(n_customers: int, tight_ratio: float, base_pop: int) -> int:
    """
    Adapt population size based on problem size and constraint tightness.

    Evidence:
    - Thangiah (1994): 30% tight needs 2x population
    - Braysy (2005): Recommended ranges by size
    - Homberger (2005): Superlinear scaling

    Args:
        n_customers: Number of customers
        tight_ratio: Ratio of tight time windows
        base_pop: Base population size

    Returns:
        Adapted population size
    """
    rules = ADAPTIVE_SIZING_RULES['population']

    # Apply size multiplier (Homberger 2005)
    size_mult = 1.0
    for min_n, max_n, mult in rules['size_ranges']:
        if min_n <= n_customers <= max_n:
            size_mult = mult
            break

    # Apply tightness multiplier (Thangiah 1994)
    tight_mult = 1.0
    for min_t, max_t, mult in rules['tightness_ranges']:
        if min_t <= tight_ratio < max_t:
            tight_mult = mult
            break

    # Calculate adapted population
    adapted = int(base_pop * size_mult * tight_mult)

    # Apply min/max bounds from literature
    recommendations = rules['size_recommendations']

    # Find nearest size range
    nearest_size = min(recommendations.keys(), key=lambda x: abs(x - n_customers))
    min_pop, max_pop = recommendations[nearest_size]

    # Clamp to recommended range
    adapted = max(min_pop, min(adapted, max_pop))

    return adapted


def adapt_generation_count(n_customers: int, difficulty: float, base_gen: int) -> int:
    """
    Adapt generation count based on problem size and difficulty.

    Evidence:
    - Taillard (1997): Difficulty-based requirements
    - Potvin (1996): n=100 needs 2500 generations
    - Homberger (2005): Size-based minimums

    Args:
        n_customers: Number of customers
        difficulty: Difficulty score (0-4)
        base_gen: Base generation count

    Returns:
        Adapted generation count
    """
    rules = ADAPTIVE_SIZING_RULES['generations']

    # Apply difficulty multiplier (Taillard 1997)
    diff_mult = 1.0
    for min_d, max_d, mult in rules['difficulty_ranges']:
        if min_d <= difficulty < max_d:
            diff_mult = mult
            break

    # Calculate adapted generations
    adapted = int(base_gen * diff_mult)

    # Apply size-based minimum (Homberger 2005)
    minimums = rules['size_minimums']
    nearest_size = min(minimums.keys(), key=lambda x: abs(x - n_customers))
    min_gen = minimums[nearest_size]

    # Ensure meets minimum
    adapted = max(adapted, min_gen)

    return adapted


def adapt_mutation_rate(tight_ratio: float, base_mut: float) -> float:
    """
    Adapt mutation rate based on constraint tightness.

    Evidence:
    - Potvin (1996): Higher mutation for tight TW (0.20 vs 0.15)

    Args:
        tight_ratio: Ratio of tight time windows
        base_mut: Base mutation rate

    Returns:
        Adapted mutation rate
    """
    rules = ADAPTIVE_SIZING_RULES['mutation']

    # Add bonus for tight constraints
    if tight_ratio > 0.30:  # >30% tight
        bonus = rules['tight_bonus']
    else:
        bonus = 0.0

    adapted = base_mut + bonus
    adapted = min(adapted, rules['max'])  # Cap at maximum

    return adapted


def adapt_tournament_size(population_size: int) -> int:
    """
    Adapt tournament size to population size.

    Evidence:
    - Standard practice: 3-5% of population

    Args:
        population_size: Population size

    Returns:
        Adapted tournament size
    """
    rules = ADAPTIVE_SIZING_RULES['tournament']

    # Calculate as ratio of population
    adapted = int(population_size * rules['ratio'])

    # Apply min/max bounds
    adapted = max(rules['min'], min(adapted, rules['max']))

    return adapted


def get_config_summary(config: Dict) -> str:
    """
    Generate human-readable summary of configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Summary string
    """
    if not config.get('_adapted', False):
        return "Using base configuration (adaptive sizing disabled)"

    metrics = config.get('_metrics', {})

    summary = []
    summary.append("=== ADAPTIVE CONFIGURATION ===")
    summary.append(f"Problem size: {metrics.get('n_customers', 'N/A')} customers")
    summary.append(f"Tight ratio: {metrics.get('tight_ratio', 0.0):.1%}")
    summary.append(f"Difficulty: {metrics.get('difficulty_score', 0.0):.2f}")
    summary.append("")
    summary.append("Adapted parameters:")
    summary.append(f"  Population: {config.get('population_size')}")
    summary.append(f"  Generations: {config.get('generations')}")
    summary.append(f"  Mutation: {config.get('mutation_prob'):.3f}")
    summary.append(f"  Tournament: {config.get('tournament_size')}")
    summary.append(f"  Elitism: {config.get('elitism_rate'):.3f}")

    return "\n".join(summary)
