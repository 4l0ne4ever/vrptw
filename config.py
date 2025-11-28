# Configuration parameters for VRP-GA System
# Based on Table 2.18 in main.tex (thesis)

# Genetic Algorithm Configuration
# UPDATED: Based on empirical evidence from literature (Potvin 1996, Braysy 2005, Homberger 2005)
# Evidence: 3200+ experiments across 5 peer-reviewed studies
GA_CONFIG = {
    # ADAPTIVE SIZING: Parameters adjust based on problem size and constraint tightness
    'use_adaptive_sizing': True,  # Enable adaptive parameter sizing

    # BASE CONFIG (for n<=30, loose constraints)
    'population_size': 150,      # UPDATED from 100 (Thangiah 1994: 73% success with tight TW)
    'generations': 1500,         # UPDATED from 1000 (Taillard 1997: medium difficulty)
    'crossover_prob': 0.9,       # ✓ Optimal (Potvin 1996)
    'mutation_prob': 0.15,       # Will adapt based on constraint tightness
    'tournament_size': 5,        # ✓ Optimal (standard 5% of pop)
    'elitism_rate': 0.05,        # UPDATED from 0.10 (Whitley 1991: better quality)

    # ADAPTIVE PARAMETERS (auto-adjust in runtime)
    'adaptive_mutation': True,   # Enable adaptive mutation based on TW tightness
    'convergence_threshold': 0.005,  # UPDATED from 0.001 (0.3% → 0.5% CV for convergence)
    'stagnation_limit': 50,      # Stop if no improvement for 50 generations
    'use_split_algorithm': True, # Enable Split Algorithm (Prins 2004)
    'local_search_prob': 0.3,    # Probability of applying local search
    'local_search_iterations': 30,
    'force_full_generations': False,  # UPDATED from True (enable early stopping)

    # EARLY STOPPING PATIENCE (NEW)
    # Formula: patience = max(100, generations * patience_ratio)
    # Default 10% = wait 10% of max_generations without improvement
    'patience_ratio': 0.10,  # 10% of max generations (adjustable: 0.05-0.20)
    'min_improvement_threshold': 0.001,  # 0.1% relative improvement to count as progress
    # NOTE: Patience is calculated dynamically based on max generations
    # Example: 1500 gens → patience = 150, 3000 gens → patience = 300
    'tw_repair': {
        'enabled': False,  # ❌ DISABLED: Let Strong Repair (Phase 3) handle violations!
        'max_iterations': 100,  # Not used (GA doesn't repair)
        'max_iterations_soft': 10,  # Not used
        'violation_weight': 50.0,
        'max_relocations_per_route': 3,
        'max_routes_to_try': None,
        'max_positions_to_try': None,
        'max_positions_soft_limit': 6,
        'max_routes_soft_limit': 5,
        'lateness_soft_threshold': float('inf'),
        'lateness_skip_threshold': float('inf'),
        'apply_in_decoder': False,  # ❌ DISABLED: GA evolves freely
        'apply_in_decoder_solomon': False,  # ❌ DISABLED: Penalty guides, not repair
        'apply_after_genetic_operators': False,  # ❌ DISABLED: No repair during evolution
        'apply_after_genetic_operators_prob': 0.0,  # ❌ DISABLED: Pure GA
        'apply_after_local_search': False,  # ❌ DISABLED
        'apply_post_generation': False,  # ❌ DISABLED
        'apply_post_generation_prob': 0.0,
        'post_generation_top_k_ratio': 0.0,
        'apply_after_local_search_solomon': False,
        'apply_on_final_solution': False  # ❌ DISABLED: Strong Repair will fix everything!
    }
}

# ADAPTIVE SIZING RULES
# Based on empirical evidence: Potvin (1996), Gehring (1999), Braysy (2005), Homberger (2005)
ADAPTIVE_SIZING_RULES = {
    # Population sizing based on problem size and constraint tightness
    # Formula: Pop = base_pop * size_multiplier * tightness_multiplier
    'population': {
        # Size-based multipliers (Homberger 2005: superlinear scaling)
        'size_ranges': [
            (0, 30, 1.0),      # n <= 30: base
            (31, 75, 1.3),     # 31-75: +30%
            (76, 150, 1.7),    # 76-150: +70%
            (151, float('inf'), 2.0)  # >150: +100%
        ],
        # Tightness-based multipliers (Thangiah 1994)
        # tight_ratio = (very_tight + tight) / total_customers
        'tightness_ranges': [
            (0.0, 0.15, 1.0),   # <15% tight: base
            (0.15, 0.30, 1.2),  # 15-30%: +20% (log-normal 20)
            (0.30, 0.50, 1.5),  # 30-50%: +50% (log-normal 100)
            (0.50, 1.0, 1.8)    # >50%: +80%
        ],
        # Recommended values by size (Braysy 2005)
        'size_recommendations': {
            20: (150, 200),    # (min, max) for n=20
            50: (180, 250),
            100: (200, 300),
            200: (300, 400)
        }
    },

    # Generation sizing (Taillard 1997, Potvin 1996)
    'generations': {
        # Difficulty-based (calculated from constraint tightness)
        # Difficulty = (very_tight*4 + tight*2) / n_customers
        'difficulty_ranges': [
            (0.0, 1.0, 1.0),    # Easy: base (mostly normal TWs)
            (1.0, 1.5, 1.3),    # Medium: +30% (some tight TWs, e.g. 1.2)
            (1.5, 2.5, 1.7),    # Hard: +70% (many tight TWs, e.g. 1.88)
            (2.5, float('inf'), 2.0)  # Extreme: +100% (mostly tight)
        ],
        # Size-based minimum (Homberger 2005)
        'size_minimums': {
            20: 1500,
            50: 2000,
            100: 2500,
            200: 3500
        }
    },

    # Mutation rate (Potvin 1996: higher for tight TW)
    'mutation': {
        'base': 0.15,
        'tight_bonus': 0.05,  # Add 0.05 for tight constraints
        'max': 0.25
    },

    # Tournament size (adaptive to population)
    'tournament': {
        'ratio': 0.03,  # 3% of population
        'min': 3,
        'max': 10
    }
}

# GA Preset Configurations
GA_PRESETS = {
    'fast': {
        'population_size': 50,      # Reduced for speed
        'generations': 500,          # Fewer generations
        'crossover_prob': 0.9,
        'mutation_prob': 0.15,
        'tournament_size': 5,
        'elitism_rate': 0.10,
        'local_search_prob': 0.05,  # Reduced local search
        'local_search_iterations': 5,
        'use_split_algorithm': True,
        'penalty_weight': None,  # Will use mode-specific default (1200 Hanoi, 5000 Solomon)
        # Estimated runtime: ~20 minutes for 1000 customers
    },
    'standard': {
        # Balanced quality and speed
        'population_size': 100,
        'generations': 1000,
        'crossover_prob': 0.9,
        'mutation_prob': 0.15,
        'tournament_size': 5,
        'elitism_rate': 0.10,
        'local_search_prob': 0.05,   # Reduced from 0.2 to 0.05 for speed
        'local_search_iterations': 10,  # Reduced from 20 to 10 for speed
        'use_split_algorithm': True,
        'penalty_weight': None,  # Will use mode-specific default (1200 Hanoi, 5000 Solomon)
        # Estimated runtime: ~30-40 minutes for 100 customers
    },
    'benchmark': {
        # High quality for Solomon benchmarks
        'population_size': 100,
        'generations': 1000,
        'crossover_prob': 0.9,
        'mutation_prob': 0.15,
        'tournament_size': 5,
        'elitism_rate': 0.10,
        'local_search_prob': 0.1,   # Reduced from 0.3 to 0.1 for speed (10% during evolution)
        'local_search_iterations': 15,  # Reduced from 30 to 15 for speed
        'use_split_algorithm': True,
        'penalty_weight': None,  # Will use mode-specific default (1200 Hanoi, 5000 Solomon)
        # Estimated runtime: ~40-50 minutes for 100 customers
    },
    'production': {
        # Production-ready: Maximum quality for both modes
        'population_size': 100,
        'generations': 1000,
        'crossover_prob': 0.9,
        'mutation_prob': 0.15,
        'tournament_size': 5,
        'elitism_rate': 0.10,
        'local_search_prob': 0.4,   # High exploitation for production quality
        'local_search_iterations': 40,  # Deep local optimization
        'use_split_algorithm': True,
        'penalty_weight': None,  # Will use mode-specific default (1200 Hanoi, 5000 Solomon)
        # Estimated runtime: ~120 minutes for 1000 customers
        # Use for: Final production runs, customer deliverables, benchmark submissions
    }
}

# VRP Problem Configuration
# Based on Table 2.14 in main.tex (thesis) and realistic Hanoi conditions
VRP_CONFIG = {
    'vehicle_capacity': 200,     # 200 units (~500kg van) (thesis: Table 2.14)
    'num_vehicles': 25,          # Default number of vehicles (thesis: Table 2.14)
    'num_vehicles_formula': 'ceil(n/8)',  # Formula: ⌈n/8⌉ (average 8 customers per vehicle)
    'traffic_factor': 1.3,       # Base traffic factor for Hanoi (normal hours)
    'penalty_weight': 5000,      # Penalty for constraint violations (increased to ensure infeasible solutions are heavily penalized)
    'depot_id': 0,               # Depot node ID
    'service_time': 12,          # Average service time in minutes (realistic Hanoi: 10-15 min)
    'service_time_min': 10,      # Minimum service time (minutes)
    'service_time_max': 15,      # Maximum service time (minutes)
    'time_window_start': 480,    # 8:00 = 480 minutes (start of working day)
    'time_window_end': 1200,     # 20:00 = 1200 minutes (end of working day)
    
    # Adaptive Traffic Factor Configuration
    'use_adaptive_traffic': True,  # Enable adaptive traffic factor based on time of day
    'traffic_factor_peak': 1.8,   # Peak hours (7-9h, 17-19h)
    'traffic_factor_normal': 1.2, # Normal hours (10-16h, 20-22h)
    'traffic_factor_low': 1.0,    # Low hours (22h-7h)
    'peak_hours': [(420, 540), (1020, 1140)],  # Peak hours in minutes: (7-9h, 17-19h)
    
    # Shipping cost configs (Ahamove model - updated for 2024-2025)
    'use_waiting_fee': True,       # Enable waiting fee (realistic Hanoi)
    'waiting_fee_per_minute': 300,  # VND per minute (18,000 VND/hour)
    'waiting_fee_per_hour': 18000,  # VND per hour
    'free_waiting_time': 15,      # Free waiting time in minutes
    'cod_fee_rate': 0.006,        # 0.6% COD fee (thesis: Section 2.3.3)
    'cod_ratio': 0.75,            # 75% of orders have COD (realistic Hanoi)
    'order_value_min': 100000,    # Minimum order value (VND)
    'order_value_max': 800000,    # Maximum order value (VND) - reduced from 1M
    
    # Additional cost factors (realistic Hanoi)
    'cost_per_km': 1.0,           # Base cost per km (for KPI calculation)
    'fuel_cost_per_km': 4000,     # Fuel cost per km (VND)
    'driver_cost_per_hour': 40000, # Driver cost per hour (VND)
    'vehicle_fixed_cost': 75000,   # Fixed cost per vehicle (VND)
}

# Mockup Data Generation Configuration
# Based on Table 2.14 in main.tex (thesis)
MOCKUP_CONFIG = {
    'n_customers': 50,           # Default size (thesis: can use 25, 50, 100)
    'demand_lambda': 7,           # Poisson(λ=7) for demand (thesis: Table 2.14)
    'demand_min': 1,             # Minimum demand (thesis: Table 2.14)
    'demand_max': 20,            # Maximum demand (thesis: Table 2.14)
    'area_bounds': (0, 100),     # Cartesian 2D space [0,100]×[0,100] (thesis: Table 2.14)
    'clustering': 'kmeans',      # Clustering method (kmeans/random/radial) (thesis: Table 2.14)
    'n_clusters': 5,             # Number of clusters
    'seed': 42,                  # Random seed for reproducibility
    'service_time': 600,         # 10 minutes = 600 seconds (thesis: Table 2.14 - "10 phút")
    'time_window_width': 200     # Time window width in time units
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'colors': ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', '#A52A2A', '#FFC0CB', '#808080'],
    'marker_size': 50,
    'line_width': 2,
    'font_size': 12
}

# File Paths
PATHS = {
    'data_raw': 'data/raw/',
    'data_processed': 'data/processed/',
    'results': 'results/',
    'solomon_dataset': 'data/solomon_dataset/'
}
