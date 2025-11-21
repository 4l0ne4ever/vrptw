# Configuration parameters for VRP-GA System
# Based on Table 2.18 in main.tex (thesis)

# Genetic Algorithm Configuration
GA_CONFIG = {
    'population_size': 100,      # Standard for VRP problems (thesis: Table 2.18)
    'generations': 1000,         # Sufficient for convergence (thesis: Table 2.18)
    'crossover_prob': 0.9,       # High crossover rate (thesis: Table 2.18)
    'mutation_prob': 0.15,        # Moderate mutation rate (thesis: Table 2.18)
    'tournament_size': 5,         # Tournament selection (thesis: Table 2.18)
    'elitism_rate': 0.10,         # Keep top 10% (reduced from 15% to prevent premature convergence)
    'adaptive_mutation': False,   # Fixed mutation rate (not mentioned in thesis)
    'convergence_threshold': 0.001,
    'stagnation_limit': 50,        # Stop if no improvement for 50 generations
    'use_split_algorithm': True,  # Enable Split Algorithm (Prins 2004) for optimal route splitting
    'local_search_prob': 0.3,     # Probability of applying 2-opt/TW repair to offspring
    'local_search_iterations': 30,  # Max iterations for local search
    'force_full_generations': True,
    'tw_repair': {
        'enabled': True,
        'max_iterations': 20,  # Base iterations
        'max_iterations_soft': 10,  # Soft mode iterations
        'violation_weight': 50.0,
        'max_relocations_per_route': 2,
        'max_routes_to_try': None,
        'max_positions_to_try': None,
        'max_routes_soft_limit': 5,
        'max_positions_soft_limit': 6,
        'lateness_soft_threshold': 5000.0,  # Threshold for soft mode
        'lateness_skip_threshold': 100000.0,  # Never skip repair
        'apply_in_decoder': False,  # Disabled: expensive
        'apply_in_decoder_solomon': False,
        'apply_after_genetic_operators': False,  # DISABLED for speed: old version didn't have this
        'apply_after_genetic_operators_prob': 0.0,  # Disabled
        'apply_after_local_search': True,
        'apply_post_generation': True,  # Apply to top individuals
        'apply_post_generation_prob': 1.0,  # Always apply (but to fewer individuals)
        'post_generation_top_k_ratio': 0.1,  # Top 10% only (reduced from 20% for speed)
        'apply_after_local_search_solomon': True,
        'apply_on_final_solution': True
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
        # Same as GA_CONFIG defaults
        'population_size': 100,
        'generations': 1000,
        'crossover_prob': 0.9,
        'mutation_prob': 0.15,
        'tournament_size': 5,
        'elitism_rate': 0.10,
        'local_search_prob': 0.1,
        'local_search_iterations': 10,
        'use_split_algorithm': True,
        'penalty_weight': None,  # Will use mode-specific default (1200 Hanoi, 5000 Solomon)
        # Estimated runtime: ~79 minutes for 1000 customers
    },
    'benchmark': {
        'population_size': 100,
        'generations': 1000,
        'crossover_prob': 0.9,
        'mutation_prob': 0.15,
        'tournament_size': 5,
        'elitism_rate': 0.10,
        'local_search_prob': 0.15,  # More local search for quality
        'local_search_iterations': 15,
        'use_split_algorithm': True,
        'penalty_weight': None,  # Will use mode-specific default (1200 Hanoi, 5000 Solomon)
        # Estimated runtime: ~85 minutes for 1000 customers
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
