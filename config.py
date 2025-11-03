# Configuration parameters for VRP-GA System
# Based on Table 2.18 in main.tex (thesis)

# Genetic Algorithm Configuration
GA_CONFIG = {
    'population_size': 100,      # Standard for VRP problems (thesis: Table 2.18)
    'generations': 1000,         # Sufficient for convergence (thesis: Table 2.18)
    'crossover_prob': 0.9,       # High crossover rate (thesis: Table 2.18)
    'mutation_prob': 0.15,        # Moderate mutation rate (thesis: Table 2.18)
    'tournament_size': 5,         # Tournament selection (thesis: Table 2.18)
    'elitism_rate': 0.10,         # Keep top 10% (thesis: Table 2.18)
    'adaptive_mutation': False,   # Fixed mutation rate (not mentioned in thesis)
    'convergence_threshold': 0.001,
    'stagnation_limit': 50,        # Stop if no improvement for 50 generations
    'use_split_algorithm': True  # Enable Split Algorithm (Prins 2004) for optimal route splitting
}

# VRP Problem Configuration
# Based on Table 2.14 in main.tex (thesis)
VRP_CONFIG = {
    'vehicle_capacity': 200,     # 200 units (~500kg van) (thesis: Table 2.14)
    'num_vehicles': 25,          # Default number of vehicles (thesis: Table 2.14)
    'traffic_factor': 1.0,       # No congestion (thesis assumption)
    'penalty_weight': 5000,      # Penalty for constraint violations (increased to ensure infeasible solutions are heavily penalized)
    'depot_id': 0,               # Depot node ID
    # Shipping cost configs (Ahamove model - Section 2.3.3)
    'use_waiting_fee': False,    # Waiting cost = 0 (thesis assumption)
    'cod_fee_rate': 0.006        # 0.6% COD fee (thesis: Section 2.3.3)
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
