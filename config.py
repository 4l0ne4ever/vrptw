# Configuration parameters for VRP-GA System

# Genetic Algorithm Configuration
GA_CONFIG = {
    'population_size': 15,  # Optimized for performance
    'generations': 3,       # Optimized for performance
    'crossover_prob': 0.9,
    'mutation_prob': 0.15,
    'tournament_size': 5,
    'elitism_rate': 0.15,
    'adaptive_mutation': True,
    'convergence_threshold': 0.001,
    'stagnation_limit': 50
}

# VRP Problem Configuration
VRP_CONFIG = {
    'vehicle_capacity': 200,
    'num_vehicles': 25,
    'traffic_factor': 1.0,
    'penalty_weight': 1000,
    'depot_id': 0,
    # Shipping cost configs
    'use_waiting_fee': False,
    'cod_fee_rate': 0.006  # default 0.6%
}

# Mockup Data Generation Configuration
MOCKUP_CONFIG = {
    'n_customers': 50,
    'demand_lambda': 7,
    'demand_min': 1,
    'demand_max': 20,
    'area_bounds': (0, 100),  # x, y coordinate bounds
    'clustering': 'kmeans',  # random, kmeans, radial
    'n_clusters': 5,
    'seed': 42,
    'service_time': 90,  # seconds
    'time_window_width': 200  # time units
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
