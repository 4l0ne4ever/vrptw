"""
Application-wide constants.
"""

# File upload limits
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ['.json']

# Hanoi coordinate bounds
HANOI_BOUNDS = {
    'min_lat': 20.7,
    'max_lat': 21.4,
    'min_lon': 105.3,
    'max_lon': 106.0
}

# Route colors for visualization
ROUTE_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F1948A', '#85C1E2', '#F8C471', '#82E0AA', '#F7DC6F'
]

# Default GA parameters
DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATIONS = 1000
DEFAULT_CROSSOVER_PROB = 0.9
DEFAULT_MUTATION_PROB = 0.15
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_ELITISM_RATE = 0.15

