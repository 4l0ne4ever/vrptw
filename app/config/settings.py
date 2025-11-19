"""
Application settings and configuration.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Database configuration
DATABASE_DIR = BASE_DIR / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "vrp_app.db"
DATABASE_BACKUP_DIR = DATABASE_DIR / "backups"

# Ensure directories exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Upload configuration
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# Export configuration
EXPORT_DIR = BASE_DIR / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Application settings
APP_NAME = "VRP-GA Optimization System"
APP_VERSION = "1.0.0"

# Default GA parameters (can be overridden by user)
DEFAULT_GA_CONFIG = {
    'population_size': 100,
    'generations': 1000,
    'crossover_prob': 0.9,
    'mutation_prob': 0.15,
    'tournament_size': 5,
    'elitism_rate': 0.15,
    'use_split_algorithm': True
}

# Hanoi bounds for validation
HANOI_BOUNDS = {
    'min_lat': 20.7,
    'max_lat': 21.4,
    'min_lon': 105.3,
    'max_lon': 106.0
}

# Color palette for routes (high-contrast against map tiles)
ROUTE_COLORS = [
    '#FF3B30', '#FF9500', '#FFD60A', '#0A84FF', '#5856D6',
    '#AF52DE', '#FF2D55', '#00C7BE', '#34C759', '#5AC8FA',
    '#FF6B81', '#F97316', '#FB7185', '#6366F1', '#14B8A6'
]

# Routing settings
OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org")
OSRM_TIMEOUT = float(os.getenv("OSRM_TIMEOUT", "5"))
USE_REAL_ROUTES = os.getenv("USE_REAL_ROUTES", "true").lower() == "true"
# OSRM batch size: max points per request (default 100, can be reduced to 50 if URL length issues occur)
OSRM_MAX_BATCH_SIZE = int(os.getenv("OSRM_MAX_BATCH_SIZE", "100"))
# OSRM parallel processing: max concurrent batch requests (default 3 to avoid rate limiting)
OSRM_MAX_CONCURRENCY = int(os.getenv("OSRM_MAX_CONCURRENCY", "3"))

# Cache warming settings for Solomon datasets
SOLOMON_WARM_CACHE_ENABLED = os.getenv("SOLOMON_WARM_CACHE_ENABLED", "true").lower() == "true"
SOLOMON_WARM_CACHE_ASYNC = os.getenv("SOLOMON_WARM_CACHE_ASYNC", "true").lower() == "true"
_SOLOMON_WARM_CACHE_LIMIT_RAW = os.getenv("SOLOMON_WARM_CACHE_LIMIT")
if _SOLOMON_WARM_CACHE_LIMIT_RAW is not None:
    try:
        SOLOMON_WARM_CACHE_LIMIT = int(_SOLOMON_WARM_CACHE_LIMIT_RAW)
    except ValueError:
        SOLOMON_WARM_CACHE_LIMIT = None
else:
    SOLOMON_WARM_CACHE_LIMIT = None

