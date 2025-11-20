"""
Database package initialization.
"""

from app.database.models import Dataset, OptimizationRun, RouteDetail, SavedConfiguration, BestResult
from app.database.crud import (
    create_dataset, get_dataset, get_datasets, update_dataset, delete_dataset,
    create_optimization_run, get_optimization_run, get_optimization_runs,
    update_optimization_run, delete_optimization_run,
    create_route_detail, get_route_details,
    create_saved_configuration, get_saved_configurations, get_default_configuration,
    get_best_result, get_all_best_results, create_or_update_best_result, delete_best_result
)

__all__ = [
    'Dataset', 'OptimizationRun', 'RouteDetail', 'SavedConfiguration', 'BestResult',
    'create_dataset', 'get_dataset', 'get_datasets', 'update_dataset', 'delete_dataset',
    'create_optimization_run', 'get_optimization_run', 'get_optimization_runs',
    'update_optimization_run', 'delete_optimization_run',
    'create_route_detail', 'get_route_details',
    'create_saved_configuration', 'get_saved_configurations', 'get_default_configuration',
    'get_best_result', 'get_all_best_results', 'create_or_update_best_result', 'delete_best_result'
]

