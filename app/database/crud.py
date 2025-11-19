"""
CRUD operations for database models.
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from app.database.models import Dataset, OptimizationRun, RouteDetail, SavedConfiguration, DistanceMatrixCache


# Dataset CRUD
def create_dataset(db: Session, name: str, description: str, type: str, 
                   data_json: str, metadata_json: Optional[str] = None) -> Dataset:
    """Create a new dataset."""
    dataset = Dataset(
        name=name,
        description=description,
        type=type,
        data_json=data_json,
        metadata_json=metadata_json
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def get_dataset(db: Session, dataset_id: int) -> Optional[Dataset]:
    """Get dataset by ID."""
    return db.query(Dataset).filter(Dataset.id == dataset_id).first()


def get_datasets(db: Session, skip: int = 0, limit: int = 100, 
                 type: Optional[str] = None) -> List[Dataset]:
    """Get all datasets with optional filtering."""
    query = db.query(Dataset)
    if type:
        query = query.filter(Dataset.type == type)
    return query.offset(skip).limit(limit).all()


def update_dataset(db: Session, dataset_id: int, **kwargs) -> Optional[Dataset]:
    """Update dataset."""
    dataset = get_dataset(db, dataset_id)
    if dataset:
        for key, value in kwargs.items():
            if hasattr(dataset, key):
                setattr(dataset, key, value)
        dataset.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(dataset)
    return dataset


def delete_dataset(db: Session, dataset_id: int) -> bool:
    """Delete dataset."""
    dataset = get_dataset(db, dataset_id)
    if dataset:
        db.delete(dataset)
        db.commit()
        return True
    return False


# OptimizationRun CRUD
def create_optimization_run(db: Session, dataset_id: int, name: str,
                           parameters_json: str, results_json: str,
                           notes: Optional[str] = None,
                           status: str = "completed") -> OptimizationRun:
    """Create a new optimization run."""
    run = OptimizationRun(
        dataset_id=dataset_id,
        name=name,
        notes=notes,
        parameters_json=parameters_json,
        results_json=results_json,
        status=status,
        started_at=datetime.utcnow() if status == "running" else None,
        completed_at=datetime.utcnow() if status == "completed" else None
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_optimization_run(db: Session, run_id: int) -> Optional[OptimizationRun]:
    """Get optimization run by ID."""
    return db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()


def get_optimization_runs(db: Session, skip: int = 0, limit: int = 100,
                          dataset_id: Optional[int] = None) -> List[OptimizationRun]:
    """Get all optimization runs with optional filtering."""
    query = db.query(OptimizationRun)
    if dataset_id:
        query = query.filter(OptimizationRun.dataset_id == dataset_id)
    return query.order_by(OptimizationRun.created_at.desc()).offset(skip).limit(limit).all()


def update_optimization_run(db: Session, run_id: int, **kwargs) -> Optional[OptimizationRun]:
    """Update optimization run."""
    run = get_optimization_run(db, run_id)
    if run:
        for key, value in kwargs.items():
            if hasattr(run, key):
                setattr(run, key, value)
        if 'status' in kwargs and kwargs['status'] == 'completed':
            run.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(run)
    return run


def delete_optimization_run(db: Session, run_id: int) -> bool:
    """Delete optimization run."""
    run = get_optimization_run(db, run_id)
    if run:
        db.delete(run)
        db.commit()
        return True
    return False


# RouteDetail CRUD
def create_route_detail(db: Session, run_id: int, route_number: int,
                       sequence_json: str, total_distance: float,
                       total_load: float, utilization_percentage: float,
                       statistics_json: Optional[str] = None) -> RouteDetail:
    """Create a new route detail."""
    route = RouteDetail(
        run_id=run_id,
        route_number=route_number,
        sequence_json=sequence_json,
        total_distance=total_distance,
        total_load=total_load,
        utilization_percentage=utilization_percentage,
        statistics_json=statistics_json
    )
    db.add(route)
    db.commit()
    db.refresh(route)
    return route


def get_route_details(db: Session, run_id: int) -> List[RouteDetail]:
    """Get all route details for an optimization run."""
    return db.query(RouteDetail).filter(RouteDetail.run_id == run_id).order_by(RouteDetail.route_number).all()


# SavedConfiguration CRUD
def create_saved_configuration(db: Session, name: str, parameters_json: str,
                              description: Optional[str] = None,
                              is_default: bool = False) -> SavedConfiguration:
    """Create a new saved configuration."""
    config = SavedConfiguration(
        name=name,
        description=description,
        parameters_json=parameters_json,
        is_default=is_default
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def get_saved_configurations(db: Session) -> List[SavedConfiguration]:
    """Get all saved configurations."""
    return db.query(SavedConfiguration).order_by(SavedConfiguration.is_default.desc(), SavedConfiguration.name).all()


def get_default_configuration(db: Session) -> Optional[SavedConfiguration]:
    """Get default configuration."""
    return db.query(SavedConfiguration).filter(SavedConfiguration.is_default == True).first()


# DistanceMatrixCache CRUD
def create_distance_matrix_cache(db: Session, cache_key: str, coordinates_json: str,
                                 distance_matrix_json: Optional[str] = None,
                                 distance_matrix_binary: Optional[bytes] = None,
                                 serialization_version: int = 1,
                                 dataset_type: str = "hanoi",
                                 use_real_routes: bool = False, 
                                 num_points: int = 0) -> DistanceMatrixCache:
    """Create or update distance matrix cache entry."""
    # Check if cache entry exists
    existing = db.query(DistanceMatrixCache).filter(DistanceMatrixCache.cache_key == cache_key).first()
    
    if existing:
        # Update existing entry
        existing.coordinates_json = coordinates_json
        existing.distance_matrix_json = distance_matrix_json
        existing.distance_matrix_binary = distance_matrix_binary
        existing.serialization_version = serialization_version
        existing.dataset_type = dataset_type
        existing.use_real_routes = use_real_routes
        existing.num_points = num_points
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing
    else:
        # Create new entry
        cache = DistanceMatrixCache(
            cache_key=cache_key,
            coordinates_json=coordinates_json,
            distance_matrix_json=distance_matrix_json,
            distance_matrix_binary=distance_matrix_binary,
            serialization_version=serialization_version,
            dataset_type=dataset_type,
            use_real_routes=use_real_routes,
            num_points=num_points
        )
        db.add(cache)
        db.commit()
        db.refresh(cache)
        return cache


def get_distance_matrix_cache(db: Session, cache_key: str) -> Optional[DistanceMatrixCache]:
    """Get distance matrix cache by cache key."""
    return db.query(DistanceMatrixCache).filter(DistanceMatrixCache.cache_key == cache_key).first()


def delete_distance_matrix_cache(db: Session, cache_key: str) -> bool:
    """Delete distance matrix cache entry."""
    cache = get_distance_matrix_cache(db, cache_key)
    if cache:
        db.delete(cache)
        db.commit()
        return True
    return False

