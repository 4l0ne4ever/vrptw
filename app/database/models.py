"""
SQLAlchemy models for database schema.
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
from app.config.database import Base


class Dataset(Base):
    """Dataset model for storing VRP problem instances."""
    
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    type = Column(String(50), nullable=False)  # 'solomon' or 'hanoi_mockup'
    data_json = Column(Text, nullable=False)  # Complete VRP problem data as JSON
    metadata_json = Column(Text, nullable=True)  # Additional metadata as JSON
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    optimization_runs = relationship("OptimizationRun", back_populates="dataset", cascade="all, delete-orphan")


class OptimizationRun(Base):
    """Optimization run model for storing GA execution results."""
    
    __tablename__ = "optimization_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    notes = Column(Text, nullable=True)
    parameters_json = Column(Text, nullable=False)  # GA parameters as JSON
    results_json = Column(Text, nullable=False)  # Optimization results as JSON
    status = Column(String(50), default="completed", nullable=False)  # pending, running, completed, failed, stopped
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="optimization_runs")
    route_details = relationship("RouteDetail", back_populates="optimization_run", cascade="all, delete-orphan")


class RouteDetail(Base):
    """Route detail model for storing individual route information."""
    
    __tablename__ = "route_details"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("optimization_runs.id"), nullable=False, index=True)
    route_number = Column(Integer, nullable=False)
    sequence_json = Column(Text, nullable=False)  # Customer IDs sequence as JSON array
    total_distance = Column(Float, nullable=False)
    total_load = Column(Float, nullable=False)
    utilization_percentage = Column(Float, nullable=False)
    statistics_json = Column(Text, nullable=True)  # Additional route statistics as JSON
    
    # Relationships
    optimization_run = relationship("OptimizationRun", back_populates="route_details")


class SavedConfiguration(Base):
    """Saved configuration model for storing parameter presets."""
    
    __tablename__ = "saved_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    parameters_json = Column(Text, nullable=False)  # GA parameters as JSON
    is_default = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DistanceMatrixCache(Base):
    """Distance matrix cache model for storing pre-calculated distance matrices."""
    
    __tablename__ = "distance_matrix_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(64), nullable=False, unique=True, index=True)  # MD5 hash of coordinates + routing info
    coordinates_json = Column(Text, nullable=False)  # Coordinates as JSON array
    distance_matrix_json = Column(Text, nullable=True)  # Distance matrix as JSON (legacy, version 1)
    distance_matrix_binary = Column(LargeBinary, nullable=True)  # Distance matrix as binary (pickle, version 2)
    serialization_version = Column(Integer, default=1, nullable=False)  # 1 = JSON, 2 = binary
    dataset_type = Column(String(50), nullable=False)  # 'hanoi' or 'solomon'
    use_real_routes = Column(Boolean, default=False, nullable=False)  # Whether OSRM was used
    num_points = Column(Integer, nullable=False)  # Number of points in matrix
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

