"""
Basic tests for database operations.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.database import SessionLocal, init_db, reset_db
from app.database import crud, models
import json


@pytest.fixture(scope="function")
def db_session():
    """Create a test database session."""
    # Use in-memory SQLite for testing
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.models import Base
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()


def test_create_dataset(db_session):
    """Test creating a dataset."""
    dataset_data = {
        "depot": {"id": 0, "x": 105.8542, "y": 21.0285, "demand": 0},
        "customers": [{"id": 1, "x": 105.8400, "y": 21.0200, "demand": 10}],
        "vehicle_capacity": 200,
        "num_vehicles": 5
    }
    
    dataset = crud.create_dataset(
        db=db_session,
        name="Test Dataset",
        description="Test description",
        type="hanoi_mockup",
        data_json=json.dumps(dataset_data)
    )
    
    assert dataset.id is not None
    assert dataset.name == "Test Dataset"
    assert dataset.type == "hanoi_mockup"


def test_get_dataset(db_session):
    """Test retrieving a dataset."""
    # Create a dataset first
    dataset_data = {"depot": {}, "customers": []}
    dataset = crud.create_dataset(
        db=db_session,
        name="Test Dataset",
        description="Test",
        type="hanoi_mockup",
        data_json=json.dumps(dataset_data)
    )
    
    # Retrieve it
    retrieved = crud.get_dataset(db_session, dataset.id)
    
    assert retrieved is not None
    assert retrieved.id == dataset.id
    assert retrieved.name == "Test Dataset"


def test_get_datasets(db_session):
    """Test retrieving multiple datasets."""
    # Create multiple datasets
    for i in range(3):
        dataset_data = {"depot": {}, "customers": []}
        crud.create_dataset(
            db=db_session,
            name=f"Dataset {i}",
            description="Test",
            type="hanoi_mockup",
            data_json=json.dumps(dataset_data)
        )
    
    datasets = crud.get_datasets(db_session)
    assert len(datasets) == 3


def test_delete_dataset(db_session):
    """Test deleting a dataset."""
    # Create a dataset
    dataset_data = {"depot": {}, "customers": []}
    dataset = crud.create_dataset(
        db=db_session,
        name="Test Dataset",
        description="Test",
        type="hanoi_mockup",
        data_json=json.dumps(dataset_data)
    )
    
    # Delete it
    result = crud.delete_dataset(db_session, dataset.id)
    assert result is True
    
    # Verify it's deleted
    retrieved = crud.get_dataset(db_session, dataset.id)
    assert retrieved is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

