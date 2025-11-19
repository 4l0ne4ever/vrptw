"""
Database configuration and setup.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
from app.config.settings import DATABASE_PATH

# Create database engine
# Using StaticPool for SQLite to handle multiple threads
engine = create_engine(
    f"sqlite:///{DATABASE_PATH}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def get_db():
    """
    Get database session.
    Use as dependency in FastAPI-style or context manager.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database by creating all tables."""
    from app.database import models  # Import models to register them
    Base.metadata.create_all(bind=engine)

def reset_db():
    """Reset database by dropping and recreating all tables."""
    Base.metadata.drop_all(bind=engine)
    init_db()

