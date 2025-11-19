"""
Database initialization script.
Creates database tables and optionally seeds sample data.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.database import init_db, engine
from app.database.models import Base
from app.core.logger import setup_app_logger
from app.database import seed_data

logger = setup_app_logger()


def initialize_database(seed: bool = True):
    """
    Initialize database by creating all tables.
    
    Args:
        seed: Whether to seed sample data
    """
    try:
        logger.info("Initializing database...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully!")
        
        # Check if tables already have data
        from app.config.database import SessionLocal
        from app.database.models import Dataset
        
        db = SessionLocal()
        try:
            existing_datasets = db.query(Dataset).count()
            if existing_datasets > 0:
                logger.info(f"Database already contains {existing_datasets} datasets. Skipping seed.")
                seed = False
        finally:
            db.close()
        
        # Seed sample data if requested and database is empty
        if seed:
            logger.info("Seeding sample data...")
            seed_data.seed_sample_datasets()
            seed_data.seed_sample_configurations()
            logger.info("Sample data seeded successfully!")
        
        logger.info("Database initialization completed!")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize database")
    parser.add_argument("--no-seed", action="store_true", help="Skip seeding sample data")
    args = parser.parse_args()
    
    initialize_database(seed=not args.no_seed)

