"""
Seed database with sample data for testing and demo.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.database import SessionLocal
from app.database import crud
from app.core.logger import setup_app_logger

logger = setup_app_logger()


def seed_sample_datasets():
    """Create sample datasets for testing."""
    db = SessionLocal()
    
    try:
        # Sample Hanoi dataset with 20 customers
        hanoi_customers = []
        base_lat, base_lon = 21.0285, 105.8542  # Hoan Kiem Lake
        
        import random
        for i in range(1, 21):
            # Generate customers around Hanoi center
            random.seed(i)  # For reproducibility
            lat_offset = (random.random() - 0.5) * 0.1
            lon_offset = (random.random() - 0.5) * 0.1
            hanoi_customers.append({
                "id": i,
                "x": base_lon + lon_offset,
                "y": base_lat + lat_offset,
                "demand": random.randint(5, 30),
                "ready_time": 0.0,
                "due_date": 1000.0,
                "service_time": 10.0
            })
        
        hanoi_data = {
            "depot": {
                "id": 0,
                "x": base_lon,
                "y": base_lat,
                "demand": 0,
                "ready_time": 0.0,
                "due_date": 1000.0,
                "service_time": 0.0
            },
            "customers": hanoi_customers,
            "vehicle_capacity": 200,
            "num_vehicles": 5
        }
        
        hanoi_metadata = {
            "num_customers": 20,
            "total_demand": sum(c["demand"] for c in hanoi_customers),
            "bounds": {
                "min_lat": min(c["y"] for c in hanoi_customers),
                "max_lat": max(c["y"] for c in hanoi_customers),
                "min_lon": min(c["x"] for c in hanoi_customers),
                "max_lon": max(c["x"] for c in hanoi_customers)
            }
        }
        
        crud.create_dataset(
            db=db,
            name="Sample Hanoi Dataset (20 customers)",
            description="Sample dataset for Hanoi delivery optimization with 20 customers",
            type="hanoi_mockup",
            data_json=json.dumps(hanoi_data),
            metadata_json=json.dumps(hanoi_metadata)
        )
        
        # Sample Solomon dataset (simplified)
        solomon_customers = []
        for i in range(1, 11):
            solomon_customers.append({
                "id": i,
                "x": i * 10.0,
                "y": i * 10.0,
                "demand": i * 5,
                "ready_time": 0.0,
                "due_date": 1000.0,
                "service_time": 10.0
            })
        
        solomon_data = {
            "depot": {
                "id": 0,
                "x": 0.0,
                "y": 0.0,
                "demand": 0,
                "ready_time": 0.0,
                "due_date": 1000.0,
                "service_time": 0.0
            },
            "customers": solomon_customers,
            "vehicle_capacity": 200,
            "num_vehicles": 5
        }
        
        solomon_metadata = {
            "num_customers": 10,
            "total_demand": sum(c["demand"] for c in solomon_customers),
            "bounds": {
                "min_lat": 0.0,
                "max_lat": 100.0,
                "min_lon": 0.0,
                "max_lon": 100.0
            }
        }
        
        crud.create_dataset(
            db=db,
            name="Sample Solomon Dataset (10 customers)",
            description="Sample dataset for Solomon benchmark with 10 customers",
            type="solomon",
            data_json=json.dumps(solomon_data),
            metadata_json=json.dumps(solomon_metadata)
        )
        
        logger.info("Sample datasets created successfully!")
        
    except Exception as e:
        logger.error(f"Error seeding sample datasets: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def seed_sample_configurations():
    """Create sample parameter configurations."""
    db = SessionLocal()
    
    try:
        # Fast preset
        fast_config = {
            "population_size": 50,
            "generations": 100,
            "crossover_prob": 0.9,
            "mutation_prob": 0.2,
            "tournament_size": 3,
            "elitism_rate": 0.1,
            "use_split_algorithm": False
        }
        
        crud.create_saved_configuration(
            db=db,
            name="Fast",
            description="Quick optimization with smaller population and fewer generations",
            parameters_json=json.dumps(fast_config),
            is_default=False
        )
        
        # Balanced preset
        balanced_config = {
            "population_size": 100,
            "generations": 1000,
            "crossover_prob": 0.9,
            "mutation_prob": 0.15,
            "tournament_size": 5,
            "elitism_rate": 0.15,
            "use_split_algorithm": True
        }
        
        crud.create_saved_configuration(
            db=db,
            name="Balanced",
            description="Balance between speed and solution quality",
            parameters_json=json.dumps(balanced_config),
            is_default=True
        )
        
        # Best Quality preset
        best_quality_config = {
            "population_size": 200,
            "generations": 2000,
            "crossover_prob": 0.95,
            "mutation_prob": 0.1,
            "tournament_size": 7,
            "elitism_rate": 0.2,
            "use_split_algorithm": True
        }
        
        crud.create_saved_configuration(
            db=db,
            name="Best Quality",
            description="Maximum solution quality with larger population and more generations",
            parameters_json=json.dumps(best_quality_config),
            is_default=False
        )
        
        logger.info("Sample configurations created successfully!")
        
    except Exception as e:
        logger.error(f"Error seeding sample configurations: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Seeding database with sample data...")
    seed_sample_datasets()
    seed_sample_configurations()
    logger.info("Database seeding completed!")
