"""
Migration script to create best_results table.
Run this once to add the new table to your database.

Usage:
  python migrate_best_results.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from app.config.database import engine, Base
    from app.database.models import BestResult
    
    print("Creating best_results table...")
    Base.metadata.create_all(engine, tables=[BestResult.__table__])
    print("✓ Migration complete! best_results table created.")
except ImportError as e:
    print(f"⚠ Migration requires dependencies: {e}")
    print("Install requirements: pip install -r requirements.txt")
except Exception as e:
    print(f"✗ Migration failed: {e}")

