"""
Migration script to make distance_matrix_json nullable.
This allows version 2 (binary) serialization to work without errors.
"""

import sqlite3
import os
from app.config.settings import DATABASE_PATH


def run_migration():
    """Make distance_matrix_json column nullable."""
    db_path = DATABASE_PATH
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check current schema
        cursor.execute("PRAGMA table_info(distance_matrix_cache)")
        columns = {col[1]: col for col in cursor.fetchall()}
        
        if 'distance_matrix_json' not in columns:
            print("distance_matrix_json column not found!")
            return
        
        col_info = columns['distance_matrix_json']
        is_nullable = col_info[3] == 0  # 0 = NOT NULL, 1 = NULL allowed
        
        if is_nullable:
            print("distance_matrix_json is already nullable. No migration needed.")
            return
        
        print("Making distance_matrix_json nullable...")
        
        # SQLite doesn't support ALTER COLUMN directly, so we need to:
        # 1. Create new table with correct schema
        # 2. Copy data
        # 3. Drop old table
        # 4. Rename new table
        
        # Create new table with nullable distance_matrix_json
        cursor.execute("""
            CREATE TABLE distance_matrix_cache_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key VARCHAR(64) NOT NULL UNIQUE,
                coordinates_json TEXT NOT NULL,
                distance_matrix_json TEXT NULL,
                distance_matrix_binary BLOB NULL,
                serialization_version INTEGER NOT NULL DEFAULT 1,
                dataset_type VARCHAR(50) NOT NULL,
                use_real_routes BOOLEAN NOT NULL DEFAULT 0,
                num_points INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_distance_matrix_cache_cache_key_new ON distance_matrix_cache_new(cache_key)")
        
        # Copy data
        cursor.execute("""
            INSERT INTO distance_matrix_cache_new 
            (id, cache_key, coordinates_json, distance_matrix_json, distance_matrix_binary, 
             serialization_version, dataset_type, use_real_routes, num_points, created_at, updated_at)
            SELECT 
                id, cache_key, coordinates_json, distance_matrix_json, distance_matrix_binary,
                serialization_version, dataset_type, use_real_routes, num_points, created_at, updated_at
            FROM distance_matrix_cache
        """)
        
        # Drop old table
        cursor.execute("DROP TABLE distance_matrix_cache")
        
        # Rename new table
        cursor.execute("ALTER TABLE distance_matrix_cache_new RENAME TO distance_matrix_cache")
        
        # Recreate indexes with original names
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_distance_matrix_cache_cache_key ON distance_matrix_cache(cache_key)")
        
        conn.commit()
        print("Migration completed successfully! distance_matrix_json is now nullable.")
        
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_migration()

