"""
Migration to add binary distance cache columns.
"""

import sqlite3
from pathlib import Path

from app.config.settings import DATABASE_PATH


TABLE_NAME = "distance_matrix_cache"
DISTANCE_BINARY_COL = "distance_matrix_binary"
SERIALIZATION_COL = "serialization_version"


def _column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table});")
    return any(row[1] == column for row in cursor.fetchall())


def run_migration():
    db_path = Path(DATABASE_PATH)
    if not db_path.exists():
        print(f"Database not found at {db_path}. Skipping migration.")
        return

    print(f"Migrating database at {db_path}...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        if not _column_exists(cursor, TABLE_NAME, DISTANCE_BINARY_COL):
            print(f"Adding {DISTANCE_BINARY_COL} column...")
            cursor.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN {DISTANCE_BINARY_COL} BLOB"
            )
        else:
            print(f"{DISTANCE_BINARY_COL} already exists. Skipping.")

        if not _column_exists(cursor, TABLE_NAME, SERIALIZATION_COL):
            print(f"Adding {SERIALIZATION_COL} column with default 1...")
            cursor.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN {SERIALIZATION_COL} INTEGER NOT NULL DEFAULT 1"
            )
        else:
            print(f"{SERIALIZATION_COL} already exists. Ensuring defaults...")
            cursor.execute(
                f"""
                UPDATE {TABLE_NAME}
                SET {SERIALIZATION_COL} = 1
                WHERE {SERIALIZATION_COL} IS NULL
                """
            )

        conn.commit()

    print("Migration completed successfully!")


if __name__ == "__main__":
    run_migration()


