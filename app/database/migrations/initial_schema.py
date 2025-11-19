"""
Initial database schema migration.
Creates all tables defined in models.
"""

from app.config.database import init_db

def run_migration():
    """Run initial schema migration."""
    print("Creating database schema...")
    init_db()
    print("Database schema created successfully!")

if __name__ == "__main__":
    run_migration()

