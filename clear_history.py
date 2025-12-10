#!/usr/bin/env python3
"""
Script to clear optimization history (best results and runs) from database.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.database import SessionLocal
from app.database.crud import get_all_best_results, get_optimization_runs
from app.database.models import BestResult, OptimizationRun
from sqlalchemy import delete

def clear_history():
    """Clear all optimization history."""
    db = SessionLocal()
    try:
        # Count before deletion
        best_results_count = db.query(BestResult).count()
        runs_count = db.query(OptimizationRun).count()
        
        print("=" * 80)
        print("CLEARING OPTIMIZATION HISTORY")
        print("=" * 80)
        print(f"Best results to delete: {best_results_count}")
        print(f"Optimization runs to delete: {runs_count}")
        
        if best_results_count == 0 and runs_count == 0:
            print("\n Database is already empty. Nothing to delete.")
            return
        
        # Delete best results
        if best_results_count > 0:
            deleted_best = db.query(BestResult).delete()
            print(f"\n Deleted {deleted_best} best result(s)")
        
        # Delete optimization runs
        if runs_count > 0:
            deleted_runs = db.query(OptimizationRun).delete()
            print(f" Deleted {deleted_runs} optimization run(s)")
        
        # Commit changes
        db.commit()
        
        print("\n" + "=" * 80)
        print(" History cleared successfully!")
        print("=" * 80)
        print("\nYou can now run new optimizations and they will be saved fresh.")
        
    except Exception as e:
        print(f"\n Error clearing history: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear optimization history from database")
    parser.add_argument('--yes', '-y', action='store_true', 
                       help='Skip confirmation and delete immediately')
    args = parser.parse_args()
    
    if not args.yes:
        # Confirm before deletion
        print("  WARNING: This will delete ALL optimization history!")
        print("   - All best results will be deleted")
        print("   - All optimization runs will be deleted")
        print("\nThis action cannot be undone.")
        print("\nUse --yes flag to skip confirmation: python3 clear_history.py --yes")
        
        try:
            response = input("\nAre you sure you want to continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("\n Operation cancelled.")
                sys.exit(0)
        except EOFError:
            print("\n Cannot read input. Use --yes flag: python3 clear_history.py --yes")
            sys.exit(1)
    
    clear_history()

