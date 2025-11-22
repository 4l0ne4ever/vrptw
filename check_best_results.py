#!/usr/bin/env python3
"""
Script to check best results in database and debug why results might not be showing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.database import SessionLocal
from app.database.crud import get_all_best_results, get_dataset
from datetime import datetime

def check_best_results():
    """Check all best results in database."""
    db = SessionLocal()
    try:
        best_results = get_all_best_results(db, limit=1000)
        
        print("=" * 80)
        print(f"BEST RESULTS IN DATABASE (Total: {len(best_results)})")
        print("=" * 80)
        
        if not best_results:
            print("❌ No best results found in database!")
            return
        
        for i, result in enumerate(best_results, 1):
            # Get dataset info
            dataset = get_dataset(db, result.dataset_id)
            dataset_type = dataset.type if dataset else "unknown"
            
            print(f"\n[{i}] Dataset: {result.dataset_name} (ID: {result.dataset_id})")
            print(f"    Type: {dataset_type}")
            print(f"    Distance: {result.total_distance:.2f} km")
            print(f"    Violations: {result.time_window_violations}")
            print(f"    Routes: {result.num_routes}")
            print(f"    Fitness: {result.fitness:.6f}")
            print(f"    Penalty: {result.penalty:.2f}")
            print(f"    Run ID: {result.run_id}")
            print(f"    Updated: {result.updated_at}")
            print(f"    Achieved: {result.achieved_at}")
            
            # Check if this looks like the result user mentioned
            if 600 <= result.total_distance <= 700 and 90 <= result.time_window_violations <= 100:
                print(f"    ⭐ THIS MATCHES USER'S RESULT (distance ~680km, violations ~94)")
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total best results: {len(best_results)}")
        
        # Group by dataset type
        type_counts = {}
        for result in best_results:
            dataset = get_dataset(db, result.dataset_id)
            dataset_type = dataset.type if dataset else "unknown"
            type_counts[dataset_type] = type_counts.get(dataset_type, 0) + 1
        
        print("\nBy dataset type:")
        for dtype, count in sorted(type_counts.items()):
            print(f"  {dtype}: {count}")
        
    except Exception as e:
        print(f"❌ Error checking best results: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    check_best_results()

