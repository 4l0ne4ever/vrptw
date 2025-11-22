#!/usr/bin/env python3
"""
Script to check recent optimization runs in database.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.database import SessionLocal
from app.database.crud import get_optimization_runs
import json

def check_recent_runs():
    """Check recent optimization runs."""
    db = SessionLocal()
    try:
        runs = get_optimization_runs(db, limit=20)
        
        print("=" * 80)
        print(f"RECENT OPTIMIZATION RUNS (Last {len(runs)})")
        print("=" * 80)
        
        if not runs:
            print("❌ No runs found!")
            return
        
        for i, run in enumerate(runs, 1):
            try:
                results_data = json.loads(run.results_json) if run.results_json else {}
            except:
                results_data = {}
            
            distance = results_data.get('total_distance', 0)
            violations = results_data.get('time_window_violations', 0)
            
            print(f"\n[{i}] Run: {run.name} (ID: {run.id})")
            print(f"    Dataset ID: {run.dataset_id}")
            print(f"    Distance: {distance:.2f} km")
            print(f"    Violations: {violations}")
            print(f"    Status: {run.status}")
            print(f"    Created: {run.created_at}")
            
            # Check if this matches user's result
            if 600 <= distance <= 700 and 90 <= violations <= 100:
                print(f"    ⭐ THIS MATCHES USER'S RESULT (distance ~680km, violations ~94)")
            elif 600 <= distance <= 700:
                print(f"    ⚠️  Distance matches but violations don't: {violations}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    check_recent_runs()

