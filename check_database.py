#!/usr/bin/env python3
"""
Script to check database state and verify history saving.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.database import SessionLocal
from app.database.crud import get_all_best_results, get_optimization_runs, get_datasets
from app.database.models import OptimizationRun, BestResult, Dataset

def check_database():
    """Check database state."""
    db = SessionLocal()
    try:
        print("=" * 80)
        print("DATABASE STATE CHECK")
        print("=" * 80)
        
        # Check datasets
        datasets = get_datasets(db, limit=1000)
        print(f"\n📊 Datasets: {len(datasets)}")
        for ds in datasets[:10]:  # Show first 10
            print(f"   - {ds.name} (ID: {ds.id}, type: {ds.type})")
        
        # Check optimization runs
        runs = get_optimization_runs(db, limit=1000)
        print(f"\n📊 Optimization Runs: {len(runs)}")
        for run in runs[:10]:  # Show first 10
            print(f"   - Run {run.id}: {run.name} (status: {run.status}, created: {run.created_at})")
            if run.results_json:
                try:
                    import json
                    results = json.loads(run.results_json)
                    distance = results.get('total_distance', 0)
                    violations = results.get('time_window_violations', 0)
                    print(f"     Distance: {distance:.2f}km, Violations: {violations}")
                except:
                    print(f"     (Could not parse results_json)")
        
        # Check best results
        best_results = get_all_best_results(db, limit=1000)
        print(f"\n📊 Best Results: {len(best_results)}")
        for br in best_results[:10]:  # Show first 10
            print(f"   - {br.dataset_name} (ID: {br.dataset_id}): "
                  f"distance={br.total_distance:.2f}km, violations={br.time_window_violations}, "
                  f"run_id={br.run_id}")
        
        # Direct query to verify
        print(f"\n📊 Direct Database Queries:")
        all_runs = db.query(OptimizationRun).all()
        print(f"   Total runs in database: {len(all_runs)}")
        
        all_best = db.query(BestResult).all()
        print(f"   Total best results in database: {len(all_best)}")
        
        all_datasets = db.query(Dataset).all()
        print(f"   Total datasets in database: {len(all_datasets)}")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    check_database()

