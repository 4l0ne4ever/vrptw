#!/usr/bin/env python3
"""
Script to fix best results that have penalty values instead of violation counts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.database import SessionLocal
from app.database.crud import get_all_best_results, get_optimization_run, get_dataset
from app.database.models import BestResult
import json

def fix_best_result_violations():
    """Fix best results that have penalty values instead of violation counts."""
    db = SessionLocal()
    try:
        best_results = get_all_best_results(db, limit=1000)
        
        print("=" * 80)
        print("FIXING BEST RESULTS WITH INCORRECT VIOLATIONS")
        print("=" * 80)
        
        fixed_count = 0
        
        for result in best_results:
            # Check if violations look like penalty (too large)
            if result.time_window_violations > 10000:
                print(f"\n🔧 Fixing: {result.dataset_name} (ID: {result.dataset_id})")
                print(f"   Current violations: {result.time_window_violations} (looks like penalty)")
                
                # Try to get correct violations from the run's results_json
                run = get_optimization_run(db, result.run_id)
                if run and run.results_json:
                    try:
                        results_data = json.loads(run.results_json)
                        correct_violations = results_data.get('time_window_violations', 0)
                        
                        # Sanity check: should be reasonable
                        if 0 <= correct_violations <= 10000:
                            print(f"   Found correct violations in run: {correct_violations}")
                            
                            # Update best result
                            result.time_window_violations = int(correct_violations)
                            
                            # Recalculate compliance rate
                            total_customers = len(results_data.get('statistics', {}).get('total_customers', 100))
                            if total_customers > 0:
                                result.compliance_rate = max(0.0, min(100.0, 
                                    ((total_customers - correct_violations) / total_customers) * 100))
                            
                            db.add(result)
                            db.commit()
                            
                            print(f"   ✅ Fixed: violations={correct_violations}, compliance={result.compliance_rate:.1f}%")
                            fixed_count += 1
                        else:
                            print(f"   ⚠️  Run violations also look wrong: {correct_violations}")
                    except Exception as e:
                        print(f"   ❌ Error parsing run results: {e}")
                else:
                    print(f"   ⚠️  No run data found (run_id={result.run_id})")
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: Fixed {fixed_count} best result(s)")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    fix_best_result_violations()

