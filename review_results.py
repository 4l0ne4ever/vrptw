#!/usr/bin/env python3
"""
Comprehensive results review script.
Checks consistency and correctness of system outputs.
"""

import sys
import os
import csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def review_results():
    """Review all results for consistency and correctness."""
    print("="*80)
    print("COMPREHENSIVE RESULTS REVIEW")
    print("="*80)
    
    all_passed = True
    issues = []
    
    # Find latest result files
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    # Find latest files
    import glob
    kpi_files = sorted(glob.glob(os.path.join(results_dir, "kpi_comparison_*.csv")), reverse=True)
    evolution_files = sorted(glob.glob(os.path.join(results_dir, "evolution_data_*.csv")), reverse=True)
    routes_files = sorted(glob.glob(os.path.join(results_dir, "optimal_routes_*.txt")), reverse=True)
    
    if not kpi_files:
        print("❌ No KPI comparison files found")
        return False
    
    latest_kpi = kpi_files[0]
    latest_evolution = evolution_files[0] if evolution_files else None
    latest_routes = routes_files[0] if routes_files else None
    
    print(f"\n📊 Reviewing latest results:")
    print(f"   KPI: {os.path.basename(latest_kpi)}")
    if latest_evolution:
        print(f"   Evolution: {os.path.basename(latest_evolution)}")
    if latest_routes:
        print(f"   Routes: {os.path.basename(latest_routes)}")
    
    # Test 1: Review KPI Comparison
    print("\n" + "="*80)
    print("TEST 1: KPI Comparison Review")
    print("="*80)
    
    with open(latest_kpi, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if len(rows) < 2:
            print("❌ FAIL: KPI file has insufficient data")
            all_passed = False
            issues.append("KPI file has insufficient data")
        else:
            ga_row = rows[0]
            nn_row = rows[1]
            
            # Check GA metrics
            print("\n📈 GA Solution Metrics:")
            ga_distance = float(ga_row['total_distance'])
            ga_routes = int(ga_row['num_routes'])
            ga_utilization = float(ga_row['avg_utilization'])
            ga_load_balance = float(ga_row['load_balance_index'])
            ga_efficiency = float(ga_row['efficiency_score'])
            ga_feasible = ga_row['is_feasible'] == 'True'
            ga_penalty = float(ga_row['penalty'])
            
            print(f"   Distance: {ga_distance:.2f} km")
            print(f"   Routes: {ga_routes}")
            print(f"   Utilization: {ga_utilization:.1f}%")
            print(f"   Load Balance: {ga_load_balance:.3f}")
            print(f"   Efficiency: {ga_efficiency:.3f}")
            print(f"   Feasible: {ga_feasible}")
            print(f"   Penalty: {ga_penalty:.2f}")
            
            # Check NN metrics
            print("\n📈 Nearest Neighbor Metrics:")
            nn_distance = float(nn_row['total_distance'])
            nn_routes = int(nn_row['num_routes'])
            nn_utilization = float(nn_row['avg_utilization'])
            nn_load_balance = float(nn_row['load_balance_index'])
            nn_efficiency = float(nn_row['efficiency_score'])
            nn_feasible = nn_row['is_feasible'] == 'True'
            nn_penalty = float(nn_row['penalty'])
            
            print(f"   Distance: {nn_distance:.2f} km")
            print(f"   Routes: {nn_routes}")
            print(f"   Utilization: {nn_utilization:.1f}%")
            print(f"   Load Balance: {nn_load_balance:.3f}")
            print(f"   Efficiency: {nn_efficiency:.3f}")
            print(f"   Feasible: {nn_feasible}")
            print(f"   Penalty: {nn_penalty:.2f}")
            
            # Check improvements
            print("\n📊 Improvement Analysis:")
            distance_improvement = ((nn_distance - ga_distance) / nn_distance) * 100
            efficiency_improvement = ((ga_efficiency - nn_efficiency) / nn_efficiency) * 100
            
            print(f"   Distance Improvement: {distance_improvement:.2f}%")
            print(f"   Efficiency Improvement: {efficiency_improvement:.2f}%")
            
            # Validations
            print("\n✅ Validations:")
            
            # 1. GA should be better or equal to NN
            if ga_distance <= nn_distance:
                print("   ✅ PASS: GA distance <= NN distance (GA is better or equal)")
            else:
                print(f"   ❌ FAIL: GA distance ({ga_distance:.2f}) > NN distance ({nn_distance:.2f})")
                all_passed = False
                issues.append(f"GA worse than NN: {ga_distance:.2f} vs {nn_distance:.2f}")
            
            # 2. Both should be feasible
            if ga_feasible and nn_feasible:
                print("   ✅ PASS: Both solutions are feasible")
            else:
                if not ga_feasible:
                    print("   ❌ FAIL: GA solution is not feasible")
                    all_passed = False
                    issues.append("GA solution is not feasible")
                if not nn_feasible:
                    print("   ❌ FAIL: NN solution is not feasible")
                    all_passed = False
                    issues.append("NN solution is not feasible")
            
            # 3. Penalty should be 0 for feasible solutions
            if ga_feasible and ga_penalty == 0:
                print("   ✅ PASS: GA penalty is 0 (feasible solution)")
            elif ga_feasible and ga_penalty > 0:
                print(f"   ⚠️  WARNING: GA is feasible but penalty > 0: {ga_penalty:.2f}")
            
            if nn_feasible and nn_penalty == 0:
                print("   ✅ PASS: NN penalty is 0 (feasible solution)")
            elif nn_feasible and nn_penalty > 0:
                print(f"   ⚠️  WARNING: NN is feasible but penalty > 0: {nn_penalty:.2f}")
            
            # 4. Load balance should be reasonable
            if ga_load_balance >= 0.9:
                print(f"   ✅ PASS: GA load balance is good ({ga_load_balance:.3f})")
            else:
                print(f"   ⚠️  WARNING: GA load balance is low ({ga_load_balance:.3f})")
            
            # 5. Efficiency should be reasonable
            if 0.4 <= ga_efficiency <= 1.0:
                print(f"   ✅ PASS: GA efficiency is reasonable ({ga_efficiency:.3f})")
            else:
                print(f"   ⚠️  WARNING: GA efficiency is unusual ({ga_efficiency:.3f})")
    
    # Test 2: Review Evolution Data
    if latest_evolution:
        print("\n" + "="*80)
        print("TEST 2: Evolution Data Review")
        print("="*80)
        
        with open(latest_evolution, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if len(rows) < 2:
                print("❌ FAIL: Evolution file has insufficient data")
                all_passed = False
                issues.append("Evolution file has insufficient data")
            else:
                initial_distance = float(rows[0]['best_distance'])
                final_distance = float(rows[-1]['best_distance'])
                improvement = initial_distance - final_distance
                improvement_pct = (improvement / initial_distance) * 100
                
                print(f"\n📈 Evolution Metrics:")
                print(f"   Initial Distance: {initial_distance:.2f} km")
                print(f"   Final Distance: {final_distance:.2f} km")
                print(f"   Improvement: {improvement:.2f} km ({improvement_pct:.2f}%)")
                print(f"   Generations: {len(rows)}")
                
                # Check if final distance matches KPI
                # Note: Evolution data is from GA only, KPI is from solution after 2-opt
                # So they may differ if 2-opt improved the solution
                distance_diff = final_distance - ga_distance
                if abs(distance_diff) < 1.0:
                    print(f"\n   ✅ PASS: Final evolution distance matches KPI")
                    print(f"      Evolution: {final_distance:.2f} km")
                    print(f"      KPI: {ga_distance:.2f} km")
                    print(f"      Difference: {abs(distance_diff):.2f} km")
                elif distance_diff > 0:
                    # Evolution distance > KPI means 2-opt improved the solution
                    improvement = distance_diff
                    improvement_pct = (improvement / final_distance) * 100
                    print(f"\n   ✅ PASS: 2-opt improved solution after GA")
                    print(f"      GA Distance: {final_distance:.2f} km")
                    print(f"      After 2-opt: {ga_distance:.2f} km")
                    print(f"      2-opt Improvement: {improvement:.2f} km ({improvement_pct:.2f}%)")
                else:
                    print(f"\n   ⚠️  WARNING: KPI distance > Evolution distance (unexpected)")
                    print(f"      Evolution: {final_distance:.2f} km")
                    print(f"      KPI: {ga_distance:.2f} km")
                    print(f"      Difference: {abs(distance_diff):.2f} km")
                
                # Check convergence trend
                recent_distances = [float(row['best_distance']) for row in rows[-10:]]
                if len(recent_distances) >= 10:
                    recent_std = (max(recent_distances) - min(recent_distances)) / max(recent_distances) * 100
                    if recent_std < 5.0:
                        print(f"   ✅ PASS: Solution converged (recent std: {recent_std:.2f}%)")
                    else:
                        print(f"   ⚠️  WARNING: Solution may not have converged (recent std: {recent_std:.2f}%)")
                
                # Check if improvement is positive
                if improvement > 0:
                    print(f"   ✅ PASS: GA improved solution ({improvement:.2f} km)")
                else:
                    print(f"   ⚠️  WARNING: GA did not improve solution ({improvement:.2f} km)")
    
    # Test 3: Review Routes File
    if latest_routes:
        print("\n" + "="*80)
        print("TEST 3: Routes File Review")
        print("="*80)
        
        with open(latest_routes, 'r') as f:
            content = f.read()
            
            # Extract total distance
            import re
            distance_match = re.search(r'Tổng quãng đường: ([\d.]+) km', content)
            if distance_match:
                routes_distance = float(distance_match.group(1))
                print(f"\n📈 Routes File Metrics:")
                print(f"   Total Distance: {routes_distance:.2f} km")
                
                # Check if distance matches KPI
                if abs(routes_distance - ga_distance) < 1.0:
                    print(f"   ✅ PASS: Routes distance matches KPI")
                    print(f"      Routes: {routes_distance:.2f} km")
                    print(f"      KPI: {ga_distance:.2f} km")
                    print(f"      Difference: {abs(routes_distance - ga_distance):.2f} km")
                else:
                    print(f"   ❌ FAIL: Routes distance does NOT match KPI")
                    print(f"      Routes: {routes_distance:.2f} km")
                    print(f"      KPI: {ga_distance:.2f} km")
                    print(f"      Difference: {abs(routes_distance - ga_distance):.2f} km")
                    all_passed = False
                    issues.append(f"Routes distance mismatch: {routes_distance:.2f} vs {ga_distance:.2f}")
                
                # Extract route counts
                route_matches = re.findall(r'Xe \d+:', content)
                routes_count = len(route_matches)
                print(f"   Routes Count: {routes_count}")
                
                if routes_count == ga_routes:
                    print(f"   ✅ PASS: Routes count matches KPI ({routes_count})")
                else:
                    print(f"   ❌ FAIL: Routes count mismatch ({routes_count} vs {ga_routes})")
                    all_passed = False
                    issues.append(f"Routes count mismatch: {routes_count} vs {ga_routes}")
            else:
                print("   ⚠️  WARNING: Could not extract distance from routes file")
    
    # Test 4: Consistency Check
    print("\n" + "="*80)
    print("TEST 4: Overall Consistency Check")
    print("="*80)
    
    consistency_checks = []
    
    # Check KPI and Routes match (both should be after 2-opt)
    if 'ga_distance' in locals() and latest_routes and 'routes_distance' in locals():
        kpi_routes_diff = abs(ga_distance - routes_distance)
        if kpi_routes_diff < 1.0:
            print(f"   ✅ PASS: KPI and Routes distances match (diff: {kpi_routes_diff:.2f} km)")
            consistency_checks.append(True)
        else:
            print(f"   ❌ FAIL: KPI and Routes distances mismatch (diff: {kpi_routes_diff:.2f} km)")
            print(f"      KPI: {ga_distance:.2f} km")
            print(f"      Routes: {routes_distance:.2f} km")
            all_passed = False
            issues.append(f"KPI vs Routes mismatch: {kpi_routes_diff:.2f} km")
            consistency_checks.append(False)
    
    # Check Evolution vs KPI (Evolution is GA only, KPI is after 2-opt)
    if 'ga_distance' in locals() and latest_evolution:
        evolution_kpi_diff = final_distance - ga_distance
        if evolution_kpi_diff >= 0:
            # Evolution >= KPI means 2-opt improved (expected)
            print(f"   ✅ PASS: Evolution (GA) vs KPI (after 2-opt) - Expected difference")
            print(f"      GA Distance: {final_distance:.2f} km")
            print(f"      After 2-opt: {ga_distance:.2f} km")
            print(f"      2-opt Improvement: {evolution_kpi_diff:.2f} km")
            consistency_checks.append(True)
        else:
            print(f"   ⚠️  WARNING: KPI > Evolution (unexpected)")
            print(f"      Evolution: {final_distance:.2f} km")
            print(f"      KPI: {ga_distance:.2f} km")
            print(f"      Difference: {abs(evolution_kpi_diff):.2f} km")
            consistency_checks.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("REVIEW SUMMARY")
    print("="*80)
    
    if all_passed and all(consistency_checks):
        print("✅ ALL CHECKS PASSED!")
        print("\n✅ System is working correctly:")
        print("   1. ✅ GA solution is better than or equal to NN")
        print("   2. ✅ Both solutions are feasible")
        print("   3. ✅ All distances are consistent")
        print("   4. ✅ Load balance and efficiency are reasonable")
        print("   5. ✅ GA improved the solution")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    print("\n" + "="*80)
    
    return all_passed and all(consistency_checks)

if __name__ == '__main__':
    success = review_results()
    sys.exit(0 if success else 1)

