"""
Service for managing optimization history and best results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.database_service import DatabaseService
from app.database.crud import (
    get_best_result,
    create_or_update_best_result,
    create_optimization_run,
    get_dataset
)
from app.core.logger import setup_app_logger

logger = setup_app_logger()


class HistoryService:
    """Service for managing optimization history."""
    
    def __init__(self):
        """Initialize history service."""
        self.db_service = DatabaseService()
    
    def is_better_result(
        self,
        current_distance: float,
        current_violations: int,
        current_fitness: float,
        best_distance: Optional[float] = None,
        best_violations: Optional[int] = None,
        best_fitness: Optional[float] = None
    ) -> bool:
        """
        Determine if current result is better than best result.
        
        Uses fitness as primary comparison since it already balances:
        - Distance (lower is better)
        - Penalties/violations (lower is better)
        - Both are combined in fitness calculation
        
        For edge cases:
        - If both have 0 violations: compare by distance
        - If fitness is very close: use violations as tie-breaker
        - Otherwise: fitness is the best indicator
        
        Args:
            current_distance: Current solution distance
            current_violations: Current time window violations
            current_fitness: Current fitness value (already includes distance + penalties)
            best_distance: Best known distance (optional)
            best_violations: Best known violations (optional)
            best_fitness: Best known fitness (optional)
            
        Returns:
            True if current result is better, False otherwise
        """
        # If no best result exists, current is better
        if best_distance is None or best_fitness is None:
            return True
        
        # Primary comparison: Use fitness (already balances distance and violations)
        # Fitness = -(penalty + distance), so higher fitness = better
        fitness_diff = current_fitness - best_fitness
        
        # If fitness is significantly better (> 1% improvement), use it
        if abs(best_fitness) > 0:
            fitness_improvement_ratio = fitness_diff / abs(best_fitness)
            if fitness_improvement_ratio > 0.01:  # > 1% improvement
                return True
            if fitness_improvement_ratio < -0.01:  # > 1% worse
                return False
        
        # If fitness is very close (within 1%), use detailed comparison
        # Case 1: Both have 0 violations - compare by distance
        if current_violations == 0 and best_violations == 0:
            return current_distance < best_distance
        
        # Case 2: One has violations, one doesn't - prefer feasible (0 violations)
        # Feasible solutions are generally better than infeasible ones
        if current_violations == 0 and best_violations > 0:
            # Current is feasible, best is not - prefer feasible
            # Only prefer infeasible if distance is MUCH better (>70% reduction)
            if best_distance > 0 and current_distance > 0:
                distance_reduction = (best_distance - current_distance) / best_distance
                if distance_reduction > 0.7:  # More than 70% distance reduction
                    return False  # Huge distance improvement might be worth violations
                else:
                    return True  # Prefer feasible solution (0 violations is better)
            else:
                return True  # Prefer feasible solution
        
        if current_violations > 0 and best_violations == 0:
            # Current has violations, best is feasible - prefer feasible
            # Only prefer infeasible if distance is MUCH better (>70% reduction)
            if current_distance > 0 and best_distance > 0:
                distance_reduction = (best_distance - current_distance) / best_distance
                if distance_reduction > 0.7:  # More than 70% distance reduction
                    return True  # Huge distance improvement might be worth violations
                else:
                    return False  # Prefer feasible solution (0 violations is better)
            else:
                return False  # Prefer feasible solution
        
        # Case 3: Both have violations - compare by violations first, then distance
        if current_violations < best_violations:
            return True
        if current_violations > best_violations:
            # More violations, but check if distance improvement is huge
            if current_distance > 0 and best_distance > 0:
                distance_improvement = (best_distance - current_distance) / best_distance
                # If distance improvement is > 60%, might be worth more violations
                if distance_improvement > 0.6:
                    return True
            return False
        
        # Violations are equal - compare by distance
        return current_distance < best_distance
    
    def save_result(
        self,
        dataset_id: int,
        dataset_name: str,
        solution: any,
        statistics: Dict,
        config: Dict,
        dataset_type: str = "solomon",
        problem: Optional[any] = None  # VRPProblem object from runtime (preferred over DB rebuild)
    ) -> Tuple[Optional[int], bool]:
        """
        Save optimization result and update best result if better.
        
        Args:
            dataset_id: Dataset ID
            dataset_name: Dataset name
            solution: Best solution (Individual)
            statistics: GA statistics
            config: GA configuration
            dataset_type: Dataset type (solomon or hanoi)
            
        Returns:
            Tuple of (run_id, is_new_best)
        """
        try:
            logger.info(f"🔵 save_result called: dataset_name={dataset_name}, dataset_id={dataset_id}, "
                      f"distance={solution.total_distance:.2f}, dataset_type={dataset_type}")
            print(f"🔵 [HISTORY] Starting save_result: dataset={dataset_name}, id={dataset_id}")
            
            from app.config.database import SessionLocal
            db = SessionLocal()
            logger.info(f"✅ Database connection established")
            
            # Extract metrics
            total_distance = solution.total_distance
            num_routes = len(solution.routes) if solution.routes else 0
            fitness = solution.fitness
            penalty = getattr(solution, 'penalty', 0.0)
            
            # Calculate violations and compliance - calculate accurately from solution routes
            time_window_violations = 0
            compliance_rate = 0.0
            
            # Get violations from statistics (calculated by optimization_service using KPICalculator)
            # CRITICAL: Do NOT estimate from penalty - the new tiered penalty system
            # makes penalty-to-violation ratio non-constant and estimation unreliable
            logger.info(f"🔵 Checking statistics for violations: keys={list(statistics.keys())}")
            
            if 'time_window_violations' in statistics:
                raw_value = statistics['time_window_violations']
                time_window_violations = int(raw_value)
                logger.info(f"✅ Found time_window_violations in statistics: {raw_value} -> {time_window_violations}")
                
                # SANITY CHECK: Violations should be reasonable (not penalty value)
                # Penalty values are typically millions/billions, violations should be < 10000
                if time_window_violations > 10000:
                    logger.error(f"❌ SANITY CHECK FAILED: violations={time_window_violations} looks like penalty, not count!")
                    logger.error(f"   This is likely a bug - violations should be < 10000 for typical problems")
                    # Try to get from constraint_violations instead
                    if 'constraint_violations' in statistics:
                        violations_data = statistics.get('constraint_violations', {})
                        if isinstance(violations_data, dict):
                            alt_violations = violations_data.get('time_window_violations', 0)
                            if alt_violations < 10000:
                                logger.warning(f"   Using alternative value from constraint_violations: {alt_violations}")
                                time_window_violations = int(alt_violations)
                            else:
                                logger.error(f"   Alternative value also looks wrong: {alt_violations}")
                                time_window_violations = 0
                    else:
                        time_window_violations = 0
            elif 'constraint_violations' in statistics:
                # Try alternative key name
                violations_data = statistics.get('constraint_violations', {})
                if isinstance(violations_data, dict):
                    raw_value = violations_data.get('time_window_violations', 0)
                    time_window_violations = int(raw_value)
                    logger.info(f"✅ Found time_window_violations in constraint_violations: {raw_value} -> {time_window_violations}")
                else:
                    time_window_violations = 0
                    logger.warning("⚠️  constraint_violations exists but is not a dict")
            else:
                # WARNING: Violations not found in statistics!
                # This should NOT happen if optimization_service calculated KPIs correctly
                logger.warning("❌ Time window violations not found in statistics - defaulting to 0. "
                             "This may indicate optimization_service didn't calculate KPIs properly.")
                logger.warning(f"   Available statistics keys: {list(statistics.keys())}")
                time_window_violations = 0

                # Mark in statistics that violations are missing (for debugging)
                statistics['_violations_missing'] = True
            
            logger.info(f"🔵 Final violations count: {time_window_violations} (penalty={penalty:.2f})")
            
            # Ensure statistics has time_window_violations for future use
            if 'time_window_violations' not in statistics:
                statistics['time_window_violations'] = time_window_violations
            
            # Calculate compliance rate
            total_customers = len(solution.chromosome) if hasattr(solution, 'chromosome') else 0
            if total_customers > 0:
                compliance_rate = max(0.0, min(100.0, ((total_customers - time_window_violations) / total_customers) * 100))
            else:
                compliance_rate = 100.0 if time_window_violations == 0 else 0.0
            
            # Get BKS info for Solomon datasets
            gap_vs_bks = None
            bks_distance = None
            if dataset_type == "solomon":
                # Try to get from statistics first
                if 'bks_distance' in statistics:
                    bks_distance = statistics['bks_distance']
                else:
                    # Fallback: try to calculate from dataset name
                    try:
                        from src.evaluation.bks_validator import BKSValidator
                        # Try to get dataset name from dataset_id
                        from app.database.crud import get_dataset
                        db_temp = SessionLocal()
                        try:
                            dataset = get_dataset(db_temp, dataset_id)
                            if dataset and dataset.metadata_json:
                                metadata = json.loads(dataset.metadata_json)
                                dataset_name = metadata.get('name', '')
                                if dataset_name:
                                    bks_validator = BKSValidator()
                                    bks_data = bks_validator.get_bks(dataset_name)
                                    if bks_data:
                                        bks_distance = bks_data.get('distance')
                        finally:
                            db_temp.close()
                    except Exception as e:
                        logger.warning(f"Could not load BKS data: {str(e)}")
                
                if bks_distance and bks_distance > 0:
                    gap_vs_bks = ((total_distance - bks_distance) / bks_distance) * 100
            
            # Get execution_time from statistics for runtime display
            execution_time = statistics.get('execution_time', 0.0)
            
            # Calculate comprehensive KPIs using KPICalculator for complete metrics
            # This ensures all metrics are accurate and complete in results_json
            kpis = {}
            try:
                from src.evaluation.metrics import KPICalculator
                
                # PREFERRED: Use problem object passed from runtime (avoids data sync issues)
                # FALLBACK: Reconstruct problem from dataset in DB (may be outdated)
                problem_for_kpis = problem
                
                if problem_for_kpis is None:
                    logger.info("🔵 No problem object provided, reconstructing from database...")
                    try:
                        from app.database.crud import get_dataset
                        dataset = get_dataset(db, dataset_id)
                        if dataset and dataset.data_json:
                            dataset_data = json.loads(dataset.data_json)
                            from app.services.data_service import DataService
                            data_service = DataService()
                            problem_for_kpis = data_service.create_vrp_problem(dataset_data, calculate_distance=True, dataset_type=dataset_type)
                            
                            if problem_for_kpis:
                                logger.info("✅ Problem reconstructed from database")
                            else:
                                logger.warning("⚠️  Problem reconstruction returned None, using basic metrics only")
                        else:
                            logger.warning("⚠️  Dataset data not available, using basic metrics only")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not reconstruct problem from dataset: {e}, using basic metrics only")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.info("✅ Using problem object from runtime (avoids data sync issues)")
                
                if problem_for_kpis:
                    kpi_calculator = KPICalculator(problem_for_kpis)
                    kpis = kpi_calculator.calculate_kpis(solution, execution_time=execution_time)
                    logger.info(f"✅ Calculated comprehensive KPIs: {len(kpis)} metrics")
            except Exception as e:
                logger.warning(f"⚠️  Failed to calculate KPIs: {e}, using basic metrics only")
                import traceback
                traceback.print_exc()
            
            # Prepare results JSON - include ALL metrics for complete record
            # Basic metrics (always included)
            results_data = {
                'total_distance': total_distance,
                'num_routes': num_routes,
                'time_window_violations': time_window_violations,
                'compliance_rate': compliance_rate,
                'fitness': fitness,
                'penalty': penalty,
                'gap_vs_bks': gap_vs_bks,
                'bks_distance': bks_distance,
                'execution_time': execution_time,
            }
            
            # Add comprehensive KPIs if available
            if kpis:
                # Add all KPI metrics
                results_data.update({
                    'num_customers': kpis.get('num_customers', len(solution.chromosome) if hasattr(solution, 'chromosome') else 0),
                    'num_vehicles_used': kpis.get('num_vehicles_used', num_routes),
                    'avg_route_length': kpis.get('avg_route_length', 0.0),
                    'max_route_length': kpis.get('max_route_length', 0.0),
                    'min_route_length': kpis.get('min_route_length', 0.0),
                    'route_length_std': kpis.get('route_length_std', 0.0),
                    'avg_utilization': kpis.get('avg_utilization', 0.0),
                    'max_utilization': kpis.get('max_utilization', 0.0),
                    'min_utilization': kpis.get('min_utilization', 0.0),
                    'utilization_std': kpis.get('utilization_std', 0.0),
                    'load_balance_index': kpis.get('load_balance_index', 0.0),
                    'total_cost': kpis.get('total_cost', 0.0),
                    'cost_per_km': kpis.get('cost_per_km', 0.0),
                    'cost_per_customer': kpis.get('cost_per_customer', 0.0),
                    'cost_per_route': kpis.get('cost_per_route', 0.0),
                    'solution_quality': kpis.get('solution_quality', 0.0),
                    'efficiency_score': kpis.get('efficiency_score', 0.0),
                    'feasibility_score': kpis.get('feasibility_score', 0.0),
                    'is_feasible': kpis.get('is_feasible', time_window_violations == 0),
                })
                
                # Add constraint violations details
                if 'constraint_violations' in kpis:
                    results_data['constraint_violations'] = kpis['constraint_violations']
                
                # Add shipping cost if available
                if 'shipping_cost' in kpis:
                    results_data['shipping_cost'] = kpis['shipping_cost']
                if 'total_cost_with_operational' in kpis:
                    results_data['total_cost_with_operational'] = kpis['total_cost_with_operational']
                if 'cost_breakdown' in kpis:
                    results_data['cost_breakdown'] = kpis['cost_breakdown']
            
            # Add statistics for backward compatibility and detailed analysis
            results_data['statistics'] = statistics

            # Convert numpy/pandas types to Python native types for JSON serialization
            def convert_to_native(obj):
                """Recursively convert numpy/pandas types to Python native types."""
                try:
                    import numpy as np
                except ImportError:
                    np = None

                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_native(item) for item in obj]
                elif np and isinstance(obj, np.integer):
                    return int(obj)
                elif np and isinstance(obj, np.floating):
                    return float(obj)
                elif np and isinstance(obj, np.bool_):
                    return bool(obj)
                elif np and isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    # Fallback: try to convert to string or return as-is
                    try:
                        return str(obj)
                    except:
                        return obj

            results_data = convert_to_native(results_data)
            results_json = json.dumps(results_data)
            
            # Prepare parameters JSON
            parameters_json = json.dumps(config)
            
            # CRITICAL: Create optimization run FIRST and commit immediately
            # This ensures run is ALWAYS saved, even if best result update fails
            # User requirement: "lưu toàn bộ kết quả luôn chứ không cần kết quả tốt nhất nữa"
            run_name = f"{dataset_name} - {statistics.get('generations', 0)} generations"
            logger.info(f"🔵 Creating optimization run: name={run_name}, dataset_id={dataset_id}")
            print(f"🔵 [HISTORY] Creating run: {run_name}")
            
            try:
                run = create_optimization_run(
                    db,
                    dataset_id=dataset_id,
                    name=run_name,
                    parameters_json=parameters_json,
                    results_json=results_json,
                    status="completed"
                )
                run_id = run.id
                
                # Commit run immediately to ensure it's saved (CRITICAL)
                db.commit()
                logger.info(f"✅ Optimization run created and committed: run_id={run_id}, dataset={dataset_name}")
                print(f"✅ [HISTORY] Run saved successfully: run_id={run_id}, distance={total_distance:.2f}km, violations={time_window_violations}")
                
                # Verify run was saved
                db_verify = SessionLocal()
                try:
                    from app.database.crud import get_optimization_run
                    verified_run = get_optimization_run(db_verify, run_id)
                    if verified_run:
                        logger.info(f"✅ Run verification successful: run_id={run_id} exists in database")
                        print(f"✅ [HISTORY] Run verified in database: run_id={run_id}")
                    else:
                        logger.error(f"❌ Run verification failed: run_id={run_id} not found in database!")
                        print(f"❌ [HISTORY] Run NOT found in database after save!")
                finally:
                    db_verify.close()
                    
            except Exception as run_error:
                logger.error(f"❌ CRITICAL: Failed to create optimization run: {run_error}", exc_info=True)
                print(f"❌ [HISTORY] Failed to create run: {run_error}")
                import traceback
                traceback.print_exc()
                db.rollback()
                db.close()
                return None, False
            
            # OPTIONAL: Update best result (separate from run saving)
            # User requirement: "không cần kết quả tốt nhất nữa" - but we keep it for backward compatibility
            # Best result update is now optional and won't prevent run from being saved
            is_new_best = False
            try:
                best_result = get_best_result(db, dataset_id)
                
                if best_result:
                    # Log old values for comparison
                    logger.info(f"Found existing best result for dataset {dataset_name} (ID: {dataset_id}): "
                              f"distance={best_result.total_distance:.2f}, violations={best_result.time_window_violations}, "
                              f"fitness={best_result.fitness:.2f}, run_id={best_result.run_id}")
                    
                    # Check if new result is better (for logging/info purposes only)
                    is_new_best = self.is_better_result(
                        current_distance=total_distance,
                        current_violations=time_window_violations,
                        current_fitness=fitness,
                        best_distance=best_result.total_distance,
                        best_violations=best_result.time_window_violations,
                        best_fitness=best_result.fitness
                    )
                else:
                    is_new_best = True
                    logger.info(f"No existing best result for dataset {dataset_name} (ID: {dataset_id}), creating new one")
                
                # Update best result (optional - won't fail if this errors)
                logger.info(f"🔵 Updating best result for dataset {dataset_name} (ID: {dataset_id}): "
                          f"distance={total_distance:.2f}, violations={time_window_violations}, "
                          f"fitness={fitness:.2f}, run_id={run_id}")
                
                updated_best_result = create_or_update_best_result(
                    db,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    run_id=run_id,
                    total_distance=total_distance,
                    num_routes=num_routes,
                    time_window_violations=time_window_violations,
                    compliance_rate=compliance_rate,
                    fitness=fitness,
                    penalty=penalty,
                    parameters_json=parameters_json,
                    gap_vs_bks=gap_vs_bks,
                    bks_distance=bks_distance
                )
                
                # Verify update was successful
                db.commit()
                verified_result = get_best_result(db, dataset_id)
                
                if verified_result and verified_result.run_id == run_id:
                    logger.info(f"✅ Best result updated: distance={verified_result.total_distance:.2f}, "
                              f"violations={verified_result.time_window_violations}, run_id={verified_result.run_id}")
                else:
                    logger.warning(f"⚠️  Best result update verification failed (but run is saved): "
                                 f"expected run_id={run_id}, got={verified_result.run_id if verified_result else None}")
            except Exception as best_result_error:
                # Best result update failed, but run is already saved - just log warning
                logger.warning(f"⚠️  Best result update failed (but run {run_id} is saved): {best_result_error}")
                # Don't rollback - run is already committed
                # Don't re-raise - run saving is more important
            
            db.close()
            
            logger.info(f"✅ save_result completed: run_id={run_id}, is_new_best={is_new_best}")
            print(f"✅ Result saved successfully: run_id={run_id}, distance={total_distance:.2f}km, violations={time_window_violations}")
            
            return run_id, is_new_best
            
        except Exception as e:
            logger.error(f"Error saving result: {e}", exc_info=True)
            return None, False

