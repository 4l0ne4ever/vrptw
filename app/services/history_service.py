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
        
        Priority:
        1. Fewer violations (if both have violations)
        2. Lower distance (if violations are similar)
        3. Higher fitness (tie-breaker)
        
        Args:
            current_distance: Current solution distance
            current_violations: Current time window violations
            current_fitness: Current fitness value
            best_distance: Best known distance (optional)
            best_violations: Best known violations (optional)
            best_fitness: Best known fitness (optional)
            
        Returns:
            True if current result is better, False otherwise
        """
        # If no best result exists, current is better
        if best_distance is None:
            return True
        
        # Priority 1: Fewer violations
        if current_violations < best_violations:
            return True
        if current_violations > best_violations:
            return False
        
        # Priority 2: Lower distance (violations are equal)
        if current_distance < best_distance:
            return True
        if current_distance > best_distance:
            return False
        
        # Priority 3: Higher fitness (tie-breaker)
        return current_fitness > best_fitness
    
    def save_result(
        self,
        dataset_id: int,
        dataset_name: str,
        solution: any,
        statistics: Dict,
        config: Dict,
        dataset_type: str = "solomon"
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
            from app.config.database import SessionLocal
            db = SessionLocal()
            
            # Extract metrics
            total_distance = solution.total_distance
            num_routes = len(solution.routes) if solution.routes else 0
            fitness = solution.fitness
            penalty = getattr(solution, 'penalty', 0.0)
            
            # Calculate violations and compliance - use KPIs for accurate count
            time_window_violations = 0
            compliance_rate = 0.0
            
            # Try to get violations from statistics (should be actual count now)
            if 'time_window_violations' in statistics:
                time_window_violations = int(statistics['time_window_violations'])
            else:
                # Calculate from solution's constraint violations if available
                try:
                    # Try to get from solution's routes using constraint handler
                    if hasattr(solution, 'routes') and solution.routes:
                        # This is a fallback - ideally statistics should have violations
                        # For now, set to 0 and let it be calculated in next run
                        time_window_violations = 0
                except:
                    pass
            
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
                                import json
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
            
            # Prepare results JSON
            results_json = json.dumps({
                'total_distance': total_distance,
                'num_routes': num_routes,
                'time_window_violations': time_window_violations,
                'compliance_rate': compliance_rate,
                'fitness': fitness,
                'penalty': penalty,
                'gap_vs_bks': gap_vs_bks,
                'bks_distance': bks_distance,
                'statistics': statistics
            })
            
            # Prepare parameters JSON
            parameters_json = json.dumps(config)
            
            # Create optimization run
            run_name = f"{dataset_name} - {statistics.get('generations', 0)} generations"
            run = create_optimization_run(
                db,
                dataset_id=dataset_id,
                name=run_name,
                parameters_json=parameters_json,
                results_json=results_json,
                status="completed"
            )
            run_id = run.id
            
            # Check if this is better than existing best result
            best_result = get_best_result(db, dataset_id)
            is_new_best = False
            
            if best_result:
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
            
            # Update best result if better
            if is_new_best:
                create_or_update_best_result(
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
                logger.info(f"New best result saved for dataset {dataset_name}")
            
            db.close()
            return run_id, is_new_best
            
        except Exception as e:
            logger.error(f"Error saving result: {e}", exc_info=True)
            return None, False

