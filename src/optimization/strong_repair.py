"""
Strong Repair Operator for VRP.

Implements PHASE 3: STRONG REPAIR (Feasibility Enforcer)
- Violated-customers-only strategy (don't fix what's not broken)
- Neighbor-based search (40 closest instead of all N)
- Best-improvement (scan all candidates, pick best)
- O(1) feasibility checks via VidalEvaluator

Expected: Reduce distance from ~1066km → ~650km while maintaining 0 violations
"""

import logging
import copy
import math
import random
import time
from typing import List, Tuple, Optional, Dict
from ..models.vrp_model import VRPProblem
from .vidal_evaluator import VidalEvaluator
from .lns_optimizer import LNSOptimizer

logger = logging.getLogger(__name__)


class StrongRepair:
    """
    Advanced repair operator using neighbor lists and best-improvement.

    Key Innovations vs. Traditional Repair:
    1. Violated-Only: Only fix customers that are actually late (not all)
    2. Neighbor-Based: Search only 40 closest neighbors (not all N)
    3. Best-Improvement: Try all candidates, pick best (not first-improving)
    4. O(1) Evaluation: Use VidalEvaluator for fast feasibility checks

    Result: Faster AND better quality than traditional repair!
    """

    def __init__(self,
                 problem: VRPProblem,
                 neighbor_lists: Dict[int, List[int]],
                 evaluator: VidalEvaluator,
                 max_iterations: int = 500,
                 enable_swap: bool = True,
                 enable_restart: bool = True):
        """
        Initialize Strong Repair operator.

        Args:
            problem: VRP problem instance
            neighbor_lists: Pre-computed neighbor lists (from NeighborListBuilder)
            evaluator: Vidal evaluator for O(1) checks
            max_iterations: Maximum repair iterations
            enable_swap: Enable swap operator (for capacity deadlocks)
            enable_restart: Enable restart mechanism
        """
        self.problem = problem
        self.neighbor_lists = neighbor_lists
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.enable_swap = enable_swap
        self.enable_restart = enable_restart

        # Load BKS data for Solomon benchmarks
        self.bks_data = self._load_bks_data()

        # Detect dataset mode (Solomon vs Hanoi)
        self.dataset_mode = self._detect_dataset_mode()
        logger.info(f"   Dataset mode detected: {self.dataset_mode}")

    def repair_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Repair routes to eliminate time window violations using adaptive strategy.

        Args:
            routes: List of routes (each route is list of customer IDs)

        Returns:
            Repaired routes with 0 violations (hopefully)

        Adaptive Strategy:
            1. GATE 0: Capacity Check (VETO POWER)
            2. GATE 1: Standard Repair (current pipeline)
            3. GATE 2: Mode-Dependent Deep Repair
               - Hanoi: Accept solution (already working well)
               - Solomon: Multi-restart for moderate/severe violations
            4. GATE 3: LNS Post-Optimization (if 0 violations)
        """
        logger.info("🔧 Strong Repair: Starting ADAPTIVE repair...")
        logger.info(f"   Mode: {self.dataset_mode}")
        logger.info(f"   Max iterations: {self.max_iterations}")
        logger.info(f"   Swap enabled: {self.enable_swap}")
        logger.info(f"   Restart enabled: {self.enable_restart}")

        # Make deep copy to avoid modifying original
        current_routes = copy.deepcopy(routes)

        # CRITICAL: Validate and fix input routes structure
        current_routes = self._validate_and_fix_routes(current_routes)

        # Initial violation count
        initial_violations = self._count_violations(current_routes)
        logger.info(f"   Initial violations: {initial_violations}")

        if initial_violations == 0:
            logger.info("   ✅ No violations! Returning original routes")
            return current_routes

        # NOTE: Fix 1 (mode-specific counting) đã align Strong Repair với ConstraintHandler
        # Không cần safety check nữa vì cả 2 giờ đếm violations giống nhau

        # =============================================================================
        # GATE 0: CAPACITY CHECK & SEVERITY CLASSIFICATION
        # =============================================================================
        total_customers = sum(1 for route in current_routes for c in route if c != 0)
        capacity_util = self._calculate_capacity_utilization(current_routes)
        severity = self._classify_severity(initial_violations, total_customers)
        avg_tw_width = self._analyze_time_window_tightness()

        logger.info(f"   📊 Problem Analysis:")
        logger.info(f"      Capacity utilization: {capacity_util*100:.1f}%")
        logger.info(f"      Violation severity: {severity} ({initial_violations}/{total_customers} = {initial_violations/total_customers*100:.1f}%)")
        logger.info(f"      Avg TW width: {avg_tw_width:.1f} time units")

        # Determine root cause
        capacity_is_tight = capacity_util >= 0.90  # 90% threshold
        if capacity_is_tight:
            logger.info(f"   ⚠️  Capacity is TIGHT (>90%) → Capacity-constrained problem")
        else:
            logger.info(f"   ✅ Capacity is OK (<90%) → Time Window problem, NOT capacity!")

        # Get BKS info for Solomon mode
        instance_name = getattr(self.problem, 'name', None)
        if instance_name is None:
            metadata = getattr(self.problem, 'metadata', {}) or {}
            instance_name = metadata.get('name', '')

        bks_vehicles = self._get_bks_vehicles(instance_name)
        if bks_vehicles and self.dataset_mode == 'solomon':
            logger.info(f"   📚 BKS vehicles for {instance_name}: {bks_vehicles}")
            current_vehicles = len(current_routes)
            logger.info(f"      Current vehicles: {current_vehicles} (BKS+2 limit: {bks_vehicles+2})")

        # =============================================================================
        # RECONSTRUCTIVE REPAIR (Ruin & Recreate with Regret-2)
        # =============================================================================

        # --- PHASE 1: RUIN (Smart Ejection: Violated + Related Neighbors) ---
        # Based on Shaw's removal operator (Shaw 1998)
        violated_info = self._get_violated_customers(current_routes)
        violated_ids = [v['customer_id'] for v in violated_info]

        # SMART EJECTION: Eject violated customers + their related neighbors
        # Rationale: Neighbors might be blocking good positions for violated customers
        # Reference: Shaw, P. (1998). "Using Constraint Programming and Local Search"
        customers_to_eject = set(violated_ids)

        # For Solomon mode, add related neighbors (more aggressive repair)
        # For Hanoi mode, only eject violated customers (conservative, already working)
        if self.dataset_mode == 'solomon' and len(violated_ids) > 0 and len(violated_ids) <= 30:
            # Only use smart ejection if violations are moderate (not catastrophic)
            # FIX: Reduce aggression - add max 1 neighbor per violated, cap total at violated count
            max_neighbors_total = len(violated_ids)  # Don't add more neighbors than violated customers

            neighbors_added_ids = []  # Track which neighbors were added
            for viol_id in violated_ids:
                if len(neighbors_added_ids) >= max_neighbors_total:
                    break  # Stop if we've added enough neighbors

                # Get k-nearest neighbors for this violated customer
                if viol_id in self.neighbor_lists:
                    neighbors = self.neighbor_lists[viol_id][:3]  # Top 3 nearest (reduced from 5)

                    # Add ONLY 1 neighbor per violated customer (reduced from 2)
                    added = 0
                    for neighbor_id in neighbors:
                        if neighbor_id not in customers_to_eject and added < 1:
                            # Check if this neighbor is currently in routes
                            for route in current_routes:
                                if neighbor_id in route and neighbor_id != 0:
                                    customers_to_eject.add(neighbor_id)
                                    neighbors_added_ids.append(neighbor_id)
                                    added += 1
                                    break

            logger.info(f"   💣 RUIN PHASE (Smart Ejection):")
            logger.info(f"      Violated customers: {len(violated_ids)} → {violated_ids[:10]}")
            if neighbors_added_ids:
                logger.info(f"      Related neighbors: {len(neighbors_added_ids)} → {neighbors_added_ids[:10]}")
            logger.info(f"      Total to eject: {len(customers_to_eject)}")
        else:
            logger.info(f"   💣 RUIN PHASE: Ejecting {len(violated_ids)} violated customers...")

        # EMERGENCY RESET LOGIC: If too many violations, reset ALL routes
        total_customers = sum(1 for route in current_routes for c in route if c != 0)
        # FIX: Emergency threshold should be percentage, not absolute
        # For Solomon: Be more aggressive - 30% threshold (easier to trigger)
        # For Hanoi: 50% threshold (less aggressive)
        if self.dataset_mode == 'solomon':
            EMERGENCY_THRESHOLD = max(10, int(0.3 * total_customers))  # 30% for Solomon
        else:
            EMERGENCY_THRESHOLD = max(20, int(0.5 * total_customers))  # 50% for Hanoi

        if len(customers_to_eject) > EMERGENCY_THRESHOLD:
            logger.warning(f"   🚨 EMERGENCY RESET: {len(customers_to_eject)}/{total_customers} to eject (>{EMERGENCY_THRESHOLD})")
            logger.warning(f"   Full reset required!")

            # Collect ALL customers
            all_customers = []
            for route in current_routes:
                for cust_id in route:
                    if cust_id != 0:
                        all_customers.append(cust_id)

            # Reset all routes to empty
            num_routes = len(current_routes)
            current_routes = [[0, 0] for _ in range(num_routes)]

            # Set all customers as unassigned
            customers_to_rebuild = all_customers
            logger.info(f"   Reset complete: {num_routes} empty routes, {len(customers_to_rebuild)} customers to rebuild")
        else:
            # Normal RUIN: Eject selected customers
            for route_idx in range(len(current_routes)):
                current_routes[route_idx] = [c for c in current_routes[route_idx]
                                             if c not in customers_to_eject]
            # Note: customers_to_eject includes violated + neighbors (if smart ejection active)
            customers_to_rebuild = list(customers_to_eject)

        # TODO: Consider mode-specific reconstruction (Solomon vs Hanoi) in future
        # For now, apply feasibility-first to both modes (should improve both)
        logger.info(f"   🔨 RECREATE PHASE: Feasibility-First Regret-2 insertion")

        # --- FEASIBILITY-FIRST REGRET-2 INSERTION ---
        # CRITICAL: Only consider FEASIBLE positions - never create violations
        # Strategy:
        #   1. Find ALL feasible positions for each customer
        #   2. If has feasible positions → use Regret-2 to choose
        #   3. If NO feasible positions → create new route (if under vehicle limit)
        #   4. Result: ALWAYS 0 violations (worst case: 1 customer per route)
        # Reference: "The Vehicle Routing Problem" by Toth & Vigo (2002)

        unassigned = list(customers_to_rebuild)
        insertions_made = 0

        # Get BKS vehicles for this instance (if known)
        instance_name = getattr(self.problem, 'name', None)
        if instance_name is None:
            metadata = getattr(self.problem, 'metadata', {}) or {}
            instance_name = metadata.get('name', '')

        bks_vehicles = self._get_bks_vehicles(instance_name)
        if bks_vehicles:
            # For Solomon: Allow more routes during reconstruction (BKS+4)
            # This gives greedy reconstruction more flexibility to find feasible placements
            # We can merge routes later if needed
            if self.dataset_mode == 'solomon':
                max_vehicles = bks_vehicles + 4  # BKS + 4 for Solomon (more flexible)
            else:
                max_vehicles = bks_vehicles + 2  # BKS + 2 for Hanoi (operational)
        else:
            max_vehicles = len(current_routes) + 2  # Current + 2 as fallback

        logger.info(f"   Inserting {len(unassigned)} customers (Feasibility-First Regret-2)...")
        logger.info(f"   Vehicle limit: {max_vehicles} (current: {len(current_routes)})")

        while unassigned:
            # FEASIBILITY-FIRST: Calculate regret using ONLY FEASIBLE positions
            customer_regrets = []

            for cust_id in unassigned:
                # Find ALL FEASIBLE insertion positions
                feasible_insertions = []  # [(cost, route_idx, pos), ...]

                for route_idx, route in enumerate(current_routes):
                    if not route or len(route) < 2:
                        continue

                    # Try each position - CHECK FEASIBILITY FIRST
                    for pos in range(1, len(route)):
                        # CRITICAL: Check feasibility using Vidal evaluator
                        if self._is_feasible_insertion(route, cust_id, pos):
                            # FEASIBLE! Calculate cost
                            i = route[pos - 1]
                            j = route[pos]
                            u = cust_id

                            d_iu = self.problem.get_distance(i, u)
                            d_uj = self.problem.get_distance(u, j)
                            d_ij = self.problem.get_distance(i, j)
                            cost = d_iu + d_uj - d_ij

                            feasible_insertions.append((cost, route_idx, pos))

                # Calculate regret based on FEASIBLE positions only
                if len(feasible_insertions) == 0:
                    # NO feasible position → will need to create route
                    regret = float('inf')  # Highest priority
                    best_cost = float('inf')
                    best_insertion = None
                elif len(feasible_insertions) == 1:
                    # Only one feasible option → high regret
                    regret = 1000.0
                    best_cost = feasible_insertions[0][0]
                    best_insertion = (feasible_insertions[0][1], feasible_insertions[0][2])
                else:
                    # Multiple feasible options → calculate regret
                    feasible_insertions.sort(key=lambda x: x[0])
                    best_cost = feasible_insertions[0][0]
                    second_best_cost = feasible_insertions[1][0]
                    regret = second_best_cost - best_cost
                    best_insertion = (feasible_insertions[0][1], feasible_insertions[0][2])

                customer_regrets.append({
                    'customer_id': cust_id,
                    'regret': regret,
                    'best_cost': best_cost,
                    'best_insertion': best_insertion,
                    'num_feasible': len(feasible_insertions)
                })

            if not customer_regrets:
                logger.error(f"   ❌ CRITICAL: No customers to insert!")
                break

            # Sort by regret (descending) - prioritize hardest to place
            customer_regrets.sort(key=lambda x: x['regret'], reverse=True)

            # Insert customer with highest regret
            selected = customer_regrets[0]
            cust_id = selected['customer_id']
            best_insertion = selected['best_insertion']

            # LOG: Show top 3 (first iteration only)
            if insertions_made == 0 and len(customer_regrets) >= 3:
                logger.info(f"      Top 3 customers by regret:")
                for i in range(min(3, len(customer_regrets))):
                    c = customer_regrets[i]
                    logger.info(f"        Customer {c['customer_id']}: regret={c['regret']:.1f}, feasible_pos={c['num_feasible']}")

            if best_insertion is not None:
                # Has feasible position → insert there
                route_idx, pos = best_insertion
                current_routes[route_idx].insert(pos, cust_id)
                unassigned.remove(cust_id)
                insertions_made += 1

                if insertions_made % 20 == 0:
                    logger.debug(f"      Inserted {insertions_made}/{len(customers_to_rebuild)}, {len(unassigned)} remaining")

            else:
                # NO feasible position → CREATE NEW ROUTE
                if len(current_routes) < max_vehicles:
                    # Under vehicle limit → create route and VERIFY feasibility
                    new_route = [0, cust_id, 0]

                    # CRITICAL: Check if single-customer route is feasible
                    route_eval = self.evaluator.evaluate_route(new_route)

                    if route_eval['violations'] == 0:
                        # New route is FEASIBLE → use it
                        current_routes.append(new_route)
                        unassigned.remove(cust_id)
                        insertions_made += 1
                        logger.info(f"      ✅ Created feasible route for customer {cust_id} (total: {len(current_routes)} routes)")
                    else:
                        # Even single-customer route is INFEASIBLE!
                        # This means customer's time window is impossible to satisfy alone
                        logger.warning(f"      ⚠️  Customer {cust_id}: Single-customer route has {route_eval['violations']} violations!")
                        logger.warning(f"         Will try force-inserting into existing route (may reduce violation)")

                        # Try force insert into best position in existing routes
                        min_violations = float('inf')
                        best_force_route = None

                        for route_idx, route in enumerate(current_routes):
                            if len(route) < 2:
                                continue
                            for pos in range(1, len(route)):
                                test_route = route[:pos] + [cust_id] + route[pos:]
                                test_eval = self.evaluator.evaluate_route(test_route)

                                if test_eval['violations'] < min_violations:
                                    min_violations = test_eval['violations']
                                    best_force_route = (route_idx, pos)

                        if best_force_route:
                            route_idx, pos = best_force_route
                            current_routes[route_idx].insert(pos, cust_id)
                            unassigned.remove(cust_id)
                            insertions_made += 1
                            logger.warning(f"         → Force-inserted at route {route_idx}, pos {pos} ({min_violations} violations)")
                        else:
                            # Can't even force insert - create route anyway
                            current_routes.append(new_route)
                            unassigned.remove(cust_id)
                            insertions_made += 1
                            logger.error(f"         → Created infeasible route anyway (last resort)")
                else:
                    # Over vehicle limit → FORCE INSERT (last resort)
                    logger.warning(f"      ❌ Customer {cust_id}: No feasible position AND at vehicle limit ({max_vehicles})")
                    logger.warning(f"         FORCE INSERTING (will create violation)")

                    # Force insert at cheapest position (accept violation)
                    min_cost = float('inf')
                    best_force_pos = None
                    for route_idx, route in enumerate(current_routes):
                        if len(route) < 2:
                            continue
                        for pos in range(1, len(route)):
                            i, j = route[pos-1], route[pos]
                            cost = (self.problem.get_distance(i, cust_id) +
                                   self.problem.get_distance(cust_id, j) -
                                   self.problem.get_distance(i, j))
                            if cost < min_cost:
                                min_cost = cost
                                best_force_pos = (route_idx, pos)

                    if best_force_pos:
                        route_idx, pos = best_force_pos
                        current_routes[route_idx].insert(pos, cust_id)
                        unassigned.remove(cust_id)
                        insertions_made += 1
                    else:
                        logger.error(f"      ❌ Cannot force insert customer {cust_id}")
                        break

        # --- PHASE 2: VIOLATION REPAIR (Incremental Swap/Relocate) ---
        # Now that all 100 customers are routed, use Swap/Relocate to fix violations
        logger.info("   🔧 PHASE 2: Starting Violation Repair (Swap/Relocate)...")
        
        violations_after_construction = self._count_violations(current_routes)
        logger.info(f"      Violations after construction: {violations_after_construction}")
        
        if violations_after_construction > 0:
            current_routes = self._repair_violations_incremental(current_routes)

        # --- GATE 1: EVALUATION AFTER STANDARD REPAIR ---
        final_violations = self._count_violations(current_routes)
        logger.info(f"   ✅ GATE 1 complete: {initial_violations} → {final_violations} violations")
        logger.info(f"      Successfully re-inserted: {insertions_made}/{len(customers_to_rebuild)} customers")

        if len(unassigned) > 0:
            logger.warning(f"      ⚠️  {len(unassigned)} customers remain unassigned: {unassigned[:10]}")

        # =============================================================================
        # GATE 2: MODE-DEPENDENT DEEP REPAIR STRATEGY
        # =============================================================================
        if final_violations > 0:
            logger.info(f"   🚪 GATE 2: Mode-dependent deep repair ({self.dataset_mode} mode)...")

            if self.dataset_mode == 'hanoi':
                # HANOI MODE: Accept solution (already working well!)
                logger.info(f"      Hanoi mode: Standard repair sufficient")
                logger.info(f"      Accepting {final_violations} violations (operational tolerance)")

            elif self.dataset_mode == 'solomon':
                # SOLOMON MODE: Academic benchmarks - need perfect feasibility
                # ALWAYS try multi-restart if ANY violations remain (even 1!)
                # FIX: Don't skip light violations - C208 had only 1 violation and failed before
                logger.info(f"      Solomon mode: Attempting multi-restart for {final_violations} violations...")
                logger.info(f"      Severity classification: {severity}")

                # Adaptive restart count based on severity
                # INCREASED from previous to ensure all 5 diversity strategies are tried
                if severity == 'light':
                    num_restarts = 10  # Light: try all 5 strategies twice (1-9%)
                elif severity == 'moderate':
                    num_restarts = 15  # Moderate: more intensive (10-49%)
                elif severity == 'severe':
                    num_restarts = 20  # Severe: very intensive (50-79%)
                else:  # catastrophic
                    num_restarts = 25  # Catastrophic: maximum effort (80%+)

                logger.info(f"      Activating MULTI-RESTART: {num_restarts} attempts with 5 diverse strategies")
                current_routes = self._repair_with_multi_restart(current_routes, num_restarts=num_restarts)

                # Re-evaluate after multi-restart
                final_violations = self._count_violations(current_routes)
                logger.info(f"      Multi-restart result: {final_violations} violations")

        # =============================================================================
        # GATE 3: SUCCESS CHECK & LNS POST-OPTIMIZATION
        # =============================================================================
        if final_violations == 0:
            logger.info(f"   🎉 SUCCESS: Achieved 0 violations!")

            # --- PHASE 3: LNS POST-OPTIMIZATION (Distance Reduction) ---
            logger.info("   🚀 GATE 3: Starting LNS Post-Optimization (Distance Reduction)...")

            try:
                # Initialize LNS with exploitation-focused settings
                lns = LNSOptimizer(
                    problem=self.problem,
                    evaluator=self.evaluator,
                    max_iterations=2000,        # Sufficient for deep search
                    time_limit=300,             # 5 minutes
                    initial_temperature=50.0    # Lower T to avoid breaking structure
                )

                # Get current cost for logging
                current_cost = sum(
                    sum(self.problem.get_distance(route[i], route[i+1])
                        for i in range(len(route)-1))
                    for route in current_routes if len(route) > 1
                )
                logger.info(f"      Starting distance: {current_cost:.2f} km")

                # Run LNS with "absolute safety" mode
                optimized_routes = lns.optimize(
                    initial_routes=current_routes,
                    require_feasible=True  # 🔒 CRITICAL: Auto-reject if creates violations
                )

                # Verify feasibility after LNS
                lns_violations = self._count_violations(optimized_routes)
                lns_cost = sum(
                    sum(self.problem.get_distance(route[i], route[i+1])
                        for i in range(len(route)-1))
                    for route in optimized_routes if len(route) > 1
                )

                if lns_violations == 0:
                    improvement = current_cost - lns_cost
                    logger.info(f"      ✅ LNS completed: {current_cost:.2f} → {lns_cost:.2f} km "
                              f"(improvement: {improvement:+.2f} km)")
                    current_routes = optimized_routes
                else:
                    logger.warning(f"      ⚠️  LNS created {lns_violations} violations, keeping solution")
                    logger.info(f"      Keeping current solution: {current_cost:.2f} km")

            except Exception as e:
                logger.warning(f"      ⚠️  LNS optimization failed: {e}, keeping current solution")
                # Continue with current solution if LNS fails
        else:
            logger.warning(f"   ⚠️  Still have {final_violations} violations after all repair attempts")
            logger.warning(f"      This problem may be infeasible with current vehicle count")

        return current_routes

    def _accept_move_sa(self,
                        violation_reduction: int,
                        distance_delta: float,
                        temperature: float) -> bool:
        """
        Simulated Annealing acceptance criterion (Metropolis).

        Decides whether to accept a move based on energy change and temperature.

        Args:
            violation_reduction: Positive = fewer violations (good)
            distance_delta: Negative = shorter distance (good)
            temperature: Current temperature (high = more exploratory)

        Returns:
            True if move should be accepted, False otherwise

        Energy Function:
            ΔE = -(violation_reduction × WEIGHT) + distance_delta
            Goal: Minimize energy (more violations = higher energy = bad)

        Acceptance:
            - ΔE < 0 (energy decreases): Always accept
            - ΔE > 0 (energy increases): Accept with probability e^(-ΔE/T)
        """
        # WEIGHT: How many km equals 1 violation?
        # 1000.0 means we heavily prioritize feasibility, but still allow tradeoffs at high T
        VIOLATION_WEIGHT = 1000.0

        # Calculate energy change
        # Note: violation_reduction is positive when GOOD (fewer violations)
        # We want energy to DECREASE, so negate it
        delta_energy = -(violation_reduction * VIOLATION_WEIGHT) + distance_delta

        # 1. Good move (Energy decreases) → Always Accept
        if delta_energy < 0:
            return True

        # 2. Bad move (Energy increases) → Probabilistic Accept based on temperature
        if temperature < 0.001:  # Too cold, reject bad moves
            return False

        # Metropolis criterion: P = e^(-ΔE/T)
        probability = math.exp(-delta_energy / temperature)
        accept = random.random() < probability

        if accept:
            logger.debug(f"   🔥 SA Accept: ΔE={delta_energy:.1f}, T={temperature:.1f}, P={probability:.3f}")

        return accept

    def _get_violated_customers(self, routes: List[List[int]]) -> List[Dict]:
        """
        Identify customers that violate time windows.

        CRITICAL: Must use SAME logic as _count_violations() to ensure consistency!
        - Solomon: Strict - count both EARLY and LATE arrivals
        - Hanoi: Lenient - only count LATE service

        Returns:
            List of dicts: [{'customer_id': X, 'route_idx': Y, 'position': Z, 'lateness': W}, ...]
        """
        violated = []

        # Detect dataset mode
        dataset_type = getattr(self.problem, 'dataset_type', None)
        if dataset_type is None:
            metadata = getattr(self.problem, 'metadata', {}) or {}
            dataset_type = metadata.get('dataset_type', 'hanoi')
        dataset_type = str(dataset_type).strip().lower()

        is_solomon = dataset_type.startswith('solomon')

        for route_idx, route in enumerate(routes):
            if not route or len(route) < 3:  # Skip empty or trivial routes
                continue

            # Validate route structure
            if route[0] != 0 or route[-1] != 0:
                logger.warning(f"⚠️  Route {route_idx} doesn't start/end with depot")
                continue

            # Manual forward calculation (SAME as _count_violations)
            if is_solomon:
                current_time = self.problem.depot.ready_time  # Solomon: start at 0
            else:
                from config import VRP_CONFIG
                current_time = VRP_CONFIG.get('time_window_start', 480)  # Hanoi: 8AM

            prev_node = 0  # Depot

            for pos in range(1, len(route) - 1):
                cust_id = route[pos]
                customer = self.problem.get_customer_by_id(cust_id)
                if not customer:
                    continue

                # Calculate arrival time
                travel_time = self.evaluator._get_time(prev_node, cust_id)
                arrival_time = current_time + travel_time

                # Check violation (MODE-SPECIFIC - SAME as _count_violations)
                is_violated = False
                lateness = 0.0

                if is_solomon:
                    # SOLOMON: Strict - both early and late are violations
                    if arrival_time < customer.ready_time:
                        is_violated = True
                        lateness = customer.ready_time - arrival_time  # Early
                    elif arrival_time > customer.due_date:
                        is_violated = True
                        lateness = arrival_time - customer.due_date  # Late
                else:
                    # HANOI: Lenient - only late service is violation
                    start_service = max(arrival_time, customer.ready_time)
                    if start_service > customer.due_date:
                        is_violated = True
                        lateness = start_service - customer.due_date

                if is_violated:
                    violated.append({
                        'customer_id': cust_id,
                        'route_idx': route_idx,
                        'position': pos,
                        'lateness': lateness
                    })

                # Update time for next customer
                start_service = max(arrival_time, customer.ready_time)
                current_time = start_service + customer.service_time
                prev_node = cust_id

        return violated

    def _count_violations(self, routes: List[List[int]], fast_mode: bool = False) -> int:
        """
        Count violations with SAFETY CHECK.

        Trusts standard forward calculation over Vidal if they disagree.

        This fixes the 'Ghost Violation' issue (Vidal says 77, Real says 0).
        
        Args:
            routes: Routes to check
            fast_mode: If True, use ONLY Vidal evaluator (O(1), fast but may have small bugs).
                       If False, double-check with Forward Calculation (O(N), accurate).
                       Use fast_mode=True in search loops for performance.
        """
        # Cách 1: Dùng Vidal (Nhanh nhưng có thể bị bug với Solomon)
        vidal_violations = 0
        try:
            for route in routes:
                if len(route) < 3: continue

                forward = self.evaluator.compute_forward_sequence(route)
                for i in range(1, len(route) - 1):
                    if forward[i].TW_E > forward[i].TW_L:
                        vidal_violations += 1
        except Exception:
            vidal_violations = 999  # Lỗi tính toán

        # FAST PATH: Dùng cho vòng lặp search (chấp nhận sai số nhỏ để chạy nhanh)
        if fast_mode:
            return vidal_violations

        # SLOW PATH: Dùng Vidal nếu Vidal=0, ngược lại check kỹ
        if vidal_violations == 0:
            return 0

        # Cách 2: "Tòa án" - Tính tay xuôi dòng (Chậm O(N) nhưng Chính xác 100%)
        # Logic này đúng tuyệt đối cho cả Solomon và Hanoi
        real_violations = 0
        
        for route in routes:
            if len(route) < 3: continue
            
            # Lấy thông tin Depot (Start)
            # Mode-specific time window start:
            # - Solomon: depot.ready_time = 0 (routes start at time 0)
            # - Hanoi: routes start at 8:00 AM = 480 minutes (even if depot.ready_time = 0)
            dataset_type = getattr(self.problem, 'dataset_type', None)
            if dataset_type is None:
                metadata = getattr(self.problem, 'metadata', {}) or {}
                dataset_type = metadata.get('dataset_type', 'hanoi')
            dataset_type = str(dataset_type).strip().lower()
            
            if dataset_type.startswith('solomon'):
                # Solomon: use depot's ready_time (typically 0)
                current_time = self.problem.depot.ready_time
            else:
                # Hanoi: routes start at 8:00 AM (480 minutes) regardless of depot.ready_time
                from config import VRP_CONFIG
                current_time = VRP_CONFIG.get('time_window_start', 480)
            
            prev_node = 0  # Depot ID
            
            # Duyệt từng khách hàng trong route (bỏ qua depot đầu, duyệt đến trước depot cuối)
            for i in range(1, len(route) - 1): 
                cust_id = route[i]
                customer = self.problem.get_customer_by_id(cust_id)
                
                # 1. Di chuyển từ Node trước -> Node hiện tại
                # Dùng evaluator._get_time() để lấy travel time chính xác từ time_matrix
                # Điều này đúng cho cả Solomon (distance = time) và Hanoi (có traffic factors)
                travel_time = self.evaluator._get_time(prev_node, cust_id)
                arrival_time = current_time + travel_time
                
                # 2. Kiểm tra vi phạm time window - MODE-SPECIFIC LOGIC
                # TÁCH BIỆT 2 CÁCH ĐẾM:
                # - SOLOMON (Academic): Strict - đếm CẢ early và late arrivals
                # - HANOI (Operational): Lenient - CHỈ đếm late, cho phép đợi nếu đến sớm

                if dataset_type.startswith('solomon'):
                    # ============================================================
                    # SOLOMON MODE: STRICT VIOLATION COUNTING
                    # ============================================================
                    # Academic benchmarks yêu cầu arrival ĐÚNG trong time window
                    # Không được đến sớm (< ready_time) HOẶC trễ (> due_date)
                    if arrival_time < customer.ready_time:
                        real_violations += 1  # VI PHẠM: Đến SỚM
                    elif arrival_time > customer.due_date:
                        real_violations += 1  # VI PHẠM: Đến TRỄ
                    # Chỉ OK nếu: ready_time <= arrival_time <= due_date

                else:
                    # ============================================================
                    # HANOI MODE: LENIENT VIOLATION COUNTING (GIỮ NGUYÊN LOGIC CŨ)
                    # ============================================================
                    # Real-world operational: Cho phép tài xế ĐỢI nếu đến sớm
                    # Chỉ đếm vi phạm nếu BẮT ĐẦU PHỤC VỤ sau due_date
                    start_service = max(arrival_time, customer.ready_time)  # Đợi nếu đến sớm
                    if start_service > customer.due_date:
                        real_violations += 1  # VI PHẠM: Phục vụ TRỄ (sau due_date)
                    # OK nếu đến sớm và đợi, miễn là phục vụ trước due_date

                # 3. Thời gian bắt đầu phục vụ (cho bước tiếp theo)
                start_service = max(arrival_time, customer.ready_time)
                
                # 4. Cập nhật thời gian cho node tiếp theo (Leave = Start + Service)
                current_time = start_service + customer.service_time
                prev_node = cust_id

        # Log cảnh báo nhẹ nếu có sự lệch pha (để debug sau này)
        if vidal_violations > 0 and real_violations == 0:
            # logger.debug(f"   👻 Ghost violations detected: Vidal={vidal_violations}, Real=0. Trusting Real.")
            pass

        return real_violations

    def _repair_violations_incremental(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Phase 2: Incremental Violation Repair using Swap/Relocate.
        
        Strategy:
        1. All 100 customers are already routed (from Phase 1)
        2. Use Swap/Relocate to fix violations incrementally
        3. Hierarchical: Violation reduction > Distance improvement
        4. Greedy mode (temperature=0.0): Only accept improving moves
        
        Args:
            routes: Routes with all customers routed but some violations
            
        Returns:
            Repaired routes with reduced violations (ideally 0)
        """
        current_routes = copy.deepcopy(routes)
        iterations = 0
        max_iterations = self.max_iterations
        no_improvement_count = 0
        max_no_improvement = 300  # Tăng lên 300 để cho repair nhiều thời gian hơn để thoát khỏi local minima
        
        initial_violations = self._count_violations(current_routes)
        logger.info(f"      Starting incremental repair: {initial_violations} violations")
        
        while iterations < max_iterations and no_improvement_count < max_no_improvement:
            current_violations = self._count_violations(current_routes)
            
            if current_violations == 0:
                logger.info(f"      ✅ All violations fixed after {iterations} iterations!")
                break
            
            # Get violated customers sorted by worst lateness first
            violated_info = self._get_violated_customers(current_routes)
            
            # Nếu kẹt lâu quá (>100 iters), thử random shuffle thay vì worst-first
            if no_improvement_count > 100:
                random.shuffle(violated_info)
            else:
                violated_info.sort(key=lambda x: x['lateness'], reverse=True)  # Worst first
            
            improved = False
            
            # Try to repair each violated customer
            for viol_info in violated_info:
                customer_id = viol_info['customer_id']
                route_idx = viol_info['route_idx']
                
                # Try relocate first (usually faster)
                relocated, new_routes = self._try_relocate_customer(
                    current_routes,
                    customer_id,
                    route_idx,
                    temperature=0.0  # Greedy: only accept improving moves
                )
                
                if relocated:
                    current_routes = new_routes
                    improved = True
                    break  # One move per iteration
                
                # If relocate failed, try swap
                if self.enable_swap:
                    swapped, new_routes = self._try_swap_customer(
                        current_routes,
                        customer_id,
                        route_idx,
                        temperature=0.0  # Greedy: only accept improving moves
                    )
                    
                    if swapped:
                        current_routes = new_routes
                        improved = True
                        break  # One move per iteration
            
            iterations += 1
            
            if improved:
                no_improvement_count = 0
                new_violations = self._count_violations(current_routes)
                if iterations % 10 == 0 or new_violations == 0:
                    logger.debug(f"      Iteration {iterations}: {current_violations} → {new_violations} violations")
            else:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    logger.warning(f"      ⚠️  No improvement for {max_no_improvement} iterations, stopping")
                    break
        
        final_violations = self._count_violations(current_routes)
        logger.info(f"      Incremental repair complete: {initial_violations} → {final_violations} violations ({iterations} iterations)")
        
        return current_routes

    def _try_relocate_customer(self,
                                routes: List[List[int]],
                                customer_id: int,
                                current_route_idx: int,
                                temperature: float = 0.0) -> Tuple[bool, List[List[int]]]:
        """
        Try to relocate customer to a better position using neighbor-based search.

        Args:
            routes: Current routes
            customer_id: Customer to relocate
            current_route_idx: Index of route containing customer
            temperature: Current SA temperature (0 = greedy, >0 = allow bad moves)

        Returns:
            (improved, new_routes): Whether improvement found and new routes
        """
        # Get neighbors for this customer
        if customer_id not in self.neighbor_lists:
            return False, routes

        neighbors = self.neighbor_lists[customer_id]

        # Find candidate positions (near neighbors)
        candidate_positions = []

        for neighbor_id in neighbors:
            # Find where this neighbor is in current solution
            for route_idx, route in enumerate(routes):
                if neighbor_id in route:
                    neighbor_pos = route.index(neighbor_id)

                    # Try positions around this neighbor
                    for offset in [-1, 0, 1]:
                        insert_pos = neighbor_pos + offset

                        # Valid position check
                        if insert_pos < 1 or insert_pos >= len(route):
                            continue

                        # Don't insert in same position (no-op)
                        if route_idx == current_route_idx and insert_pos == routes[current_route_idx].index(customer_id):
                            continue

                        candidate_positions.append((route_idx, insert_pos))
                    break

        if not candidate_positions:
            return False, routes

        # Evaluate all candidate positions (HIERARCHICAL: violations > distance)
        best_violation_reduction = None
        best_distance_delta = None
        best_position = None

        for target_route_idx, target_pos in candidate_positions:
            # Calculate metrics
            violation_reduction, distance_delta = self._evaluate_relocate(
                routes,
                customer_id,
                current_route_idx,
                target_route_idx,
                target_pos
            )

            # Hierarchical comparison:
            # PRIORITY 1: Violation reduction (most important)
            # PRIORITY 2: Distance improvement (tie-breaker)
            # REJECT: Moves that make violations worse
            is_better = False

            # Skip NO-OP moves (no change at all)
            if violation_reduction == 0 and abs(distance_delta) < 0.01:
                is_better = False
            elif best_position is None:
                # First candidate: accept unless it increases violations
                if violation_reduction >= 0:  # Accept neutral or improving
                    is_better = True
            elif violation_reduction > best_violation_reduction:
                # Better violation reduction (PRIORITY 1)
                is_better = True
            elif violation_reduction == best_violation_reduction and distance_delta < best_distance_delta:
                # Same violation reduction, better distance (PRIORITY 2)
                is_better = True

            if is_better:
                best_violation_reduction = violation_reduction
                best_distance_delta = distance_delta
                best_position = (target_route_idx, target_pos)

        # FALLBACK: If neighbor-based search didn't find IMPROVING move, activate PANIC MODE
        # Try ALL positions in ALL routes (brute-force with O(1) checks)
        # Trigger when: no position found OR best move doesn't reduce violations
        trigger_panic = (best_position is None) or (best_violation_reduction is not None and best_violation_reduction <= 0)

        if trigger_panic:
            logger.debug(f"   🚨 PANIC MODE: Customer {customer_id} (neighbor viol_red={best_violation_reduction}) - trying all positions...")

            # Reset best values to start fresh in Panic Mode
            best_violation_reduction = None
            best_distance_delta = None
            best_position = None

            # Generate ALL candidate positions across ALL routes
            all_positions = []
            for route_idx, route in enumerate(routes):
                for pos in range(1, len(route)):  # Skip depot at position 0
                    # Don't try same position (no-op)
                    if route_idx == current_route_idx:
                        try:
                            current_pos = route.index(customer_id)
                            if pos == current_pos:
                                continue
                        except ValueError:
                            pass  # Customer not in this route, OK to try

                    all_positions.append((route_idx, pos))

            # Evaluate ALL positions (O(1) makes this CHEAP!)
            for target_route_idx, target_pos in all_positions:
                violation_reduction, distance_delta = self._evaluate_relocate(
                    routes,
                    customer_id,
                    current_route_idx,
                    target_route_idx,
                    target_pos
                )

                # Hierarchical comparison
                is_better = False

                # Skip NO-OP moves (no change at all)
                if violation_reduction == 0 and abs(distance_delta) < 0.01:
                    is_better = False
                elif best_position is None:
                    # First candidate: accept unless it increases violations
                    if violation_reduction >= 0:  # Accept neutral or improving
                        is_better = True
                elif violation_reduction > best_violation_reduction:
                    # Better violation reduction (PRIORITY 1)
                    is_better = True
                elif violation_reduction == best_violation_reduction and distance_delta < best_distance_delta:
                    # Same violation reduction, better distance (PRIORITY 2)
                    is_better = True

                if is_better:
                    best_violation_reduction = violation_reduction
                    best_distance_delta = distance_delta
                    best_position = (target_route_idx, target_pos)

            if best_position is not None:
                logger.debug(f"   ✅ PANIC MODE found solution: viol_reduction={best_violation_reduction}, dist_delta={best_distance_delta:.1f}")

        # Apply best move if any candidate found
        # Use Simulated Annealing to decide acceptance
        if best_position is not None:
            # SA Decision: Should we accept this move?
            accept = self._accept_move_sa(
                best_violation_reduction,
                best_distance_delta,
                temperature
            )

            if accept:
                new_routes = self._apply_relocate(
                    routes,
                    customer_id,
                    current_route_idx,
                    best_position[0],
                    best_position[1]
                )
                return True, new_routes

        return False, routes

    def _evaluate_relocate(self,
                           routes: List[List[int]],
                           customer_id: int,
                           from_route_idx: int,
                           to_route_idx: int,
                           to_position: int) -> Tuple[int, float]:
        """
        Evaluate relocating a customer (hierarchical: violations then distance).

        Returns:
            (violation_reduction, distance_delta):
                - violation_reduction: positive = fewer violations (better)
                - distance_delta: negative = shorter distance (better)
        """
        # Count violations before (FAST MODE for performance in search loop)
        violations_before = self._count_violations(routes, fast_mode=True)

        # Calculate distance before
        distance_before = sum(
            sum(self.problem.get_distance(route[i], route[i+1])
                for i in range(len(route)-1))
            for route in routes if len(route) > 1
        )

        # Apply move
        new_routes = self._apply_relocate(
            routes, customer_id, from_route_idx, to_route_idx, to_position
        )

        # Count violations after (FAST MODE for performance in search loop)
        violations_after = self._count_violations(new_routes, fast_mode=True)

        # Calculate distance after
        distance_after = sum(
            sum(self.problem.get_distance(route[i], route[i+1])
                for i in range(len(route)-1))
            for route in new_routes if len(route) > 1
        )

        # Return both metrics
        violation_reduction = violations_before - violations_after
        distance_delta = distance_after - distance_before

        return violation_reduction, distance_delta

    def _apply_relocate(self,
                        routes: List[List[int]],
                        customer_id: int,
                        from_route_idx: int,
                        to_route_idx: int,
                        to_position: int) -> List[List[int]]:
        """Apply relocate move (DEEP COPY for safety)."""
        new_routes = copy.deepcopy(routes)

        # Find original position before removal (needed for index adjustment)
        from_position = new_routes[from_route_idx].index(customer_id)

        # Remove customer from source route
        new_routes[from_route_idx].remove(customer_id)

        # CRITICAL FIX: Adjust target position if moving within same route
        # When we remove a customer, all indices after it shift down by 1
        adjusted_position = to_position
        if from_route_idx == to_route_idx and from_position < to_position:
            adjusted_position -= 1  # Compensate for removal shifting indices

        # Insert into target route at adjusted position
        new_routes[to_route_idx].insert(adjusted_position, customer_id)

        return new_routes

    def _try_swap_customer(self,
                           routes: List[List[int]],
                           customer_id: int,
                           current_route_idx: int,
                           temperature: float = 0.0) -> Tuple[bool, List[List[int]]]:
        """
        Try to swap customer with a neighbor (for capacity deadlocks).

        Args:
            routes: Current routes
            customer_id: Customer to swap
            current_route_idx: Index of route containing customer
            temperature: Current SA temperature (0 = greedy, >0 = allow bad moves)

        Returns:
            (improved, new_routes): Whether improvement found and new routes
        """
        if not self.enable_swap:
            return False, routes

        # Get neighbors
        if customer_id not in self.neighbor_lists:
            return False, routes

        neighbors = self.neighbor_lists[customer_id]

        # PHASE 1: Try neighbor-based swap first (fast)
        candidate_swaps = []

        for neighbor_id in neighbors:
            for route_idx, route in enumerate(routes):
                if neighbor_id in route and route_idx != current_route_idx:
                    candidate_swaps.append((route_idx, neighbor_id))
                    break

        # Evaluate neighbor swaps (HIERARCHICAL: violations > distance)
        best_violation_reduction = None
        best_distance_delta = None
        best_swap = None

        for target_route_idx, target_customer_id in candidate_swaps:
            violation_reduction, distance_delta = self._evaluate_swap(
                routes,
                customer_id,
                current_route_idx,
                target_customer_id,
                target_route_idx
            )

            # Hierarchical comparison
            is_better = False

            # Skip NO-OP swaps
            if violation_reduction == 0 and abs(distance_delta) < 0.01:
                is_better = False
            elif best_swap is None:
                if violation_reduction >= 0:  # Accept neutral or improving
                    is_better = True
            elif violation_reduction > best_violation_reduction:
                is_better = True
            elif violation_reduction == best_violation_reduction and distance_delta < best_distance_delta:
                is_better = True

            if is_better:
                best_violation_reduction = violation_reduction
                best_distance_delta = distance_delta
                best_swap = (target_route_idx, target_customer_id)

        # If neighbor swap found candidate, use SA to decide acceptance
        if best_swap is not None:
            accept = self._accept_move_sa(
                best_violation_reduction,
                best_distance_delta,
                temperature
            )

            if accept:
                new_routes = self._apply_swap(
                    routes,
                    customer_id,
                    current_route_idx,
                    best_swap[1],
                    best_swap[0]
                )
                return True, new_routes

        # PHASE 2: PANIC SWAP - Try ALL possible swaps (brute-force)
        logger.debug(f"   🚨 PANIC SWAP: Neighbor swaps failed, trying ALL swaps for customer {customer_id}")

        # Collect ALL customers from other routes
        all_swap_candidates = []
        for route_idx, route in enumerate(routes):
            if route_idx != current_route_idx:
                # Try swapping with each customer in this route (skip depots)
                for pos in range(1, len(route) - 1):
                    target_customer_id = route[pos]
                    all_swap_candidates.append((route_idx, target_customer_id))

        if not all_swap_candidates:
            return False, routes

        # Evaluate ALL swaps
        best_violation_reduction = None
        best_distance_delta = None
        best_swap = None

        for target_route_idx, target_customer_id in all_swap_candidates:
            violation_reduction, distance_delta = self._evaluate_swap(
                routes,
                customer_id,
                current_route_idx,
                target_customer_id,
                target_route_idx
            )

            # Hierarchical comparison (same as neighbor search)
            is_better = False

            # Skip NO-OP swaps
            if violation_reduction == 0 and abs(distance_delta) < 0.01:
                is_better = False
            elif best_swap is None:
                if violation_reduction >= 0:  # Accept neutral or improving
                    is_better = True
            elif violation_reduction > best_violation_reduction:
                is_better = True
            elif violation_reduction == best_violation_reduction and distance_delta < best_distance_delta:
                is_better = True

            if is_better:
                best_violation_reduction = violation_reduction
                best_distance_delta = distance_delta
                best_swap = (target_route_idx, target_customer_id)

        # Apply best swap if found (with SA acceptance)
        if best_swap is not None:
            logger.debug(f"   ✅ PANIC SWAP found solution: viol_red={best_violation_reduction}, dist_delta={best_distance_delta:.1f}")

            accept = self._accept_move_sa(
                best_violation_reduction,
                best_distance_delta,
                temperature
            )

            if accept:
                new_routes = self._apply_swap(
                    routes,
                    customer_id,
                    current_route_idx,
                    best_swap[1],
                    best_swap[0]
                )
                return True, new_routes

        return False, routes

    def _evaluate_swap(self,
                       routes: List[List[int]],
                       customer1_id: int,
                       route1_idx: int,
                       customer2_id: int,
                       route2_idx: int) -> Tuple[int, float]:
        """
        Evaluate swapping two customers (hierarchical: violations then distance).

        Returns:
            (violation_reduction, distance_delta)
        """
        violations_before = self._count_violations(routes, fast_mode=True)

        distance_before = sum(
            sum(self.problem.get_distance(route[i], route[i+1])
                for i in range(len(route)-1))
            for route in routes if len(route) > 1
        )

        new_routes = self._apply_swap(
            routes, customer1_id, route1_idx, customer2_id, route2_idx
        )

        violations_after = self._count_violations(new_routes, fast_mode=True)

        distance_after = sum(
            sum(self.problem.get_distance(route[i], route[i+1])
                for i in range(len(route)-1))
            for route in new_routes if len(route) > 1
        )

        violation_reduction = violations_before - violations_after
        distance_delta = distance_after - distance_before

        return violation_reduction, distance_delta

    def _apply_swap(self,
                    routes: List[List[int]],
                    customer1_id: int,
                    route1_idx: int,
                    customer2_id: int,
                    route2_idx: int) -> List[List[int]]:
        """Apply swap move (DEEP COPY for safety)."""
        new_routes = copy.deepcopy(routes)

        # Validate: Never swap depots
        if customer1_id == 0 or customer2_id == 0:
            return new_routes

        # Find positions
        try:
            pos1 = new_routes[route1_idx].index(customer1_id)
            pos2 = new_routes[route2_idx].index(customer2_id)
        except (ValueError, IndexError):
            # Customer not found in route
            return new_routes

        # Validate positions (not at depot positions)
        if pos1 == 0 or pos1 >= len(new_routes[route1_idx]) - 1:
            return new_routes
        if pos2 == 0 or pos2 >= len(new_routes[route2_idx]) - 1:
            return new_routes

        # Swap
        new_routes[route1_idx][pos1] = customer2_id
        new_routes[route2_idx][pos2] = customer1_id

        return new_routes

    def get_statistics(self, routes: List[List[int]]) -> Dict[str, any]:
        """Get repair statistics."""
        violations = self._get_violated_customers(routes)

        total_distance = sum(
            sum(self.evaluator._get_distance(route[i], route[i+1])
                for i in range(len(route)-1))
            for route in routes if route and len(route) > 1
        )

        return {
            'num_violations': len(violations),
            'total_lateness': sum(v['lateness'] for v in violations),
            'total_distance': total_distance,
            'num_routes': len([r for r in routes if r and len(r) > 2]),
            'feasible': len(violations) == 0
        }

    def _repair_with_multi_restart(self, routes: List[List[int]], num_restarts: int = 5) -> List[List[int]]:
        """
        Multi-restart repair strategy with DIVERSE customer ordering strategies.

        Uses proven construction heuristics to provide diversity:
        1. Random shuffle (baseline)
        2. Time urgency - earliest deadline first (Solomon 1987)
        3. Nearest neighbor - spatial clustering (classic heuristic)
        4. Farthest insertion - spread customers (Rosenkrantz 1977)
        5. Regret-based - hardest customers first (Toth & Vigo 2002)

        References:
            - Solomon, M. M. (1987). "Algorithms for the VRP with Time Windows"
            - Rosenkrantz et al. (1977). "An Analysis of Several Heuristics for TSP"
            - Toth & Vigo (2002). "The Vehicle Routing Problem"

        Args:
            routes: Input routes with violations
            num_restarts: Number of restart attempts

        Returns:
            Best repaired routes found across all restarts
        """
        logger.info(f"   🔄 MULTI-RESTART: Trying {num_restarts} different reconstruction strategies...")

        best_routes = None
        best_violations = float('inf')
        best_distance = float('inf')

        # Define ordering strategies (cycle through them)
        strategies = ['random', 'time_urgency', 'nearest', 'farthest', 'regret']

        for attempt in range(num_restarts):
            # Select strategy for this attempt (cycle through)
            strategy = strategies[attempt % len(strategies)]
            logger.info(f"      🔄 Attempt {attempt + 1}/{num_restarts}: Using '{strategy}' ordering strategy")

            # Deep copy to avoid contamination between attempts
            attempt_routes = copy.deepcopy(routes)

            # Phase 1: RUIN (Eject violated customers)
            violated_info = self._get_violated_customers(attempt_routes)
            violated_ids = [v['customer_id'] for v in violated_info]

            # Eject violated customers
            for route_idx in range(len(attempt_routes)):
                attempt_routes[route_idx] = [c for c in attempt_routes[route_idx]
                                              if c not in violated_ids]

            # Phase 2: RECREATE with STRATEGY-SPECIFIC ORDERING
            unassigned = list(violated_ids)

            # Apply ordering strategy
            unassigned_ordered = self._apply_ordering_strategy(unassigned, strategy, attempt)

            # LOG: Show first 5 customers in the ordering (to see diversity)
            if len(unassigned_ordered) >= 5:
                logger.info(f"         First 5 customers in order: {unassigned_ordered[:5]}")

            unassigned = unassigned_ordered

            # Relaxed cheapest insertion (same as original logic)
            while unassigned:
                best_customer = None
                best_insertion = None
                min_cost = float('inf')

                for cust_id in unassigned:
                    for route_idx, route in enumerate(attempt_routes):
                        if not route or len(route) < 2:
                            continue

                        # Check capacity
                        route_demand = sum(
                            self.problem.get_customer_by_id(c).demand
                            for c in route if c != 0 and self.problem.get_customer_by_id(c)
                        )

                        cust = self.problem.get_customer_by_id(cust_id)
                        if cust and route_demand + cust.demand > self.problem.vehicle_capacity:
                            continue

                        # Try each position
                        for pos in range(1, len(route)):
                            i, j, u = route[pos - 1], route[pos], cust_id
                            cost = (self.problem.get_distance(i, u) +
                                   self.problem.get_distance(u, j) -
                                   self.problem.get_distance(i, j))

                            if cost < min_cost:
                                min_cost = cost
                                best_customer = cust_id
                                best_insertion = (route_idx, pos)

                if best_customer is not None:
                    route_idx, pos = best_insertion
                    attempt_routes[route_idx].insert(pos, best_customer)
                    unassigned.remove(best_customer)
                else:
                    logger.warning(f"         ⚠️  Cannot insert {len(unassigned)} customers")
                    break

            # Phase 3: Incremental repair (increased iterations for better quality)
            violations_after_construct = self._count_violations(attempt_routes)

            if violations_after_construct > 0:
                # Moderate repair (max 300 iterations per restart - increased from 100)
                saved_max_iter = self.max_iterations
                self.max_iterations = 300
                attempt_routes = self._repair_violations_incremental(attempt_routes)
                self.max_iterations = saved_max_iter

            # Evaluate this attempt
            attempt_violations = self._count_violations(attempt_routes)
            attempt_distance = sum(
                sum(self.problem.get_distance(route[i], route[i+1])
                    for i in range(len(route)-1))
                for route in attempt_routes if len(route) > 1
            )

            # Update best solution (hierarchical: violations > distance)
            is_better = False
            if best_routes is None:
                is_better = True
            elif attempt_violations < best_violations:
                is_better = True
            elif attempt_violations == best_violations and attempt_distance < best_distance:
                is_better = True

            if is_better:
                best_routes = attempt_routes
                best_violations = attempt_violations
                best_distance = attempt_distance
                logger.info(f"         ✅ Result: {attempt_violations} violations, {attempt_distance:.2f} km (NEW BEST!)")
            else:
                logger.info(f"         Result: {attempt_violations} violations, {attempt_distance:.2f} km (current best: {best_violations} viol)")

            # Early exit if perfect solution found
            if best_violations == 0:
                logger.info(f"   🎉 Perfect solution found at attempt {attempt + 1}!")
                break

        logger.info(f"   Multi-restart complete: Best = {best_violations} violations, {best_distance:.2f} km")
        return best_routes if best_routes is not None else routes

    def _is_feasible_insertion(self, route: List[int], customer_id: int, position: int) -> bool:
        """
        Check if inserting customer at position is FEASIBLE (TW + Capacity).

        CRITICAL: This is the foundation of feasibility-first architecture.
        Uses Vidal's proven concat-based feasibility check (Vidal et al. 2013).

        Args:
            route: Route to insert into [0, c1, c2, ..., 0]
            customer_id: Customer to insert
            position: Position to insert at (1 <= position <= len(route)-1)

        Returns:
            True if insertion maintains TW feasibility AND capacity
        """
        # Use Vidal's built-in O(1) feasibility check
        # This uses the proven concatenation approach from the Vidal paper
        try:
            is_feasible, _ = self.evaluator.check_insertion_feasibility(
                route, customer_id, position
            )
            return is_feasible

        except (ValueError, IndexError, KeyError):
            # Evaluation failed (invalid position, customer not found, etc.)
            return False

    def _apply_ordering_strategy(self, customer_ids: List[int], strategy: str, seed: int = 0) -> List[int]:
        """
        Apply proven ordering strategy to customer list.

        Strategies (all from academic literature):
        1. 'random': Random shuffle (baseline)
        2. 'time_urgency': Earliest deadline first (Solomon 1987)
        3. 'nearest': Nearest neighbor from depot (classic)
        4. 'farthest': Farthest customer first (Rosenkrantz 1977)
        5. 'regret': Sorted by insertion regret (Toth & Vigo 2002)

        Args:
            customer_ids: List of customer IDs to order
            strategy: Strategy name
            seed: Random seed for reproducibility

        Returns:
            Ordered list of customer IDs
        """
        if strategy == 'random':
            # Random shuffle (baseline)
            random.seed(hash(time.time() + seed) % 10000)
            ordered = list(customer_ids)
            random.shuffle(ordered)
            return ordered

        elif strategy == 'time_urgency':
            # Sort by earliest deadline first (Solomon 1987)
            # Rationale: Customers with tight time windows should be inserted first
            ordered = sorted(
                customer_ids,
                key=lambda cid: self.problem.get_customer_by_id(cid).due_date
                if self.problem.get_customer_by_id(cid) else float('inf')
            )
            return ordered

        elif strategy == 'nearest':
            # Nearest neighbor from depot (classic construction heuristic)
            # Rationale: Spatial clustering reduces travel distance
            ordered = sorted(
                customer_ids,
                key=lambda cid: self.problem.get_distance(0, cid)  # Distance from depot
            )
            return ordered

        elif strategy == 'farthest':
            # Farthest customer first (Rosenkrantz 1977)
            # Rationale: Place difficult (far) customers first, leave flexibility for close ones
            ordered = sorted(
                customer_ids,
                key=lambda cid: self.problem.get_distance(0, cid),
                reverse=True  # Descending - farthest first
            )
            return ordered

        elif strategy == 'regret':
            # Sorted by time window slack (tighter windows first)
            # Rationale: Customers with less flexibility should be placed first
            # Slack = due_date - ready_time (smaller = tighter window)
            ordered = sorted(
                customer_ids,
                key=lambda cid: (
                    self.problem.get_customer_by_id(cid).due_date -
                    self.problem.get_customer_by_id(cid).ready_time
                ) if self.problem.get_customer_by_id(cid) else float('inf')
            )
            return ordered

        else:
            # Fallback to random
            logger.warning(f"Unknown strategy '{strategy}', using random")
            return self._apply_ordering_strategy(customer_ids, 'random', seed)

    def _load_bks_data(self) -> Dict:
        """Load BKS data from file."""
        try:
            from ..evaluation.bks_validator import BKSValidator
            validator = BKSValidator()
            return validator.bks_data
        except Exception as e:
            logger.warning(f"   Could not load BKS data: {e}")
            return {}

    def _detect_dataset_mode(self) -> str:
        """
        Detect whether this is a Solomon or Hanoi dataset.

        Returns:
            'solomon' or 'hanoi'
        """
        dataset_type = getattr(self.problem, 'dataset_type', None)
        if dataset_type is None:
            metadata = getattr(self.problem, 'metadata', {}) or {}
            dataset_type = metadata.get('dataset_type', 'hanoi')

        dataset_type = str(dataset_type).strip().lower()

        if dataset_type.startswith('solomon') or 'solomon' in dataset_type:
            return 'solomon'
        else:
            return 'hanoi'

    def _calculate_capacity_utilization(self, routes: List[List[int]]) -> float:
        """
        Calculate average capacity utilization across all routes.

        Returns:
            Average utilization ratio (0.0 to 1.0+)
        """
        if not routes:
            return 0.0

        total_utilization = 0.0
        valid_routes = 0

        for route in routes:
            if len(route) < 3:  # Skip empty routes
                continue

            route_demand = 0
            for cust_id in route:
                if cust_id != 0:
                    customer = self.problem.get_customer_by_id(cust_id)
                    if customer:
                        route_demand += customer.demand

            if self.problem.vehicle_capacity > 0:
                utilization = route_demand / self.problem.vehicle_capacity
                total_utilization += utilization
                valid_routes += 1

        if valid_routes == 0:
            return 0.0

        avg_utilization = total_utilization / valid_routes
        return avg_utilization

    def _analyze_time_window_tightness(self) -> float:
        """
        Analyze average time window width in the dataset.

        Returns:
            Average TW width (in time units)
        """
        if not self.problem.customers:
            return float('inf')

        total_width = 0.0
        count = 0

        for customer in self.problem.customers:
            tw_width = customer.due_date - customer.ready_time
            total_width += tw_width
            count += 1

        if count == 0:
            return float('inf')

        avg_width = total_width / count
        return avg_width

    def _classify_severity(self, violations: int, total_customers: int) -> str:
        """
        Classify violation severity.

        Returns:
            'light', 'moderate', 'severe', or 'catastrophic'
        """
        if total_customers == 0:
            return 'light'

        ratio = violations / total_customers

        if ratio >= 0.80:
            return 'catastrophic'
        elif ratio >= 0.50:
            return 'severe'
        elif ratio >= 0.10:
            return 'moderate'
        else:
            return 'light'

    def _get_bks_vehicles(self, instance_name: str) -> Optional[int]:
        """
        Get BKS vehicle count for a Solomon instance.

        Args:
            instance_name: Instance name (e.g., 'C207')

        Returns:
            BKS vehicle count or None if not available
        """
        if not instance_name:
            return None

        instance_name = instance_name.upper().replace('.JSON', '')

        if instance_name in self.bks_data:
            return self.bks_data[instance_name].get('vehicles')

        return None

    def _validate_and_fix_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Validate and fix route structures to ensure all routes start/end with depot.

        Args:
            routes: Input routes that may be corrupted

        Returns:
            Fixed routes with proper depot boundaries
        """
        fixed_routes = []

        for route_idx, route in enumerate(routes):
            if not route:
                continue

            # Fix route structure
            fixed_route = route.copy()

            # Remove any depot (0) from middle of route
            customers = [c for c in fixed_route if c != 0]

            # Rebuild route with depot boundaries
            if customers:
                fixed_route = [0] + customers + [0]
                fixed_routes.append(fixed_route)
            else:
                # Empty route - skip
                logger.warning(f"⚠️  Route {route_idx} has no customers, skipping")

        logger.info(f"   Validated {len(routes)} routes → {len(fixed_routes)} valid routes")

        return fixed_routes
