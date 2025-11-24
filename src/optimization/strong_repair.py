"""
Strong Repair Operator for VRP.

Implements PHASE 3: STRONG REPAIR (Feasibility Enforcer)
- Violated-customers-only strategy (don't fix what's not broken)
- Neighbor-based search (40 closest instead of all N)
- Best-improvement (scan all candidates, pick best)
- O(1) feasibility checks via VidalEvaluator

Expected: Reduce distance from ~1066km → ~650km while maintaining 0 violations
"""

import numpy as np
import logging
import copy
import math
import random
from typing import List, Tuple, Optional, Dict, Set
from ..models.vrp_model import VRPProblem
from .vidal_evaluator import VidalEvaluator

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

    def repair_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Repair routes to eliminate time window violations.

        Args:
            routes: List of routes (each route is list of customer IDs)

        Returns:
            Repaired routes with 0 violations (hopefully)

        Process:
            1. Identify violated customers (those arriving late)
            2. For each violated customer (worst first):
                a. Try relocating to positions near its 40 closest neighbors
                b. Use best-improvement (try all, pick best)
                c. If relocate fails, try swap
            3. Repeat until 0 violations or max_iterations
        """
        logger.info("🔧 Strong Repair: Starting repair...")
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

        # =============================================================================
        # RECONSTRUCTIVE REPAIR (Ruin & Recreate with Regret-2)
        # =============================================================================
        logger.info(f"   💣 RUIN PHASE: Ejecting {initial_violations} violated customers...")

        # --- PHASE 1: RUIN (Eject all violated customers) ---
        violated_info = self._get_violated_customers(current_routes)
        violated_ids = [v['customer_id'] for v in violated_info]

        # EMERGENCY RESET LOGIC: If too many violations, reset ALL routes
        EMERGENCY_THRESHOLD = 50  # If >50% of customers violated, full reset
        total_customers = sum(1 for route in current_routes for c in route if c != 0)

        if len(violated_ids) > EMERGENCY_THRESHOLD:
            logger.warning(f"   🚨 EMERGENCY RESET: {len(violated_ids)}/{total_customers} violations (>{EMERGENCY_THRESHOLD})")
            logger.warning(f"   Remaining {total_customers - len(violated_ids)} customers in wrong positions → Full Reset!")

            # Collect ALL customers (not just violated ones)
            all_customers = []
            for route in current_routes:
                for cust_id in route:
                    if cust_id != 0:  # Skip depot
                        all_customers.append(cust_id)

            # Reset all routes to empty [0, 0]
            num_routes = len(current_routes)
            current_routes = [[0, 0] for _ in range(num_routes)]

            # Set all customers as unassigned
            violated_ids = all_customers
            logger.info(f"   Reset complete: {num_routes} empty routes, {len(violated_ids)} customers to rebuild")
        else:
            # Normal RUIN: Only eject violated customers
            for route_idx in range(len(current_routes)):
                current_routes[route_idx] = [c for c in current_routes[route_idx]
                                             if c not in violated_ids]

        logger.info(f"   🔨 RECREATE PHASE: Cheapest insertion (ALLOW TW violations during construction)")

        # --- PHASE 2: RECREATE (Relaxed Cheapest Insertion) ---
        # Strategy: Insert ALL customers first (allow TW violations)
        #          TW violations will be naturally minimized by choosing good positions
        # This ensures all 100 customers are routed!

        unassigned = list(violated_ids)
        insertions_made = 0

        logger.info(f"   Inserting {len(unassigned)} customers (relaxed TW constraints)...")

        while unassigned:
            best_customer = None
            best_insertion = None  # (route_idx, pos, cost)
            min_cost = float('inf')

            for cust_id in unassigned:
                # Find best insertion position (distance-based + check capacity only)
                for route_idx, route in enumerate(current_routes):
                    if not route or len(route) < 2:
                        continue

                    # Calculate route demand
                    route_demand = 0
                    for c in route:
                        if c != 0:
                            customer = self.problem.get_customer_by_id(c)
                            if customer:
                                route_demand += customer.demand

                    # Check capacity constraint
                    cust = self.problem.get_customer_by_id(cust_id)
                    if cust and route_demand + cust.demand > self.problem.vehicle_capacity:
                        continue  # Skip this route - capacity exceeded

                    # Try each position
                    for pos in range(1, len(route)):
                        i = route[pos - 1]
                        j = route[pos]
                        u = cust_id

                        # Calculate insertion cost (distance only)
                        d_iu = self.problem.get_distance(i, u)
                        d_uj = self.problem.get_distance(u, j)
                        d_ij = self.problem.get_distance(i, j)
                        cost = d_iu + d_uj - d_ij  # Net distance increase

                        if cost < min_cost:
                            min_cost = cost
                            best_customer = cust_id
                            best_insertion = (route_idx, pos)

            # Insert the selected customer
            if best_customer is not None:
                route_idx, pos = best_insertion
                current_routes[route_idx].insert(pos, best_customer)
                unassigned.remove(best_customer)
                insertions_made += 1

                if insertions_made % 20 == 0:
                    logger.debug(f"      Inserted {insertions_made}/{len(violated_ids)} customers, {len(unassigned)} remaining")
            else:
                # This should never happen with relaxed insertion
                logger.error(f"   ❌ CRITICAL: Cannot insert {len(unassigned)} customers even with relaxed constraints!")
                logger.error(f"      This suggests a bug in the insertion logic or capacity constraints too tight")
                break

        # --- PHASE 2: VIOLATION REPAIR (Incremental Swap/Relocate) ---
        # Now that all 100 customers are routed, use Swap/Relocate to fix violations
        logger.info("   🔧 PHASE 2: Starting Violation Repair (Swap/Relocate)...")
        
        violations_after_construction = self._count_violations(current_routes)
        logger.info(f"      Violations after construction: {violations_after_construction}")
        
        if violations_after_construction > 0:
            current_routes = self._repair_violations_incremental(current_routes)
        
        # --- PHASE 3: FINAL CHECK ---
        final_violations = self._count_violations(current_routes)
        logger.info(f"   ✅ Repair Pipeline complete: {initial_violations} → {final_violations} violations")
        logger.info(f"      Successfully re-inserted: {insertions_made}/{len(violated_ids)} customers")

        if len(unassigned) > 0:
            logger.warning(f"      ⚠️  {len(unassigned)} customers remain unassigned: {unassigned[:10]}")

        if final_violations == 0:
            logger.info(f"   🎉 SUCCESS: Achieved 0 violations!")

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

        Returns:
            List of dicts: [{'customer_id': X, 'route_idx': Y, 'position': Z, 'lateness': W}, ...]
        """
        violated = []

        for route_idx, route in enumerate(routes):
            if not route or len(route) < 3:  # Skip empty or trivial routes
                continue

            # Validate route structure (must start and end with depot)
            if route[0] != 0 or route[-1] != 0:
                logger.warning(f"⚠️  Route {route_idx} doesn't start/end with depot: {route[:3]}...{route[-3:]}")
                continue

            # Compute forward sequence
            try:
                forward = self.evaluator.compute_forward_sequence(route)
            except ValueError as e:
                logger.warning(f"⚠️  Route {route_idx} validation failed: {e}")
                continue

            # Check each customer (skip depots)
            for pos in range(1, len(route) - 1):
                node = forward[pos]
                customer_id = route[pos]

                # Check if late (TW_E > TW_L means infeasible)
                if node.TW_E > node.TW_L:
                    lateness = node.TW_E - node.TW_L
                    violated.append({
                        'customer_id': customer_id,
                        'route_idx': route_idx,
                        'position': pos,
                        'lateness': lateness
                    })

        return violated

    def _count_violations(self, routes: List[List[int]]) -> int:
        """Count total number of violated customers."""
        return len(self._get_violated_customers(routes))

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
        max_no_improvement = 50  # Stop if no improvement for 50 iterations
        
        initial_violations = self._count_violations(current_routes)
        logger.info(f"      Starting incremental repair: {initial_violations} violations")
        
        while iterations < max_iterations and no_improvement_count < max_no_improvement:
            current_violations = self._count_violations(current_routes)
            
            if current_violations == 0:
                logger.info(f"      ✅ All violations fixed after {iterations} iterations!")
                break
            
            # Get violated customers sorted by worst lateness first
            violated_info = self._get_violated_customers(current_routes)
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
        # Count violations before
        violations_before = self._count_violations(routes)

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

        # Count violations after
        violations_after = self._count_violations(new_routes)

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
        violations_before = self._count_violations(routes)

        distance_before = sum(
            sum(self.problem.get_distance(route[i], route[i+1])
                for i in range(len(route)-1))
            for route in routes if len(route) > 1
        )

        new_routes = self._apply_swap(
            routes, customer1_id, route1_idx, customer2_id, route2_idx
        )

        violations_after = self._count_violations(new_routes)

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

    def _find_valid_insertions(self, routes: List[List[int]], customer_id: int) -> List[Tuple[float, int, int]]:
        """
        Find all valid insertion positions for a customer using O(1) Vidal evaluator.

        Args:
            routes: Current routes
            customer_id: Customer to insert

        Returns:
            List of valid insertions sorted by cost: [(cost, route_idx, pos), ...]
            where cost = distance_delta (increase in route distance)
        """
        valid_insertions = []

        # Try inserting in each route at each position
        for route_idx, route in enumerate(routes):
            if not route or len(route) < 2:
                continue

            # Try each position (skip depot at 0, can insert from position 1 to len-1)
            for pos in range(1, len(route)):
                # Check feasibility using O(1) Vidal concatenation
                is_feasible, distance_delta = self._evaluate_insertion_feasibility(
                    route, customer_id, pos
                )

                if is_feasible:
                    # Use distance_delta as cost
                    valid_insertions.append((distance_delta, route_idx, pos))

        # Sort by cost (ascending - prefer insertions with least distance increase)
        valid_insertions.sort(key=lambda x: x[0])

        return valid_insertions

    def _find_valid_insertions_i1(
        self,
        routes: List[List[int]],
        customer_id: int,
        alpha1: float,
        alpha2: float,
        mu: float,
        avg_tw_width: float
    ) -> List[Tuple[float, int, int]]:
        """
        Find all valid insertion positions using Solomon's I1 heuristic.

        I1 formula:
            c11(i,u,j) = d(i,u) + d(u,j) - μ × d(i,j)  [distance increase]
            c12(i,u,j) = b(j|i,u) - b(j|i)              [time shift]
            I1 cost = α1 × c11 + α2 × c12

        Args:
            routes: Current routes
            customer_id: Customer to insert
            alpha1: Distance weight
            alpha2: Time urgency weight
            mu: Route savings parameter
            avg_tw_width: Average TW width for normalization

        Returns:
            List of valid insertions sorted by I1 cost: [(i1_cost, route_idx, pos), ...]
        """
        valid_insertions = []
        customer = self.problem.get_customer_by_id(customer_id)
        if not customer:
            return valid_insertions

        # Try inserting in each route at each position
        for route_idx, route in enumerate(routes):
            if not route or len(route) < 2:
                continue

            # Try each position (skip depot at 0, can insert from position 1 to len-1)
            for pos in range(1, len(route)):
                # Check basic feasibility first
                is_feasible, distance_delta = self._evaluate_insertion_feasibility(
                    route, customer_id, pos
                )

                if not is_feasible:
                    continue

                # Calculate I1 cost components
                i = route[pos - 1]  # Customer before insertion point
                j = route[pos]      # Customer after insertion point
                u = customer_id     # Customer to insert

                # c11: Distance increase (with route savings)
                d_iu = self.problem.get_distance(i, u)
                d_uj = self.problem.get_distance(u, j)
                d_ij = self.problem.get_distance(i, j)
                c11 = d_iu + d_uj - mu * d_ij

                # c12: Time shift (push-forward effect)
                # Calculate how much customer j is pushed forward in time
                try:
                    # Original arrival time at j (without u)
                    forward_original = self.evaluator.compute_forward_sequence(route)
                    original_idx = pos  # j is at position pos in original route
                    b_j_original = forward_original[original_idx].TW_E

                    # New arrival time at j (with u inserted)
                    temp_route = route[:pos] + [u] + route[pos:]
                    forward_new = self.evaluator.compute_forward_sequence(temp_route)
                    new_idx = pos + 1  # j is now at position pos+1
                    b_j_new = forward_new[new_idx].TW_E

                    # Time shift (normalized by average TW width)
                    c12 = (b_j_new - b_j_original) / avg_tw_width if avg_tw_width > 0 else 0.0

                except (ValueError, IndexError):
                    # If evaluation fails, use high penalty
                    c12 = 1000.0

                # I1 cost = weighted combination
                i1_cost = alpha1 * c11 + alpha2 * c12

                valid_insertions.append((i1_cost, route_idx, pos))

        # Sort by I1 cost (ascending - prefer insertions with lowest I1 cost)
        valid_insertions.sort(key=lambda x: x[0])

        return valid_insertions

    def _evaluate_insertion_feasibility(self, route: List[int], customer_id: int, position: int) -> Tuple[bool, float]:
        """
        Evaluate if inserting customer at position is feasible using O(1) Vidal concatenation.

        Args:
            route: The route to insert into [0, c1, c2, ..., cn, 0]
            customer_id: Customer to insert
            position: Position to insert at (1 <= position <= len(route)-1)

        Returns:
            (is_feasible, distance_delta):
                - is_feasible: True if insertion maintains TW feasibility AND capacity
                - distance_delta: Change in route distance (positive = increase)
        """
        # Build temporary route with insertion
        temp_route = route[:position] + [customer_id] + route[position:]

        # CAPACITY CHECK: Ensure route doesn't exceed vehicle capacity
        route_demand = 0
        for cust_id in temp_route:
            if cust_id != 0:  # Skip depot
                customer = self.problem.get_customer_by_id(cust_id)
                if customer:
                    route_demand += customer.demand

        if route_demand > self.problem.vehicle_capacity:
            return False, 0.0  # Capacity violation

        # Check feasibility using Vidal forward sequence
        try:
            forward = self.evaluator.compute_forward_sequence(temp_route)

            # Check if any customer violates time windows
            for i in range(1, len(temp_route) - 1):  # Skip depots
                node = forward[i]
                if node.TW_E > node.TW_L:  # Late arrival
                    return False, 0.0

            # Feasible! Calculate distance delta
            distance_before = sum(
                self.problem.get_distance(route[i], route[i+1])
                for i in range(len(route)-1)
            )

            distance_after = sum(
                self.problem.get_distance(temp_route[i], temp_route[i+1])
                for i in range(len(temp_route)-1)
            )

            distance_delta = distance_after - distance_before

            return True, distance_delta

        except (ValueError, IndexError) as e:
            # Evaluation failed (e.g., invalid route structure)
            return False, 0.0

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
