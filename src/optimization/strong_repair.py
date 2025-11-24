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

        # Remove violated customers from routes
        for route_idx in range(len(current_routes)):
            current_routes[route_idx] = [c for c in current_routes[route_idx]
                                         if c not in violated_ids]

        logger.info(f"   🔨 RECREATE PHASE: Re-inserting using Regret-2 heuristic...")

        # --- PHASE 2: RECREATE (Regret-2 Insertion) ---
        unassigned = list(violated_ids)
        insertions_made = 0

        while unassigned:
            best_customer = None
            best_insertion = None  # (route_idx, pos, cost)
            max_regret = -float('inf')

            # Calculate regret for each unassigned customer
            any_feasible = False

            for cust_id in unassigned:
                # Find all valid insertion positions for this customer
                valid_insertions = self._find_valid_insertions(current_routes, cust_id)

                if not valid_insertions:
                    continue  # This customer has no feasible positions

                any_feasible = True

                # Calculate Regret-2
                best_cost = valid_insertions[0][0]

                if len(valid_insertions) > 1:
                    second_best_cost = valid_insertions[1][0]
                    regret = second_best_cost - best_cost
                else:
                    # Only one feasible position - infinite regret (highest priority)
                    regret = float('inf')

                # Select customer with maximum regret
                if regret > max_regret:
                    max_regret = regret
                    best_customer = cust_id
                    best_insertion = valid_insertions[0]  # (cost, route_idx, pos)

            # Insert the selected customer
            if best_customer is not None:
                cost, route_idx, pos = best_insertion
                current_routes[route_idx].insert(pos, best_customer)
                unassigned.remove(best_customer)
                insertions_made += 1

                if insertions_made % 10 == 0:
                    logger.debug(f"      Inserted {insertions_made}/{len(violated_ids)} customers, {len(unassigned)} remaining")
            else:
                # No feasible positions for any remaining customers
                logger.warning(f"   ⚠️  Cannot insert remaining {len(unassigned)} customers (constraint deadlock)")
                break

        # --- PHASE 3: FINAL CHECK ---
        final_violations = self._count_violations(current_routes)
        logger.info(f"   ✅ Reconstructive Repair complete: {initial_violations} → {final_violations} violations")
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

    def _evaluate_insertion_feasibility(self, route: List[int], customer_id: int, position: int) -> Tuple[bool, float]:
        """
        Evaluate if inserting customer at position is feasible using O(1) Vidal concatenation.

        Args:
            route: The route to insert into [0, c1, c2, ..., cn, 0]
            customer_id: Customer to insert
            position: Position to insert at (1 <= position <= len(route)-1)

        Returns:
            (is_feasible, distance_delta):
                - is_feasible: True if insertion maintains TW feasibility
                - distance_delta: Change in route distance (positive = increase)
        """
        # Build temporary route with insertion
        temp_route = route[:position] + [customer_id] + route[position:]

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
