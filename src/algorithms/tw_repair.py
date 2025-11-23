"""Time-window repair operator applied after Split decoding - ENHANCED VERSION."""

from __future__ import annotations

import copy
import logging
import time
import random
from typing import List, Tuple, Optional

from src.models.vrp_model import VRPProblem

logger = logging.getLogger(__name__)


class TWRepairOperator:
    """Enhanced post-split repair operator with exhaustive search, restart, and adaptive weighting."""

    def __init__(
        self,
        problem: VRPProblem,
        max_iterations: int = 50,
        violation_weight: float = 50.0,
        max_relocations_per_route: int = 2,
        max_routes_to_try: Optional[int] = None,
        max_positions_to_try: Optional[int] = None,
        max_iterations_soft: Optional[int] = None,
        max_routes_soft_limit: Optional[int] = None,
        max_positions_soft_limit: Optional[int] = None,
        lateness_soft_threshold: Optional[float] = None,
        lateness_skip_threshold: Optional[float] = None,
        # NEW PARAMETERS (backward compatible with defaults)
        enable_exhaustive_search: bool = True,
        enable_restart: bool = True,
        restart_after_no_improvement: int = 50,
        adaptive_weight_multiplier: float = 10.0,
    ):
        self.problem = problem
        self.max_iterations = max_iterations
        self.violation_weight = violation_weight
        self.initial_violation_weight = violation_weight  # Store initial for reset
        self.max_relocations_per_route = max_relocations_per_route
        self.max_routes_to_try = max_routes_to_try
        self.max_positions_to_try = max_positions_to_try
        self.max_iterations_soft = max_iterations_soft or max(1, max_iterations // 3)
        self.max_routes_soft_limit = max_routes_soft_limit
        self.max_positions_soft_limit = max_positions_soft_limit
        self.lateness_soft_threshold = lateness_soft_threshold
        self.lateness_skip_threshold = lateness_skip_threshold

        # NEW: Enhanced features
        self.enable_exhaustive_search = enable_exhaustive_search
        self.enable_restart = enable_restart
        self.restart_after_no_improvement = restart_after_no_improvement
        self.adaptive_weight_multiplier = adaptive_weight_multiplier

    def repair_routes(self, routes: List[List[int]]) -> List[List[int]]:
        if not routes:
            return routes

        start_time = time.perf_counter()
        best_routes = copy.deepcopy(routes)
        total_lateness = self._total_lateness(best_routes)

        if total_lateness <= 1e-6:
            return best_routes

        # Reset violation weight to initial value
        self.violation_weight = self.initial_violation_weight

        # Decide mode (full vs soft) based on lateness
        soft_mode = False
        effective_max_iterations = self.max_iterations
        routes_limit = self.max_routes_to_try
        positions_limit = self.max_positions_to_try

        if self.lateness_soft_threshold and total_lateness >= self.lateness_soft_threshold:
            soft_mode = True
            effective_max_iterations = max(1, min(self.max_iterations_soft, self.max_iterations))
            routes_limit = self.max_routes_soft_limit or self.max_routes_to_try
            positions_limit = self.max_positions_soft_limit or self.max_positions_to_try

        # MULTI-PASS STRATEGY
        # Pass 1: Greedy (fast, first-improvement)
        # Pass 2: Exhaustive (thorough, best-improvement)
        # Pass 3: Restart with shuffle (if stuck)
        # Pass 4: Adaptive weight escalation (if still stuck)

        best_score = self._evaluate_routes(best_routes)
        global_best_routes = copy.deepcopy(best_routes)
        global_best_score = best_score
        global_best_lateness = total_lateness

        relocation_time = 0.0
        swap_time = 0.0
        restart_count = 0
        weight_escalation_count = 0

        # Pass 1: GREEDY search (fast)
        logger.info("🔧 TW repair Pass 1: Greedy search (first-improvement)")
        best_routes, best_score, stats = self._repair_pass(
            best_routes,
            best_score,
            max_iterations=min(50, effective_max_iterations),
            routes_limit=routes_limit,
            positions_limit=positions_limit,
            search_mode="greedy",
            max_no_improvement=5
        )
        relocation_time += stats['relocation_time']
        swap_time += stats['swap_time']

        current_lateness = self._total_lateness(best_routes)
        if current_lateness < global_best_lateness:
            global_best_routes = copy.deepcopy(best_routes)
            global_best_score = best_score
            global_best_lateness = current_lateness

        # If already 0 violations, done!
        if current_lateness <= 1.0:
            total_time = time.perf_counter() - start_time
            logger.info(
                "TW repair: ✅ 0 violations achieved in Pass 1 (greedy), "
                "%.1fms, %d iterations",
                total_time * 1000,
                stats['iterations']
            )
            return best_routes

        # Pass 2: EXHAUSTIVE search (thorough, best-improvement)
        if self.enable_exhaustive_search:
            logger.info("🔧 TW repair Pass 2: Exhaustive search (best-improvement)")
            best_routes, best_score, stats = self._repair_pass(
                best_routes,
                best_score,
                max_iterations=effective_max_iterations,
                routes_limit=routes_limit,
                positions_limit=positions_limit,
                search_mode="exhaustive",
                max_no_improvement=100  # More patient in exhaustive mode
            )
            relocation_time += stats['relocation_time']
            swap_time += stats['swap_time']

            current_lateness = self._total_lateness(best_routes)
            if current_lateness < global_best_lateness:
                global_best_routes = copy.deepcopy(best_routes)
                global_best_score = best_score
                global_best_lateness = current_lateness

            if current_lateness <= 1.0:
                total_time = time.perf_counter() - start_time
                logger.info(
                    "TW repair: ✅ 0 violations achieved in Pass 2 (exhaustive), "
                    "%.1fms, total iterations: %d",
                    total_time * 1000,
                    stats['iterations']
                )
                return best_routes

        # Pass 3: RESTART with shuffle (if stuck and enabled)
        if self.enable_restart and current_lateness > 1.0:
            logger.info("🔧 TW repair Pass 3: Restart with shuffle")

            for restart_attempt in range(5):  # Try 5 restarts for stubborn cases
                # Shuffle the route with most violations
                shuffled_routes = self._shuffle_worst_route(best_routes)
                shuffled_score = self._evaluate_routes(shuffled_routes)

                # Run exhaustive search on shuffled solution
                shuffled_routes, shuffled_score, stats = self._repair_pass(
                    shuffled_routes,
                    shuffled_score,
                    max_iterations=effective_max_iterations // 2,
                    routes_limit=routes_limit,
                    positions_limit=positions_limit,
                    search_mode="exhaustive",
                    max_no_improvement=50
                )
                relocation_time += stats['relocation_time']
                swap_time += stats['swap_time']
                restart_count += 1

                current_lateness = self._total_lateness(shuffled_routes)
                if current_lateness < global_best_lateness:
                    global_best_routes = copy.deepcopy(shuffled_routes)
                    global_best_score = shuffled_score
                    global_best_lateness = current_lateness
                    best_routes = shuffled_routes
                    best_score = shuffled_score

                if current_lateness <= 1.0:
                    total_time = time.perf_counter() - start_time
                    logger.info(
                        "TW repair: ✅ 0 violations achieved after %d restart(s), "
                        "%.1fms",
                        restart_attempt + 1,
                        total_time * 1000
                    )
                    return best_routes

        # Pass 4: ADAPTIVE WEIGHT escalation (if still stuck)
        if current_lateness > 1.0 and self.adaptive_weight_multiplier > 1.0:
            logger.info("🔧 TW repair Pass 4: Adaptive weight escalation")

            # Escalate violation_weight
            original_weight = self.violation_weight
            self.violation_weight *= self.adaptive_weight_multiplier
            weight_escalation_count += 1

            logger.info(
                "TW repair: Escalating violation_weight: %.1f → %.1f",
                original_weight,
                self.violation_weight
            )

            # Recalculate score with new weight
            best_score = self._evaluate_routes(best_routes)

            # Run exhaustive search with higher weight
            best_routes, best_score, stats = self._repair_pass(
                best_routes,
                best_score,
                max_iterations=effective_max_iterations,
                routes_limit=routes_limit,
                positions_limit=positions_limit,
                search_mode="exhaustive",
                max_no_improvement=100
            )
            relocation_time += stats['relocation_time']
            swap_time += stats['swap_time']

            current_lateness = self._total_lateness(best_routes)
            if current_lateness < global_best_lateness:
                global_best_routes = copy.deepcopy(best_routes)
                global_best_score = best_score
                global_best_lateness = current_lateness

        # Use global best
        best_routes = global_best_routes
        final_lateness = global_best_lateness

        total_time = time.perf_counter() - start_time

        # Comprehensive logging
        if total_time > 0.05 or final_lateness > 1.0:
            mode_label = "soft" if soft_mode else "full"
            logger.info(
                "TW repair (%s): lateness %.2f → %.2f, %.1fms total "
                "(reloc %.1fms, swap %.1fms), %d routes, %d customers, "
                "restarts=%d, weight_escalations=%d",
                mode_label,
                total_lateness,
                final_lateness,
                total_time * 1000,
                relocation_time * 1000,
                swap_time * 1000,
                len(routes),
                sum(len(r) for r in routes),
                restart_count,
                weight_escalation_count
            )

        # Reset violation weight to initial
        self.violation_weight = self.initial_violation_weight

        return best_routes

    def _repair_pass(
        self,
        routes: List[List[int]],
        initial_score: float,
        max_iterations: int,
        routes_limit: Optional[int],
        positions_limit: Optional[int],
        search_mode: str,  # "greedy" or "exhaustive"
        max_no_improvement: int
    ) -> Tuple[List[List[int]], float, dict]:
        """
        Single repair pass with specified search mode.

        Returns:
            (best_routes, best_score, stats_dict)
        """
        best_routes = copy.deepcopy(routes)
        best_score = initial_score

        iterations = 0
        improved = True
        no_improvement_count = 0
        relocation_time = 0.0
        swap_time = 0.0

        while (iterations < max_iterations and
               (self._total_lateness(best_routes) > 1.0 or improved) and
               no_improvement_count < max_no_improvement):

            improved = False
            iterations += 1

            # Try relocation
            reloc_start = time.perf_counter()
            move = self._find_best_relocation(
                best_routes,
                best_score,
                routes_limit,
                positions_limit,
                search_mode
            )
            relocation_time += time.perf_counter() - reloc_start

            # Try swap if relocation failed
            if not move:
                swap_start = time.perf_counter()
                move = self._find_best_swap(
                    best_routes,
                    best_score,
                    routes_limit,
                    positions_limit,
                    search_mode
                )
                swap_time += time.perf_counter() - swap_start

            if move:
                best_routes = move["routes"]
                best_score = move["score"]
                improved = True
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        stats = {
            'iterations': iterations,
            'relocation_time': relocation_time,
            'swap_time': swap_time,
            'no_improvement_count': no_improvement_count
        }

        return best_routes, best_score, stats

    # ------------------------------------------------------------------
    # Moves (ENHANCED)
    # ------------------------------------------------------------------
    def _find_best_relocation(
        self,
        routes,
        current_score,
        routes_limit: Optional[int],
        positions_limit: Optional[int],
        search_mode: str = "greedy"
    ):
        """
        Find best relocation move.

        Args:
            search_mode: "greedy" (first-improvement) or "exhaustive" (best-improvement)
        """
        best = None
        best_score_improvement = 0.0
        max_routes = routes_limit if routes_limit is not None else len(routes)
        max_positions = positions_limit if positions_limit is not None else float('inf')

        # In exhaustive mode, remove max_relocations_per_route limit
        max_customers_to_try = (
            self.max_relocations_per_route
            if search_mode == "greedy"
            else float('inf')
        )

        for i, route in enumerate(routes):
            lateness_info = self._late_customers(route)
            moves_tried = 0

            # PRIORITY FIX: Sort by lateness (worst violations first) in exhaustive mode
            if search_mode == "exhaustive":
                lateness_info = sorted(lateness_info, key=lambda x: x[1], reverse=True)

            for pos, lateness in lateness_info:
                if lateness <= 0:
                    continue

                # Limit routes to try
                routes_to_try = min(max_routes, len(routes))

                for j in range(routes_to_try):
                    # Limit positions to try per route
                    route_len = len(routes[j])
                    positions_to_try = min(max_positions, route_len - 1) if max_positions < float('inf') else route_len - 1

                    # Sample positions evenly if limiting
                    # IMPORTANT: range must be range(1, route_len - 1) to exclude ending depot
                    if positions_to_try < route_len - 2:
                        step = max(1, (route_len - 2) // positions_to_try)
                        position_range = range(1, route_len - 1, step)[:positions_to_try]
                    else:
                        position_range = range(1, route_len - 1)

                    for insert_pos in position_range:
                        if i == j and (insert_pos == pos or insert_pos == pos + 1):
                            continue

                        # DEEP COPY to avoid shared references
                        new_routes = copy.deepcopy(routes)

                        if i == j:
                            # Same route relocation
                            new_route = routes[i][:]
                            cust = new_route.pop(pos)
                            if len(new_route) < 2:
                                new_route = [0, 0]
                            new_route.insert(insert_pos, cust)
                            if not self._capacity_ok(new_route):
                                continue
                            new_routes[i] = new_route
                            old_cost = self._route_cost(routes[i])
                            new_cost = self._route_cost(new_route)
                            score = current_score - old_cost + new_cost
                        else:
                            # Cross-route relocation
                            route_from = routes[i][:]
                            cust = route_from.pop(pos)
                            if len(route_from) < 2:
                                route_from = [0, 0]
                            route_to = routes[j][:]
                            route_to.insert(insert_pos, cust)
                            if not self._capacity_ok(route_from) or not self._capacity_ok(route_to):
                                continue
                            new_routes[i] = route_from
                            new_routes[j] = route_to
                            old_cost = self._route_cost(routes[i]) + self._route_cost(routes[j])
                            new_cost = self._route_cost(route_from) + self._route_cost(route_to)
                            score = current_score - old_cost + new_cost

                        # Check if this move improves score
                        improvement = current_score - score
                        if improvement > 1e-6:
                            if search_mode == "greedy":
                                # GREEDY: Accept first improvement immediately
                                return {"routes": new_routes, "score": score}
                            else:
                                # EXHAUSTIVE: Keep track of best improvement
                                if improvement > best_score_improvement:
                                    best = {"routes": new_routes, "score": score}
                                    best_score_improvement = improvement

                moves_tried += 1
                if search_mode == "greedy" and moves_tried >= max_customers_to_try:
                    break

            # In greedy mode, stop after first route with improvement
            if search_mode == "greedy" and best:
                break

        return best

    def _find_best_swap(
        self,
        routes,
        current_score,
        routes_limit: Optional[int],
        positions_limit: Optional[int],
        search_mode: str = "greedy"
    ):
        """
        Find best swap move.

        Args:
            search_mode: "greedy" (first-improvement) or "exhaustive" (best-improvement)
        """
        best = None
        best_score_improvement = 0.0
        n_routes = len(routes)
        max_routes = routes_limit if routes_limit is not None else n_routes
        max_routes = min(max_routes, n_routes)

        for i in range(max_routes):
            for j in range(i + 1, min(i + 1 + max_routes, n_routes)):
                route1 = routes[i]
                route2 = routes[j]
                lateness1 = self._late_customers(route1)
                lateness2 = self._late_customers(route2)
                indices1 = [idx for idx, late in lateness1 if late > 0]
                indices2 = [idx for idx, late in lateness2 if late > 0]

                # If no violations in either route, skip
                if not indices1 and not indices2:
                    continue

                # Candidates: prioritize late customers, but include all if none late
                candidates1 = indices1 or list(range(1, len(route1) - 1))
                candidates2 = indices2 or list(range(1, len(route2) - 1))

                # Limit candidates if max_positions_to_try is set
                if positions_limit and len(candidates1) > positions_limit:
                    step = max(1, len(candidates1) // positions_limit)
                    candidates1 = candidates1[::step][:positions_limit]
                if positions_limit and len(candidates2) > positions_limit:
                    step = max(1, len(candidates2) // positions_limit)
                    candidates2 = candidates2[::step][:positions_limit]

                for pos1 in candidates1:
                    for pos2 in candidates2:
                        if not (1 <= pos1 < len(route1) - 1 and 1 <= pos2 < len(route2) - 1):
                            continue

                        # DEEP COPY to avoid shared references
                        new_routes = copy.deepcopy(routes)
                        new_route1 = routes[i][:]
                        new_route2 = routes[j][:]
                        new_route1[pos1], new_route2[pos2] = new_route2[pos2], new_route1[pos1]

                        if not self._capacity_ok(new_route1) or not self._capacity_ok(new_route2):
                            continue

                        new_routes[i] = new_route1
                        new_routes[j] = new_route2
                        old_cost = self._route_cost(routes[i]) + self._route_cost(routes[j])
                        new_cost = self._route_cost(new_route1) + self._route_cost(new_route2)
                        score = current_score - old_cost + new_cost

                        # Check if this move improves score
                        improvement = current_score - score
                        if improvement > 1e-6:
                            if search_mode == "greedy":
                                # GREEDY: Accept first improvement immediately
                                return {"routes": new_routes, "score": score}
                            else:
                                # EXHAUSTIVE: Keep track of best improvement
                                if improvement > best_score_improvement:
                                    best = {"routes": new_routes, "score": score}
                                    best_score_improvement = improvement

        return best

    def _shuffle_worst_route(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Shuffle the route with most time window violations.

        This helps escape local minima by disrupting the current structure.
        """
        if not routes:
            return routes

        # Find route with most violations
        worst_route_idx = 0
        max_violations = 0

        for idx, route in enumerate(routes):
            lateness_info = self._late_customers(route)
            violations = sum(1 for _, late in lateness_info if late > 0)
            if violations > max_violations:
                max_violations = violations
                worst_route_idx = idx

        # If no violations, pick random route
        if max_violations == 0:
            worst_route_idx = random.randint(0, len(routes) - 1)

        # Shuffle the customers in this route (excluding depot)
        shuffled_routes = copy.deepcopy(routes)
        route = shuffled_routes[worst_route_idx]

        if len(route) > 3:  # Has at least 2 customers (depot-cust1-cust2-depot)
            customers = route[1:-1]  # Exclude depots
            random.shuffle(customers)
            shuffled_routes[worst_route_idx] = [0] + customers + [0]

        return shuffled_routes

    # ------------------------------------------------------------------
    # Helpers (UNCHANGED)
    # ------------------------------------------------------------------
    def _capacity_ok(self, route: List[int]) -> bool:
        load = 0.0
        for cid in route:
            if cid == 0:
                continue
            customer = self.problem.get_customer_by_id(cid)
            if customer is None:
                continue
            load += customer.demand
            if load > self.problem.vehicle_capacity:
                return False
        return True

    def _late_customers(self, route: List[int]) -> List[Tuple[int, float]]:
        results = []
        current_time = 0.0
        for idx in range(1, len(route) - 1):
            prev = route[idx - 1]
            cid = route[idx]
            customer = self.problem.get_customer_by_id(cid)
            if customer is None:
                continue
            travel = self.problem.get_distance(prev, cid)
            arrival = current_time + travel
            if arrival < customer.ready_time:
                arrival = customer.ready_time
            lateness = max(0.0, arrival - customer.due_date)
            results.append((idx, lateness))
            current_time = arrival + customer.service_time
        return results

    def _evaluate_routes(self, routes: List[List[int]]) -> float:
        total_distance = 0.0
        total_lateness = 0.0
        for route in routes:
            if not route:
                continue
            total_distance += self._route_distance(route)
            total_lateness += self._route_lateness(route)
        return total_distance + self.violation_weight * total_lateness

    def _route_distance(self, route: List[int]) -> float:
        dist = 0.0
        for idx in range(len(route) - 1):
            dist += self.problem.get_distance(route[idx], route[idx + 1])
        return dist

    def _route_lateness(self, route: List[int]) -> float:
        lateness = 0.0
        current_time = 0.0
        for idx in range(1, len(route) - 1):
            prev = route[idx - 1]
            cid = route[idx]
            customer = self.problem.get_customer_by_id(cid)
            if customer is None:
                continue
            travel = self.problem.get_distance(prev, cid)
            arrival = current_time + travel
            if arrival < customer.ready_time:
                arrival = customer.ready_time
            lateness += max(0.0, arrival - customer.due_date)
            current_time = arrival + customer.service_time
        return lateness

    def _route_cost(self, route: List[int]) -> float:
        if not route:
            return 0.0
        return self._route_distance(route) + self.violation_weight * self._route_lateness(route)

    def _total_lateness(self, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self._route_lateness(route)
        return total
