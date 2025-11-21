"""Time-window repair operator applied after Split decoding."""

from __future__ import annotations

import copy
import logging
import time
from typing import List, Tuple, Optional

from src.models.vrp_model import VRPProblem

logger = logging.getLogger(__name__)


class TWRepairOperator:
    """Post-split repair operator that relocates and swaps customers to reduce TW violations."""

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
    ):
        self.problem = problem
        self.max_iterations = max_iterations
        self.violation_weight = violation_weight
        self.max_relocations_per_route = max_relocations_per_route
        self.max_routes_to_try = max_routes_to_try  # Limit routes to try for relocation
        self.max_positions_to_try = max_positions_to_try  # Limit positions to try per route
        self.max_iterations_soft = max_iterations_soft or max(1, max_iterations // 3)
        self.max_routes_soft_limit = max_routes_soft_limit
        self.max_positions_soft_limit = max_positions_soft_limit
        self.lateness_soft_threshold = lateness_soft_threshold
        self.lateness_skip_threshold = lateness_skip_threshold

    def repair_routes(self, routes: List[List[int]]) -> List[List[int]]:
        if not routes:
            return routes

        start_time = time.perf_counter()
        best_routes = copy.deepcopy(routes)
        total_lateness = self._total_lateness(best_routes)
        if total_lateness <= 1e-6:
            return best_routes

        if self.lateness_skip_threshold and total_lateness >= self.lateness_skip_threshold:
            logger.info(
                "TW repair skipped (lateness %.1f >= skip threshold %.1f)",
                total_lateness,
                self.lateness_skip_threshold,
            )
            return best_routes

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

        best_score = self._evaluate_routes(best_routes)

        iterations = 0
        improved = True
        no_improvement_count = 0
        max_no_improvement = 5  # Early termination: stop if no improvement for 5 iterations
        relocation_time = 0.0
        swap_time = 0.0
        while improved and iterations < effective_max_iterations and no_improvement_count < max_no_improvement:
            improved = False
            iterations += 1

            reloc_start = time.perf_counter()
            move = self._find_best_relocation(best_routes, best_score, routes_limit, positions_limit)
            relocation_time += time.perf_counter() - reloc_start
            
            if not move:
                swap_start = time.perf_counter()
                move = self._find_best_swap(best_routes, best_score, routes_limit, positions_limit)
                swap_time += time.perf_counter() - swap_start
            
            if move:
                best_routes = move["routes"]
                best_score = move["score"]
                improved = True
                no_improvement_count = 0  # Reset counter on improvement
            else:
                no_improvement_count += 1  # Increment if no improvement

        total_time = time.perf_counter() - start_time
        if total_time > 0.05:
            mode_label = "soft" if soft_mode else "full"
            logger.info(
                "TW repair (%s): %s iterations, %.1fms total (reloc %.1fms, swap %.1fms), "
                "%s routes, %s customers, lateness %.2f",
                mode_label,
                iterations,
                total_time * 1000,
                relocation_time * 1000,
                swap_time * 1000,
                len(routes),
                sum(len(r) for r in routes),
                total_lateness,
            )

        return best_routes

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------
    def _find_best_relocation(self, routes, current_score,
                              routes_limit: Optional[int],
                              positions_limit: Optional[int]):
        best = None
        max_routes = routes_limit if routes_limit is not None else len(routes)
        max_positions = positions_limit if positions_limit is not None else float('inf')
        
        for i, route in enumerate(routes):
            lateness_info = self._late_customers(route)
            moves_tried = 0
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
                    if positions_to_try < route_len - 1:
                        step = max(1, (route_len - 1) // positions_to_try)
                        position_range = range(1, route_len, step)[:positions_to_try]
                    else:
                        position_range = range(1, route_len)
                    
                    for insert_pos in position_range:
                        if i == j and (insert_pos == pos or insert_pos == pos + 1):
                            continue
                        new_routes = list(routes)
                        if i == j:
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

                        if score + 1e-6 < current_score:
                            best = {"routes": new_routes, "score": score}
                            current_score = score
                            break
                    if best:
                        break
                moves_tried += 1
                if moves_tried >= self.max_relocations_per_route or best:
                    break
            if best:
                break
        return best

    def _find_best_swap(self, routes, current_score,
                        routes_limit: Optional[int],
                        positions_limit: Optional[int]):
        best = None
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
                if not indices1 and not indices2:
                    continue
                candidates1 = indices1 or range(1, len(route1) - 1)
                candidates2 = indices2 or range(1, len(route2) - 1)
                
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
                        new_routes = list(routes)
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
                        if score + 1e-6 < current_score:
                            best = {"routes": new_routes, "score": score}
                            current_score = score
                            break
                    if best:
                        break
                if best:
                    break
            if best:
                break
        return best

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _capacity_ok(self, route: List[int]) -> bool:
        load = 0.0
        for cid in route:
            if cid == 0:
                continue
            customer = self.problem.get_customer_by_id(cid)
            if customer is None:
                # Invalid customer ID, skip
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
                # Invalid customer ID, skip
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
                # Invalid customer ID, skip
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

