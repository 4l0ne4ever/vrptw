"""
Large Neighborhood Search (LNS) Post-Optimizer for VRP.

Implements PHASE 4: LNS POST-OPTIMIZATION
- Destroy operators: Random, Worst, Route Removal (secret weapon)
- Repair operator: Regret-2 insertion
- Simulated Annealing acceptance
- Global best tracking

Expected: Reduce distance from ~650km → ~600km (close to BKS 588km)
"""

import numpy as np
import logging
import copy
import time
from typing import List, Tuple, Optional, Dict
from ..models.vrp_model import VRPProblem
from .vidal_evaluator import VidalEvaluator

logger = logging.getLogger(__name__)


class LNSOptimizer:
    """
    Large Neighborhood Search for post-optimization.

    Key Concept: "Destroy and Repair"
    1. DESTROY: Remove customers from solution (create holes)
    2. REPAIR: Reinsert customers optimally (fill holes better)
    3. ACCEPT: Keep if better, or accept with probability (SA)
    4. REPEAT: 2000 iterations or 5 minutes

    The "Secret Weapon": Route Removal
    - Instead of removing random customers, remove ENTIRE route
    - Forces algorithm to redistribute customers across other routes
    - Often finds much better solutions than partial removal
    """

    def __init__(self,
                 problem: VRPProblem,
                 evaluator: VidalEvaluator,
                 max_iterations: int = 2000,
                 time_limit: int = 300,
                 initial_temperature: float = 100.0):
        """
        Initialize LNS optimizer.

        Args:
            problem: VRP problem instance
            evaluator: Vidal evaluator for O(1) checks
            max_iterations: Maximum iterations (default 2000)
            time_limit: Time limit in seconds (default 300 = 5 minutes)
            initial_temperature: Initial temperature for SA
        """
        self.problem = problem
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.initial_temperature = initial_temperature

    def optimize(self,
                 initial_routes: List[List[int]],
                 require_feasible: bool = True) -> List[List[int]]:
        """
        Optimize routes using LNS.

        Args:
            initial_routes: Starting routes (should be feasible)
            require_feasible: Only accept feasible solutions (default True)

        Returns:
            Optimized routes

        Process:
            1. Start with initial solution (global best)
            2. Loop (max_iterations or time_limit):
                a. Choose destroy operator randomly
                b. Destroy: Remove customers
                c. Repair: Reinsert with Regret-2
                d. Evaluate new solution
                e. Accept/reject with SA criteria
                f. Update global best if improved
            3. Return global best
        """
        logger.info("🚀 LNS Optimizer: Starting post-optimization...")
        logger.info(f"   Max iterations: {self.max_iterations}")
        logger.info(f"   Time limit: {self.time_limit}s")
        logger.info(f"   Require feasible: {require_feasible}")

        # Initialize
        start_time = time.time()
        best_routes = copy.deepcopy(initial_routes)
        best_cost = self._calculate_cost(best_routes)
        best_violations = self._count_violations(best_routes)

        current_routes = copy.deepcopy(initial_routes)
        current_cost = best_cost

        logger.info(f"   Initial: {best_cost:.2f} km, {best_violations} violations")

        # Statistics
        improvements = 0
        accepts = 0
        destroy_counts = {'random': 0, 'worst': 0, 'route': 0}

        # Main LNS loop
        for iteration in range(self.max_iterations):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > self.time_limit:
                logger.info(f"   ⏰ Time limit reached: {elapsed:.1f}s")
                break

            # Temperature (linear cooling)
            temperature = self.initial_temperature * (1 - iteration / self.max_iterations)

            # 1. Choose destroy operator randomly
            # Increased route removal probability (0.5) for better distance reduction
            destroy_op = np.random.choice(['random', 'worst', 'route'], p=[0.2, 0.3, 0.5])
            destroy_counts[destroy_op] += 1

            # 2. Destroy
            if destroy_op == 'random':
                removed, partial_routes = self._random_removal(current_routes, removal_rate=0.15)
            elif destroy_op == 'worst':
                removed, partial_routes = self._worst_removal(current_routes, num_remove=10)
            else:  # route
                removed, partial_routes = self._route_removal(current_routes)

            if not removed:
                continue

            # 3. Repair with Regret-2
            new_routes = self._regret_insertion(partial_routes, removed)

            # 4. Evaluate
            new_cost = self._calculate_cost(new_routes)
            new_violations = self._count_violations(new_routes)

            # Check feasibility requirement
            if require_feasible and new_violations > 0:
                continue  # Reject infeasible solutions

            # 5. Acceptance criteria (Simulated Annealing)
            delta = new_cost - current_cost

            if delta < 0:
                # Better solution - always accept
                current_routes = new_routes
                current_cost = new_cost
                accepts += 1

                # Update global best
                if new_cost < best_cost:
                    best_routes = copy.deepcopy(new_routes)
                    best_cost = new_cost
                    best_violations = new_violations
                    improvements += 1
                    logger.info(f"   Iter {iteration}: NEW BEST! {best_cost:.2f} km "
                              f"({best_violations} violations) via {destroy_op}")
            elif temperature > 0 and np.random.rand() < np.exp(-delta / temperature):
                # Worse solution - accept with probability
                current_routes = new_routes
                current_cost = new_cost
                accepts += 1

            # Progress logging
            if iteration % 200 == 0 and iteration > 0:
                logger.info(f"   Iter {iteration}: Best={best_cost:.2f} km, "
                          f"Current={current_cost:.2f} km, "
                          f"Improvements={improvements}, "
                          f"Accepts={accepts}/{iteration}")

        # Final summary
        elapsed = time.time() - start_time
        logger.info(f"✅ LNS completed in {elapsed:.1f}s")
        logger.info(f"   Initial: {self._calculate_cost(initial_routes):.2f} km")
        logger.info(f"   Final: {best_cost:.2f} km")
        logger.info(f"   Improvement: {self._calculate_cost(initial_routes) - best_cost:.2f} km")
        logger.info(f"   Iterations: {iteration+1}")
        logger.info(f"   Improvements found: {improvements}")
        logger.info(f"   Acceptance rate: {accepts/(iteration+1)*100:.1f}%")
        logger.info(f"   Destroy operator usage: {destroy_counts}")

        return best_routes

    def _random_removal(self,
                        routes: List[List[int]],
                        removal_rate: float = 0.15) -> Tuple[List[int], List[List[int]]]:
        """
        Random removal: Remove random customers.

        Args:
            routes: Current routes
            removal_rate: Fraction of customers to remove (default 0.15 = 15%)

        Returns:
            (removed_customers, partial_routes)
        """
        # Count total customers
        all_customers = [c for route in routes for c in route if c != 0]
        num_to_remove = max(1, int(len(all_customers) * removal_rate))

        # Select random customers to remove
        removed = np.random.choice(all_customers, size=min(num_to_remove, len(all_customers)), replace=False).tolist()

        # Create partial routes (remove selected customers)
        partial_routes = []
        for route in routes:
            new_route = [c for c in route if c not in removed]
            if len(new_route) >= 2:  # Keep routes with at least depot-depot
                partial_routes.append(new_route)

        return removed, partial_routes

    def _worst_removal(self,
                       routes: List[List[int]],
                       num_remove: int = 10) -> Tuple[List[int], List[List[int]]]:
        """
        Worst removal: Remove most costly customers.

        Costly = customer that increases route cost the most.

        Args:
            routes: Current routes
            num_remove: Number of customers to remove (default 10)

        Returns:
            (removed_customers, partial_routes)
        """
        # Calculate removal cost for each customer
        customer_costs = []

        for route_idx, route in enumerate(routes):
            for pos in range(1, len(route) - 1):  # Skip depots
                customer_id = route[pos]

                # Cost with customer
                cost_with = self.evaluator._get_distance(route[pos-1], customer_id) + \
                           self.evaluator._get_distance(customer_id, route[pos+1])

                # Cost without customer (direct)
                cost_without = self.evaluator._get_distance(route[pos-1], route[pos+1])

                # Savings from removing this customer
                savings = cost_with - cost_without

                customer_costs.append((customer_id, savings))

        # Sort by cost (highest first = worst customers)
        customer_costs.sort(key=lambda x: x[1], reverse=True)

        # Remove top N worst customers
        removed = [c[0] for c in customer_costs[:num_remove]]

        # Create partial routes
        partial_routes = []
        for route in routes:
            new_route = [c for c in route if c not in removed]
            if len(new_route) >= 2:
                partial_routes.append(new_route)

        return removed, partial_routes

    def _route_removal(self, routes: List[List[int]]) -> Tuple[List[int], List[List[int]]]:
        """
        Route removal: Remove ENTIRE route (THE SECRET WEAPON!)

        This is often more effective than removing random customers because:
        - Forces redistribution across other routes
        - Can discover better route structures
        - Escapes local optima more effectively

        Args:
            routes: Current routes

        Returns:
            (removed_customers, partial_routes)
        """
        # Don't remove if only 1 route left
        valid_routes = [r for r in routes if len(r) > 2]  # More than just depot-depot
        if len(valid_routes) <= 1:
            # Fall back to random removal
            return self._random_removal(routes, 0.15)

        # Choose random route to remove
        route_to_remove = np.random.choice(len(valid_routes))
        selected_route = valid_routes[route_to_remove]

        # Extract all customers from this route (exclude depots)
        removed = [c for c in selected_route if c != 0]

        # Create partial routes (all routes except the removed one)
        partial_routes = []
        for route in routes:
            if route != selected_route:
                partial_routes.append(route.copy())

        logger.debug(f"   Route removal: Removed {len(removed)} customers from 1 route")

        return removed, partial_routes

    def _regret_insertion(self,
                          partial_routes: List[List[int]],
                          unassigned: List[int]) -> List[List[int]]:
        """
        Regret-2 insertion: Insert customers based on regret value.

        Regret = Cost2 - Cost1 (difference between best and 2nd best insertion)
        High regret = "painful" to delay insertion of this customer

        Algorithm:
            1. For each unassigned customer:
                - Find best insertion position (minimum cost increase)
                - Find 2nd best insertion position
                - Calculate regret = Cost2 - Cost1
            2. Insert customer with HIGHEST regret first
            3. Repeat until all customers inserted

        Args:
            partial_routes: Routes with some customers removed
            unassigned: Customers to reinsert

        Returns:
            Complete routes with all customers inserted
        """
        routes = [route.copy() for route in partial_routes]
        remaining = unassigned.copy()

        while remaining:
            # Calculate regret for each remaining customer
            regrets = []

            for customer_id in remaining:
                # Find best and 2nd best insertion
                insertion_costs = []

                for route_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):  # Try all positions
                        # Calculate insertion cost
                        if pos == 0 or pos >= len(route):
                            continue

                        cost_before = self.evaluator._get_distance(route[pos-1], route[pos])
                        cost_after = (self.evaluator._get_distance(route[pos-1], customer_id) +
                                     self.evaluator._get_distance(customer_id, route[pos]))
                        cost_increase = cost_after - cost_before

                        insertion_costs.append((cost_increase, route_idx, pos))

                # Sort by cost
                insertion_costs.sort(key=lambda x: x[0])

                if len(insertion_costs) >= 2:
                    best_cost = insertion_costs[0][0]
                    second_cost = insertion_costs[1][0]
                    regret = second_cost - best_cost

                    regrets.append((
                        regret,
                        customer_id,
                        insertion_costs[0][1],  # best route
                        insertion_costs[0][2]   # best position
                    ))
                elif len(insertion_costs) == 1:
                    # Only one insertion option - high regret by default
                    regrets.append((
                        float('inf'),
                        customer_id,
                        insertion_costs[0][1],
                        insertion_costs[0][2]
                    ))

            if not regrets:
                # No valid insertions - create new route
                if remaining:
                    routes.append([0] + remaining + [0])
                    remaining = []
                break

            # Sort by regret (highest first)
            regrets.sort(key=lambda x: x[0], reverse=True)

            # Insert customer with highest regret
            _, customer_id, route_idx, pos = regrets[0]
            routes[route_idx].insert(pos, customer_id)
            remaining.remove(customer_id)

        return routes

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total distance of routes."""
        return sum(
            sum(self.evaluator._get_distance(route[i], route[i+1])
                for i in range(len(route)-1))
            for route in routes if route and len(route) > 1
        )

    def _count_violations(self, routes: List[List[int]]) -> int:
        """Count time window violations."""
        violations = 0
        for route in routes:
            if not route or len(route) < 3:
                continue
            forward = self.evaluator.compute_forward_sequence(route)
            for i in range(1, len(route) - 1):
                if forward[i].TW_E > forward[i].TW_L:
                    violations += 1
        return violations
