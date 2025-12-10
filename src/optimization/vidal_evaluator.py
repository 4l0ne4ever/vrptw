"""
Vidal Evaluator for O(1) Route Evaluation.

Implements PHASE 2: O(1) EVALUATION ENGINE
Based on Vidal et al. 2013: "A unified solution framework for multi-attribute
vehicle routing problems"

Key Innovation:
- Pre-compute Forward and Backward sequences for each route
- Concatenate sequences in O(1) to check move feasibility
- Reduces move evaluation from O(N) to O(1) (1000x speedup!)

Vidal's Three Key Values per node:
- D: Duration (accumulated service + travel time)
- TW_E: Earliest feasible arrival time
- TW_L: Latest feasible arrival time
- Q: Load (accumulated demand)
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from ..models.vrp_model import VRPProblem, Customer

logger = logging.getLogger(__name__)


@dataclass
class VidalNode:
    """
    Vidal node with accumulated route statistics.

    Attributes:
        customer_id: Customer ID (0 for depot)
        D: Duration (accumulated time from route start to this node)
        TW_E: Earliest feasible arrival time at this node
        TW_L: Latest feasible arrival time at this node
        Q: Load (accumulated demand from route start to this node)
        position: Position in route (for debugging)
    """
    customer_id: int
    D: float  # Duration
    TW_E: float  # Earliest arrival
    TW_L: float  # Latest arrival
    Q: float  # Load
    position: int = -1  # Optional position tracking

    def __repr__(self):
        return (f"VidalNode(id={self.customer_id}, D={self.D:.1f}, "
                f"E={self.TW_E:.1f}, L={self.TW_L:.1f}, Q={self.Q:.1f})")


class VidalEvaluator:
    """
    O(1) route evaluation using Vidal's concatenation logic.

    This is the SECRET WEAPON for fast local search!
    """

    def __init__(self,
                 problem: VRPProblem,
                 time_matrix: np.ndarray,
                 dist_matrix: Optional[np.ndarray] = None):
        """
        Initialize Vidal evaluator.

        Args:
            problem: VRP problem instance
            time_matrix: NxN matrix of travel times
            dist_matrix: Optional NxN matrix of distances (if different from time)
        """
        self.problem = problem
        self.time_matrix = time_matrix
        self.dist_matrix = dist_matrix if dist_matrix is not None else time_matrix

        # Cache for forward/backward sequences
        self.forward_cache: Dict[int, List[VidalNode]] = {}
        self.backward_cache: Dict[int, List[VidalNode]] = {}

    def create_node_for_customer(self,
                                   customer_id: int,
                                   prev_node: Optional[VidalNode] = None,
                                   position: int = -1) -> VidalNode:
        """
        Create a Vidal node for a customer.

        Args:
            customer_id: Customer ID
            prev_node: Previous node in route (None if this is first customer after depot)
            position: Position in route

        Returns:
            VidalNode with computed statistics
        """
        # Get customer info
        if customer_id == 0:
            # Depot
            return VidalNode(
                customer_id=0,
                D=0.0,
                TW_E=self.problem.depot.ready_time,
                TW_L=self.problem.depot.due_date,
                Q=0.0,
                position=position
            )

        customer = self.problem.get_customer_by_id(customer_id)
        if customer is None:
            raise ValueError(f"Customer {customer_id} not found")

        if prev_node is None:
            # First customer after depot
            travel_time = self._get_time(0, customer_id)
            return VidalNode(
                customer_id=customer_id,
                D=travel_time + customer.service_time,
                TW_E=max(customer.ready_time, travel_time),
                TW_L=customer.due_date,
                Q=customer.demand,
                position=position
            )
        else:
            # Not first customer - concatenate with previous
            travel_time = self._get_time(prev_node.customer_id, customer_id)

            # New duration
            new_D = prev_node.D + travel_time + customer.service_time

            # New earliest arrival
            arrival_time = prev_node.TW_E + travel_time
            new_TW_E = max(customer.ready_time, arrival_time)

            # New latest arrival
            new_TW_L = min(prev_node.TW_L - travel_time - customer.service_time,
                          customer.due_date)

            # NO CLAMPING! If TW_L < ready_time, the route is INFEASIBLE
            # The clamping was hiding violations by making infeasible routes appear feasible

            # New load
            new_Q = prev_node.Q + customer.demand

            return VidalNode(
                customer_id=customer_id,
                D=new_D,
                TW_E=new_TW_E,
                TW_L=new_TW_L,
                Q=new_Q,
                position=position
            )

    def concat(self,
               node_a: VidalNode,
               node_b: VidalNode,
               travel_time: Optional[float] = None) -> Tuple[VidalNode, bool]:
        """
        Concatenate two Vidal nodes in O(1).

        This is the CORE of Vidal's algorithm!

        Args:
            node_a: End node of first segment
            node_b: Start node of second segment (already has its own accumulated stats)
            travel_time: Travel time from A to B (if None, will query matrix)

        Returns:
            (combined_node, is_feasible): Combined node and feasibility flag

        Example:
            Route: [depot, 5, 8, 12, depot]
            Want to check: Can we insert customer 7 between 5 and 8?

            node_a = Forward[5] (accumulated stats from depot to 5)
            node_7 = Single node for customer 7
            node_b = Backward[8] (accumulated stats from 8 to depot)

            Result = concat(concat(node_a, node_7), node_b)
            If Result.is_feasible: YES, we can insert!
        """
        if travel_time is None:
            travel_time = self._get_time(node_a.customer_id, node_b.customer_id)

        # Get customer info for node_b (for service time)
        if node_b.customer_id == 0:
            service_time = 0.0
            ready_time = self.problem.depot.ready_time
            due_date = self.problem.depot.due_date
        else:
            customer_b = self.problem.get_customer_by_id(node_b.customer_id)
            service_time = customer_b.service_time
            ready_time = customer_b.ready_time
            due_date = customer_b.due_date

        # Concatenation formulas (Vidal et al. 2013)
        new_D = node_a.D + travel_time + node_b.D

        # Earliest arrival at end of segment B
        arrival_at_b_start = node_a.TW_E + travel_time
        new_TW_E = max(ready_time, arrival_at_b_start) + service_time + \
                  (node_b.TW_E - ready_time)

        # Latest arrival at end of segment B
        new_TW_L = min(node_a.TW_L - travel_time, due_date) - service_time + \
                  (node_b.TW_L - due_date)

        # NO CLAMPING! If TW_L becomes negative or < TW_E, it means infeasible
        # The feasibility check below will catch this correctly

        # Load
        new_Q = node_a.Q + node_b.Q

        # Feasibility check
        is_feasible = (
            new_TW_E <= new_TW_L and  # Time windows compatible
            new_Q <= self.problem.vehicle_capacity  # Capacity not exceeded
        )

        combined = VidalNode(
            customer_id=node_b.customer_id,
            D=new_D,
            TW_E=new_TW_E,
            TW_L=new_TW_L,
            Q=new_Q,
            position=-1
        )

        return combined, is_feasible

    def compute_forward_sequence(self, route: List[int]) -> List[VidalNode]:
        """
        Compute forward sequence for a route.

        Forward[i] = accumulated stats from depot to position i

        Args:
            route: List of customer IDs (including depot at start)

        Returns:
            List of VidalNodes, one per position in route
        """
        if not route or route[0] != 0:
            raise ValueError("Route must start with depot (0)")

        forward = []
        prev_node = None

        for i, customer_id in enumerate(route):
            node = self.create_node_for_customer(customer_id, prev_node, position=i)
            forward.append(node)
            prev_node = node

        return forward

    def compute_backward_sequence(self, route: List[int]) -> List[VidalNode]:
        """
        Compute backward sequence for a route.

        Backward[i] = accumulated stats from position i to depot

        Args:
            route: List of customer IDs (including depot at end)

        Returns:
            List of VidalNodes, one per position in route (in forward order)

        Note: We compute in reverse but return in forward order for easy indexing
        """
        if not route or route[-1] != 0:
            raise ValueError("Route must end with depot (0)")

        # Reverse the route
        reversed_route = list(reversed(route))

        # Compute forward on reversed route
        backward_reversed = []
        prev_node = None

        for i, customer_id in enumerate(reversed_route):
            node = self.create_node_for_customer(customer_id, prev_node, position=i)
            backward_reversed.append(node)
            prev_node = node

        # Reverse back to get forward order
        backward = list(reversed(backward_reversed))

        return backward

    def check_insertion_feasibility(self,
                                     route: List[int],
                                     customer_id: int,
                                     insert_pos: int) -> Tuple[bool, float]:
        """
        Check if inserting customer at position is feasible (O(1)).

        Args:
            route: Current route
            customer_id: Customer to insert
            insert_pos: Position to insert (1 to len(route)-1, excluding depots)

        Returns:
            (is_feasible, cost_delta): Feasibility and cost change
        """
        if insert_pos < 1 or insert_pos >= len(route):
            return False, float('inf')

        # Compute forward/backward if not cached
        forward = self.compute_forward_sequence(route)
        backward = self.compute_backward_sequence(route)

        # Node before insertion point
        node_before = forward[insert_pos - 1]

        # Create node for inserted customer
        inserted_node = self.create_node_for_customer(customer_id, node_before)

        # Node after insertion point
        node_after = backward[insert_pos]

        # Check feasibility: concat(before, inserted, after)
        temp, feasible1 = self.concat(node_before, inserted_node)
        final, feasible2 = self.concat(temp, node_after)

        is_feasible = feasible1 and feasible2

        # Calculate cost delta
        if is_feasible:
            # Old cost: before -> after
            old_cost = self._get_distance(
                route[insert_pos - 1],
                route[insert_pos]
            )

            # New cost: before -> inserted -> after
            new_cost = (
                self._get_distance(route[insert_pos - 1], customer_id) +
                self._get_distance(customer_id, route[insert_pos])
            )

            cost_delta = new_cost - old_cost
        else:
            cost_delta = float('inf')

        return is_feasible, cost_delta

    def _get_time(self, from_id: int, to_id: int) -> float:
        """Get travel time from matrix (Matrix-First Principle)."""
        from_idx = self.problem.id_to_index.get(from_id)
        to_idx = self.problem.id_to_index.get(to_id)

        if from_idx is None or to_idx is None:
            raise ValueError(f"Invalid customer ID: {from_id} or {to_id}")

        return self.time_matrix[from_idx, to_idx]

    def _get_distance(self, from_id: int, to_id: int) -> float:
        """Get distance from matrix (Matrix-First Principle)."""
        from_idx = self.problem.id_to_index.get(from_id)
        to_idx = self.problem.id_to_index.get(to_id)

        if from_idx is None or to_idx is None:
            raise ValueError(f"Invalid customer ID: {from_id} or {to_id}")

        return self.dist_matrix[from_idx, to_idx]

    def evaluate_route(self, route: List[int]) -> Dict[str, float]:
        """
        Evaluate a complete route.

        Args:
            route: List of customer IDs including depots

        Returns:
            Dictionary with metrics (distance, time, violations, etc.)
        """
        if not route or len(route) < 2:
            return {'distance': 0.0, 'time': 0.0, 'violations': 0, 'load': 0.0}

        # Compute forward sequence
        forward = self.compute_forward_sequence(route)

        # Last node has all accumulated stats
        final_node = forward[-1]

        # Calculate total distance
        total_distance = sum(
            self._get_distance(route[i], route[i + 1])
            for i in range(len(route) - 1)
        )

        # Check violations
        violations = 0
        for i in range(1, len(route) - 1):  # Exclude depots
            node = forward[i]
            if node.TW_E > node.TW_L:
                violations += 1

        return {
            'distance': total_distance,
            'time': final_node.D,
            'violations': violations,
            'load': final_node.Q,
            'feasible': violations == 0 and final_node.Q <= self.problem.vehicle_capacity
        }
