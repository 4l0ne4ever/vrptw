"""
Neighbor Lists Builder for VRP Optimization.

Implements PHASE 1.2: Neighbor Lists (CRITICAL FOR SPEED)
- Pre-computes K closest neighbors for each customer based on time/distance
- Reduces search space from O(N) to O(K) where K=40
- Achieves 96% search space reduction (40/1000 = 4%)
- Used in Strong Repair and LNS to limit candidate moves
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from ..models.vrp_model import VRPProblem

logger = logging.getLogger(__name__)


class NeighborListBuilder:
    """
    Builds neighbor lists for efficient local search.

    Key Principle: NEVER iterate through all N customers.
    Instead, only search within the K closest neighbors.

    This reduces:
    - Relocate search: O(N²) → O(K·N)
    - Swap search: O(N²) → O(K·N)
    - Overall speedup: 96% reduction when K=40, N=1000
    """

    def __init__(self,
                 time_matrix: np.ndarray,
                 problem: VRPProblem,
                 k: int = 40,
                 include_depot: bool = True):
        """
        Initialize neighbor list builder.

        Args:
            time_matrix: NxN matrix of travel times (from MatrixPreprocessor)
            problem: VRP problem instance for ID mapping
            k: Number of closest neighbors to keep (default 40)
            include_depot: Whether to include depot in neighbor lists (default True)
        """
        self.time_matrix = time_matrix
        self.problem = problem
        self.k = k
        self.include_depot = include_depot
        self.neighbor_lists = None

    def build_neighbor_lists(self) -> Dict[int, List[int]]:
        """
        Build neighbor lists for all customers.

        Returns:
            Dictionary mapping customer_id → [list of K closest neighbor IDs]

        Process:
            1. For each customer i (and depot if included):
                a. Extract row i from time_matrix (times from i to all others)
                b. Sort indices by time (ascending order)
                c. Take top K indices (excluding self)
                d. Convert indices back to customer IDs
                e. Store in neighbor_lists[i]

        Example:
            neighbor_lists[5] = [12, 7, 23, 15, ...] (40 closest customers to customer 5)
        """
        logger.info("🔧 Neighbor List Builder: Building neighbor lists...")

        N = len(self.time_matrix)
        self.neighbor_lists = {}

        # Get all customer IDs (including depot if specified)
        all_ids = [0] if self.include_depot else []  # Depot ID = 0
        all_ids.extend([c.id for c in self.problem.customers])

        logger.info(f"   Matrix size: {N}x{N}")
        logger.info(f"   Number of nodes: {len(all_ids)} ({'with' if self.include_depot else 'without'} depot)")
        logger.info(f"   K (neighbors per node): {self.k}")

        # Build neighbor list for each customer
        for customer_id in all_ids:
            # Get matrix index for this customer
            idx = self.problem.id_to_index.get(customer_id)
            if idx is None:
                logger.warning(f"   ⚠️  Customer ID {customer_id} not in id_to_index mapping, skipping")
                continue

            # Extract row i (times from i to all others)
            times_from_i = self.time_matrix[idx, :]

            # Sort indices by time (ascending)
            sorted_indices = np.argsort(times_from_i)

            # Take top K+1 (because first one is self with time=0)
            # We'll skip self and take the next K
            top_k_indices = sorted_indices[1:self.k+1]  # Skip index 0 (self)

            # Convert matrix indices back to customer IDs
            neighbor_ids = []
            for matrix_idx in top_k_indices:
                # Find customer ID for this matrix index
                neighbor_id = self._get_id_from_index(matrix_idx)
                if neighbor_id is not None:
                    neighbor_ids.append(neighbor_id)

            # Store in neighbor lists
            self.neighbor_lists[customer_id] = neighbor_ids[:self.k]  # Ensure exactly K neighbors

            # Log first few for debugging
            if customer_id in [0, 2, 3]:  # Depot and first 2 customers
                logger.debug(f"   Customer {customer_id}: neighbors = {neighbor_ids[:5]}... ({len(neighbor_ids)} total)")

        logger.info(f"✅ Neighbor List Builder: Built {len(self.neighbor_lists)} neighbor lists")
        logger.info(f"   Average neighbors per node: {np.mean([len(v) for v in self.neighbor_lists.values()]):.1f}")
        logger.info(f"   Search space reduction: {(1 - self.k/N)*100:.1f}% (from {N} to {self.k} per node)")

        return self.neighbor_lists

    def _get_id_from_index(self, matrix_idx: int) -> Optional[int]:
        """
        Convert matrix index back to customer ID.

        Args:
            matrix_idx: Index in the matrix (0 to N-1)

        Returns:
            Customer ID, or None if not found
        """
        # Reverse lookup in id_to_index mapping
        for customer_id, idx in self.problem.id_to_index.items():
            if idx == matrix_idx:
                return customer_id
        return None

    def get_neighbors(self, customer_id: int) -> List[int]:
        """
        Get neighbor list for a customer.

        Args:
            customer_id: Customer ID to get neighbors for

        Returns:
            List of K closest neighbor IDs

        Raises:
            ValueError: If neighbor lists not built yet
        """
        if self.neighbor_lists is None:
            raise ValueError("Neighbor lists not built yet. Call build_neighbor_lists() first")

        if customer_id not in self.neighbor_lists:
            logger.warning(f"⚠️  Customer {customer_id} not in neighbor lists, returning empty list")
            return []

        return self.neighbor_lists[customer_id]

    def get_all_neighbor_lists(self) -> Dict[int, List[int]]:
        """
        Get all neighbor lists.

        Returns:
            Dictionary mapping customer_id → [neighbor IDs]

        Raises:
            ValueError: If neighbor lists not built yet
        """
        if self.neighbor_lists is None:
            raise ValueError("Neighbor lists not built yet. Call build_neighbor_lists() first")

        return self.neighbor_lists

    def validate_neighbor_lists(self) -> bool:
        """
        Validate that neighbor lists are correctly built.

        Checks:
            1. All customers have neighbor lists
            2. Each list has exactly K neighbors (or less if N < K)
            3. No customer is in its own neighbor list
            4. All neighbor IDs are valid

        Returns:
            True if validation passes, False otherwise
        """
        if self.neighbor_lists is None:
            logger.error("❌ Validation failed: Neighbor lists not built")
            return False

        logger.info("🔍 Validating neighbor lists...")

        all_customer_ids = [c.id for c in self.problem.customers]
        if self.include_depot:
            all_customer_ids = [0] + all_customer_ids

        issues = []

        # Check 1: All customers have neighbor lists
        for customer_id in all_customer_ids:
            if customer_id not in self.neighbor_lists:
                issues.append(f"Customer {customer_id} missing from neighbor lists")

        # Check 2: Each list has correct size
        expected_k = min(self.k, len(all_customer_ids) - 1)  # Can't have more neighbors than total customers
        for customer_id, neighbors in self.neighbor_lists.items():
            if len(neighbors) != expected_k:
                issues.append(f"Customer {customer_id} has {len(neighbors)} neighbors (expected {expected_k})")

        # Check 3: No customer is in its own neighbor list
        for customer_id, neighbors in self.neighbor_lists.items():
            if customer_id in neighbors:
                issues.append(f"Customer {customer_id} appears in its own neighbor list")

        # Check 4: All neighbor IDs are valid
        valid_ids = set(all_customer_ids)
        for customer_id, neighbors in self.neighbor_lists.items():
            invalid_neighbors = [n for n in neighbors if n not in valid_ids]
            if invalid_neighbors:
                issues.append(f"Customer {customer_id} has invalid neighbors: {invalid_neighbors}")

        if issues:
            logger.error(f"❌ Validation failed with {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                logger.error(f"   - {issue}")
            return False

        logger.info("✅ Validation passed: All neighbor lists are correct")
        return True

    def update_k(self, new_k: int) -> Dict[int, List[int]]:
        """
        Update K and rebuild neighbor lists.

        Args:
            new_k: New number of neighbors to keep

        Returns:
            Updated neighbor lists
        """
        logger.info(f"🔧 Updating K from {self.k} to {new_k}")
        self.k = new_k
        return self.build_neighbor_lists()

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about neighbor lists.

        Returns:
            Dictionary with statistics
        """
        if self.neighbor_lists is None:
            raise ValueError("Neighbor lists not built yet")

        neighbor_counts = [len(v) for v in self.neighbor_lists.values()]

        return {
            'num_nodes': len(self.neighbor_lists),
            'k': self.k,
            'avg_neighbors': np.mean(neighbor_counts),
            'min_neighbors': np.min(neighbor_counts),
            'max_neighbors': np.max(neighbor_counts),
            'search_space_reduction_pct': (1 - self.k / len(self.time_matrix)) * 100
        }
