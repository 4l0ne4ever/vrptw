"""
Matrix Preprocessor for VRP Optimization.

Implements PHASE 1.1: Universal Matrix Normalization
- Ensures all algorithms use pre-computed matrices (Matrix-First Principle)
- Handles asymmetric costs correctly (Asymmetry Principle)
- Normalizes diagonal values to 0
- Creates both distance and time matrices
"""

import numpy as np
import logging
from typing import Tuple, Optional
from ..models.vrp_model import VRPProblem

logger = logging.getLogger(__name__)


class MatrixPreprocessor:
    """
    Preprocesses distance and time matrices for optimization.

    Key Principles:
    1. Matrix-First: All cost queries must use matrix lookup (no formulas)
    2. Asymmetry: Assume cost(A->B) ≠ cost(B->A) for generalization to Hanoi
    3. Normalization: Ensure diagonal = 0 for consistency
    """

    def __init__(self, problem: VRPProblem):
        """
        Initialize matrix preprocessor.

        Args:
            problem: VRP problem instance with distance_matrix
        """
        self.problem = problem
        self.dist_matrix = None
        self.time_matrix = None
        self.is_symmetric = None

    def normalize_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize distance and time matrices.

        Returns:
            (dist_matrix, time_matrix): Normalized NxN matrices where N = num_customers + 1 (depot)

        Process:
            1. Copy raw distance matrix from problem
            2. Set diagonal to 0 (distance from node to itself)
            3. Create time matrix (for now, time == distance for Solomon)
            4. Validate matrices
            5. Check for asymmetry
        """
        logger.info(" Matrix Preprocessor: Starting normalization...")

        # Get raw distance matrix
        if self.problem.distance_matrix is None:
            raise ValueError("Distance matrix not available in problem")

        # STEP 1: Copy and normalize distance matrix
        self.dist_matrix = self.problem.distance_matrix.copy()

        N = len(self.dist_matrix)
        logger.info(f"   Matrix size: {N}x{N} (1 depot + {N-1} customers)")

        # STEP 2: Set diagonal to 0
        # IMPORTANT: This ensures distance[i][i] = 0 for all i
        np.fill_diagonal(self.dist_matrix, 0.0)

        # STEP 3: Create time matrix
        # For Solomon instances: time ≈ distance (Euclidean space)
        # For Hanoi instances: time = distance / speed (from traffic model)
        # For now, we use distance as time (can be enhanced later)
        self.time_matrix = self.dist_matrix.copy()

        # Add service times if needed (future enhancement)
        # for i in range(1, N):  # Skip depot
        #     customer = self.problem.get_customer_by_id(...)
        #     self.time_matrix[i, :] += customer.service_time

        # STEP 4: Validate matrices
        self._validate_matrices()

        # STEP 5: Check asymmetry
        self._check_asymmetry()

        logger.info(f" Matrix Preprocessor: Normalization complete")
        logger.info(f"   Distance matrix: shape={self.dist_matrix.shape}, dtype={self.dist_matrix.dtype}")
        logger.info(f"   Time matrix: shape={self.time_matrix.shape}, dtype={self.time_matrix.dtype}")
        logger.info(f"   Symmetry: {'Symmetric' if self.is_symmetric else 'Asymmetric (Hanoi mode)'}")

        return self.dist_matrix, self.time_matrix

    def _validate_matrices(self):
        """Validate matrix properties."""
        N = len(self.dist_matrix)

        # Check 1: Square matrices
        assert self.dist_matrix.shape == (N, N), "Distance matrix must be square"
        assert self.time_matrix.shape == (N, N), "Time matrix must be square"

        # Check 2: Diagonal is 0
        diagonal_dist = np.diag(self.dist_matrix)
        diagonal_time = np.diag(self.time_matrix)

        if not np.allclose(diagonal_dist, 0.0):
            logger.warning(f"  Distance matrix diagonal not all zeros: {diagonal_dist[:5]}...")
            # Fix it
            np.fill_diagonal(self.dist_matrix, 0.0)
            logger.info("   Fixed diagonal to 0")

        if not np.allclose(diagonal_time, 0.0):
            logger.warning(f"  Time matrix diagonal not all zeros: {diagonal_time[:5]}...")
            # Fix it
            np.fill_diagonal(self.time_matrix, 0.0)
            logger.info("   Fixed diagonal to 0")

        # Check 3: Non-negative values
        assert np.all(self.dist_matrix >= 0), "Distance matrix has negative values"
        assert np.all(self.time_matrix >= 0), "Time matrix has negative values"

        # Check 4: No NaN or Inf
        assert not np.any(np.isnan(self.dist_matrix)), "Distance matrix contains NaN"
        assert not np.any(np.isinf(self.dist_matrix)), "Distance matrix contains Inf"
        assert not np.any(np.isnan(self.time_matrix)), "Time matrix contains NaN"
        assert not np.any(np.isinf(self.time_matrix)), "Time matrix contains Inf"

        logger.info("    Matrix validation passed")

    def _check_asymmetry(self):
        """
        Check if matrix is symmetric or asymmetric.

        This is CRITICAL for handling Hanoi mode correctly.
        If asymmetric, we must re-query matrix when reversing segments.
        """
        # Check if distance matrix is symmetric
        is_symmetric_dist = np.allclose(self.dist_matrix, self.dist_matrix.T, rtol=1e-5)
        is_symmetric_time = np.allclose(self.time_matrix, self.time_matrix.T, rtol=1e-5)

        self.is_symmetric = is_symmetric_dist and is_symmetric_time

        if not self.is_symmetric:
            logger.warning("  ASYMMETRIC matrices detected (Hanoi mode)")
            logger.warning("   IMPORTANT: When reversing segments (2-opt), must re-query matrix!")
            logger.warning("   Never assume cost(A->B) == cost(B->A)")

            # Log some examples of asymmetry
            N = min(5, len(self.dist_matrix))
            for i in range(N):
                for j in range(i+1, N):
                    forward = self.dist_matrix[i, j]
                    backward = self.dist_matrix[j, i]
                    if abs(forward - backward) > 0.01:
                        logger.info(f"   Example: dist[{i}->{j}]={forward:.2f}, dist[{j}->{i}]={backward:.2f}")
                        break

    def get_distance(self, from_id: int, to_id: int) -> float:
        """
        Get distance from matrix (Matrix-First Principle).

        Args:
            from_id: Starting customer/depot ID
            to_id: Ending customer/depot ID

        Returns:
            Distance from from_id to to_id

        IMPORTANT: This method MUST be used instead of calculating Euclidean distance!
        """
        if self.dist_matrix is None:
            raise ValueError("Matrices not normalized yet. Call normalize_matrices() first")

        # Convert IDs to matrix indices
        from_idx = self.problem.id_to_index.get(from_id)
        to_idx = self.problem.id_to_index.get(to_id)

        if from_idx is None or to_idx is None:
            raise ValueError(f"Invalid customer ID: {from_id} or {to_id}")

        return self.dist_matrix[from_idx, to_idx]

    def get_time(self, from_id: int, to_id: int) -> float:
        """
        Get travel time from matrix (Matrix-First Principle).

        Args:
            from_id: Starting customer/depot ID
            to_id: Ending customer/depot ID

        Returns:
            Travel time from from_id to to_id
        """
        if self.time_matrix is None:
            raise ValueError("Matrices not normalized yet. Call normalize_matrices() first")

        # Convert IDs to matrix indices
        from_idx = self.problem.id_to_index.get(from_id)
        to_idx = self.problem.id_to_index.get(to_id)

        if from_idx is None or to_idx is None:
            raise ValueError(f"Invalid customer ID: {from_id} or {to_id}")

        return self.time_matrix[from_idx, to_idx]

    def get_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get normalized matrices.

        Returns:
            (dist_matrix, time_matrix): Normalized matrices
        """
        if self.dist_matrix is None or self.time_matrix is None:
            raise ValueError("Matrices not normalized yet. Call normalize_matrices() first")

        return self.dist_matrix, self.time_matrix

    def is_matrix_symmetric(self) -> bool:
        """
        Check if matrices are symmetric.

        Returns:
            True if symmetric (Solomon mode), False if asymmetric (Hanoi mode)
        """
        if self.is_symmetric is None:
            raise ValueError("Matrices not normalized yet. Call normalize_matrices() first")

        return self.is_symmetric
