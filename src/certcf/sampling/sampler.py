"""
CertCF for Robust Counterfactual Generation.

This module implements the CertCF method described in the research defense:
Given a query sample x from class A, find the closest certified point in class B
by projecting onto the union of certified polytopes for class B.

Key advantages over gradient-based methods:
1. Guaranteed validity: counterfactuals lie within certified regions
2. Manifold adherence: constrained to stay near real data
3. Convexity: solving convex QPs ensures global optimality per polytope
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Optional

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class PolytopeAtlas:
    """
    A collection of certified polytopes for a single target class.

    This represents the "atlas" or map of the target class geometry,
    constructed offline using LiRPA bounds.

    Attributes
    ----------
    label : int
        The target class label.
    A_matrices : np.ndarray
        Array of constraint matrices, shape (n_polytopes, k-1, d).
    b_vectors : np.ndarray
        Array of bias vectors, shape (n_polytopes, k-1).
    centers : np.ndarray
        Array of polytope centers (original training samples), shape (n_polytopes, d).
    eps : float
        Perturbation radius defining the validity region of each polytope.
    """
    label: int
    A_matrices: np.ndarray  # (n_polytopes, k-1, d)
    b_vectors: np.ndarray   # (n_polytopes, k-1)
    centers: np.ndarray     # (n_polytopes, d)
    eps: float

    @property
    def n_polytopes(self) -> int:
        """Number of polytopes in the atlas."""
        return len(self.centers)

    @property
    def dimension(self) -> int:
        """Input space dimension."""
        return self.centers.shape[1]


class CounterfactualSampler:
    """
    CertCF counterfactual generator.

    This class implements the CertCF algorithm:
    1. Offline: Build polytope atlases for each target class using LiRPA
    2. Online: For a query point, find k-nearest polytopes and project onto each
    3. Return the projection with minimum distance

    The key insight is that finding a counterfactual becomes a geometric
    projection problem rather than a non-convex optimization problem.

    Attributes
    ----------
    atlases : dict[int, PolytopeAtlas]
        Dictionary mapping class labels to their polytope atlases.
    solver : str
        CVXPY solver to use for QP projection (default: 'ECOS').
    verbose : bool
        Whether to print progress and diagnostic information.
    """

    def __init__(
        self,
        solver: str = 'ECOS',
        verbose: bool = False,
        distance_norm: int | float = 2
    ):
        """
        Initialize the counterfactual sampler.

        Parameters
        ----------
        solver : str, optional
            CVXPY solver name (default: 'ECOS'). Options: 'ECOS', 'SCS', 'CLARABEL'.
        verbose : bool, optional
            Whether to print verbose output (default: False).
        distance_norm : int or float, optional
            Norm to use for the distance objective (default: 2).
            - 1: L1 norm (Manhattan distance, sparse changes, good for images)
            - 2: L2 norm (Euclidean distance, smooth changes)
            - np.inf: L∞ norm (Chebyshev distance, minimize max change)

        Raises
        ------
        ImportError
            If cvxpy is not installed.
        """
        if not CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy is required for counterfactual sampling. "
                "Install it with: pip install cvxpy"
            )

        self.atlases = {}
        self.solver = solver
        self.verbose = verbose
        self.distance_norm = distance_norm

    def build_atlas(
        self,
        label: int,
        bounds: dict,
        eps: float
    ) -> PolytopeAtlas:
        """
        Build a polytope atlas for a single target class.

        This is the "offline" phase: extract the LiRPA bounds for all samples
        in the target class and package them into a PolytopeAtlas.

        Parameters
        ----------
        label : int
            The target class label.
        bounds : dict
            Bounds dictionary for this class from PreimageApproximation.compute_all_bounds().
            Expected keys: 'lA', 'lbias', 'X'.
        eps : float
            Perturbation radius used for the bounds.

        Returns
        -------
        PolytopeAtlas
            The constructed atlas for the target class.
        """
        bd = bounds[label]

        atlas = PolytopeAtlas(
            label=label,
            A_matrices=bd['lA'],      # (n_samples, k-1, d)
            b_vectors=bd['lbias'],    # (n_samples, k-1)
            centers=bd['X'],          # (n_samples, d)
            eps=eps
        )

        if self.verbose:
            print(f"Built atlas for class {label}: {atlas.n_polytopes} polytopes, "
                  f"dim={atlas.dimension}, eps={eps}")

        return atlas

    def build_all_atlases(
        self,
        all_bounds: dict,
        eps: float
    ) -> None:
        """
        Build polytope atlases for all classes.

        Parameters
        ----------
        all_bounds : dict
            Dictionary mapping labels to their bounds from PreimageApproximation.
        eps : float
            Perturbation radius.
        """
        for label in all_bounds.keys():
            self.atlases[label] = self.build_atlas(label, all_bounds, eps)

        if self.verbose:
            print(f"Built {len(self.atlases)} atlases")

    def _project_to_polytope(
        self,
        x_query: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        center: np.ndarray,
        eps: float
    ) -> tuple[Optional[np.ndarray], float]:
        """
        Project a query point onto a single certified polytope.

        Solves the convex optimization program:
            minimize    ||z - x_query||_p
            subject to  A z + b >= 0            (LiRPA certification constraints)
                        center - eps <= z <= center + eps  (validity box)

        where p is determined by self.distance_norm.

        Parameters
        ----------
        x_query : np.ndarray
            The query point to project, shape (d,).
        A : np.ndarray
            Constraint matrix for this polytope, shape (k-1, d).
        b : np.ndarray
            Bias vector for this polytope, shape (k-1,).
        center : np.ndarray
            Center of the polytope (original training point), shape (d,).
        eps : float
            Perturbation radius defining the validity box.

        Returns
        -------
        z_star : np.ndarray or None
            The projected point if successful, None if infeasible.
        distance : float
            Distance from x_query to z_star using the specified norm (inf if infeasible).
        """
        d = len(x_query)
        z = cp.Variable(d)

        # Objective: minimize distance in specified norm
        diff = z - x_query
        if self.distance_norm == 1:
            # L1 norm: sum of absolute values (encourages sparsity)
            objective = cp.Minimize(cp.sum(cp.abs(diff)))
        elif self.distance_norm == 2:
            # L2 norm: Euclidean distance (smooth changes)
            objective = cp.Minimize(cp.sum_squares(diff))
        elif self.distance_norm == np.inf:
            # L∞ norm: minimize maximum change
            objective = cp.Minimize(cp.norm(diff, np.inf))
        else:
            # General Lp norm
            objective = cp.Minimize(cp.norm(diff, self.distance_norm))


        # Constraints
        constraints = [
            A @ z + b >= 0,           # LiRPA certification constraints
            z >= center - eps,        # Box constraint (lower)
            z <= center + eps         # Box constraint (upper)
        ]

        # Solve
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self.solver, verbose=False)
        except Exception as e:
            if self.verbose:
                print(f"Solver failed: {e}")
            return None, np.inf

        # Check if solved successfully
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            return None, np.inf

        z_star = z.value
        if z_star is None:
            return None, np.inf

        # Compute distance using the same norm
        if self.distance_norm == 1:
            distance = np.linalg.norm(z_star - x_query, ord=1)
        elif self.distance_norm == 2:
            distance = np.linalg.norm(z_star - x_query, ord=2)
        elif self.distance_norm == np.inf:
            distance = np.linalg.norm(z_star - x_query, ord=np.inf)
        else:
            distance = np.linalg.norm(z_star - x_query, ord=self.distance_norm)

        return z_star, distance

    def generate_counterfactual(
        self,
        x_query: np.ndarray,
        target_label: int,
        k_candidates: int = 5,
        return_all_candidates: bool = False
    ) -> dict:
        """
        Generate a counterfactual explanation for a query point.

        This is the "online" phase:
        1. Find k nearest polytope centers in the target class
        2. Project x_query onto each candidate polytope (solving a QP)
        3. Return the projection with minimum distance

        Parameters
        ----------
        x_query : np.ndarray
            The query point to generate a counterfactual for, shape (d,).
        target_label : int
            The desired target class.
        k_candidates : int, optional
            Number of nearest polytopes to try (default: 5).
        return_all_candidates : bool, optional
            If True, return all candidate projections in the result (default: False).

        Returns
        -------
        dict
            Dictionary containing:
            - 'counterfactual': The best counterfactual point (np.ndarray or None)
            - 'distance': Distance from x_query to counterfactual (float)
            - 'target_label': The target class (int)
            - 'success': Whether a valid counterfactual was found (bool)
            - 'n_candidates_tried': Number of polytopes attempted (int)
            - 'n_candidates_feasible': Number of feasible projections (int)
            - 'candidates': List of (point, distance) if return_all_candidates=True

        Raises
        ------
        ValueError
            If target_label has no atlas built.
        """
        if target_label not in self.atlases:
            raise ValueError(
                f"No atlas found for target class {target_label}. "
                f"Available classes: {list(self.atlases.keys())}"
            )

        atlas = self.atlases[target_label]

        # Step 1: Find k-nearest polytope centers
        distances_to_centers = cdist([x_query], atlas.centers, metric='euclidean')[0]
        k_actual = min(k_candidates, atlas.n_polytopes)
        nearest_indices = np.argsort(distances_to_centers)[:k_actual]

        if self.verbose:
            print(f"Selected {k_actual} nearest polytopes from {atlas.n_polytopes} total")

        # Step 2: Project onto each candidate polytope
        candidates = []
        for idx in nearest_indices:
            z_star, dist = self._project_to_polytope(
                x_query,
                atlas.A_matrices[idx],
                atlas.b_vectors[idx],
                atlas.centers[idx],
                atlas.eps
            )

            if z_star is not None:
                candidates.append((z_star, dist))

        # Step 3: Select the best projection
        if len(candidates) == 0:
            result = {
                'counterfactual': None,
                'distance': np.inf,
                'target_label': target_label,
                'success': False,
                'n_candidates_tried': k_actual,
                'n_candidates_feasible': 0,
            }
        else:
            # Sort by distance and take the closest
            candidates.sort(key=lambda x: x[1])
            best_cf, best_dist = candidates[0]

            result = {
                'counterfactual': best_cf,
                'distance': best_dist,
                'target_label': target_label,
                'success': True,
                'n_candidates_tried': k_actual,
                'n_candidates_feasible': len(candidates),
            }

        if return_all_candidates:
            result['candidates'] = candidates

        if self.verbose:
            status = "SUCCESS" if result['success'] else "FAILED"
            print(f"{status}: distance={result['distance']:.4f}, "
                  f"feasible={result['n_candidates_feasible']}/{result['n_candidates_tried']}")

        return result

    def generate_counterfactual_batch(
        self,
        X_query: np.ndarray,
        target_labels: np.ndarray | int,
        k_candidates: int = 5
    ) -> list[dict]:
        """
        Generate counterfactuals for a batch of query points.

        Parameters
        ----------
        X_query : np.ndarray
            Array of query points, shape (n_samples, d).
        target_labels : np.ndarray or int
            Target labels for each query (array of length n_samples) or
            a single label to use for all queries.
        k_candidates : int, optional
            Number of nearest polytopes to try (default: 5).

        Returns
        -------
        list[dict]
            List of result dictionaries, one per query point.
        """
        n_samples = X_query.shape[0]

        # Handle single target label for all queries
        if isinstance(target_labels, int):
            target_labels = np.full(n_samples, target_labels)

        results = []
        for i in range(n_samples):
            result = self.generate_counterfactual(
                X_query[i],
                target_labels[i],
                k_candidates=k_candidates
            )
            results.append(result)

        return results

    def verify_counterfactual(
        self,
        x_cf: np.ndarray,
        target_label: int,
        model: torch.nn.Module,
        device: torch.device | str = 'cpu'
    ) -> dict:
        """
        Verify that a counterfactual is correctly classified by the model.

        Parameters
        ----------
        x_cf : np.ndarray
            The counterfactual point to verify, shape (d,).
        target_label : int
            The expected target class.
        model : torch.nn.Module
            The classifier model.
        device : torch.device or str, optional
            Device to run the model on (default: 'cpu').

        Returns
        -------
        dict
            Dictionary containing:
            - 'predicted_label': Model's predicted class (int)
            - 'target_label': Expected target class (int)
            - 'correct': Whether prediction matches target (bool)
            - 'logits': Model output logits (np.ndarray)
            - 'probabilities': Softmax probabilities (np.ndarray)
        """
        model.eval()
        x_tensor = torch.tensor(x_cf, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x_tensor).cpu().numpy()[0]
            probs = np.exp(logits) / np.exp(logits).sum()
            predicted = np.argmax(logits)

        return {
            'predicted_label': int(predicted),
            'target_label': target_label,
            'correct': predicted == target_label,
            'logits': logits,
            'probabilities': probs,
        }
