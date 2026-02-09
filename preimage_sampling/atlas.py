"""
CertifiedAtlas: High-level API for certified counterfactual generation.

This module provides a simple, user-friendly interface that combines:
- LiRPA bound propagation for preimage certification
- Polytope construction and union operations
- BVH spatial indexing for efficient queries
- QP-based counterfactual projection

Example usage:
    >>> from preimage_sampling import CertifiedAtlas
    >>> atlas = CertifiedAtlas(model, dataset, device)
    >>> atlas.build(eps=0.1, norm=2)
    >>> cf = atlas.find_counterfactual(x_query, target_class=3)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

from .certification.lirpa import PreimageApproximation
from .geometry.polytopes import ball_box_constraints, make_polygon
from .geometry.operations import build_class_union, refine_unions_by_priority
from .indexing.bvh import BVHIndex


@dataclass
class CounterfactualResult:
    """
    Result of a counterfactual search.

    Attributes
    ----------
    x_cf : np.ndarray or None
        The counterfactual point, or None if not found.
    distance : float
        Distance from query to counterfactual (inf if not found).
    target_class : int
        The target class for the counterfactual.
    anchor_idx : int or None
        Index of the anchor polytope containing x_cf.
    n_qp_solved : int
        Number of QP projections computed.
    success : bool
        Whether a valid counterfactual was found.
    """
    x_cf: Optional[np.ndarray]
    distance: float
    target_class: int
    anchor_idx: Optional[int]
    n_qp_solved: int
    success: bool


class CertifiedAtlas:
    """
    High-level API for certified counterfactual generation.

    This class provides a simple interface for building certified preimage
    approximations and generating counterfactuals. It handles all the
    complexity of LiRPA bounds, polytope construction, and spatial indexing.

    Parameters
    ----------
    model : nn.Module
        The neural network classifier.
    dataset : torch.utils.data.Dataset
        Dataset containing (X, y) pairs. Can be a TensorDataset or list of tuples.
    device : torch.device or str
        Device for computation ('cuda' or 'cpu').
    cnn : bool, optional
        Whether the model is a CNN (default: False).

    Attributes
    ----------
    n_classes : int
        Number of classes in the dataset.
    bounds : dict or None
        LiRPA bounds for each class (after calling build()).
    bvh_indices : dict or None
        BVH spatial indices for each class (after calling build()).
    eps : float or None
        Perturbation radius used for building (after calling build()).
    norm : int or None
        Lp norm used for building (after calling build()).
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        device: Union[torch.device, str],
        cnn: bool = False,
        solver_maxiter: int = 500,
        solver_tol: float = 1e-9
    ):
        """
        Initialize the CertifiedAtlas.

        Parameters
        ----------
        model : nn.Module
            The neural network classifier.
        dataset : Dataset
            Dataset containing (X, y) pairs.
        device : torch.device or str
            Device for computation.
        cnn : bool, optional
            Whether the model is a CNN (default: False).
        solver_maxiter : int, optional
            Maximum iterations for the QP solver (default: 500).
        solver_tol : float, optional
            Tolerance for the QP solver (default: 1e-9).
        """
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cnn = cnn

        # Solver parameters (can be overridden in find_counterfactual)
        self.solver_maxiter = solver_maxiter
        self.solver_tol = solver_tol

        # Initialize preimage approximation handler
        self._preimage = PreimageApproximation(model, dataset, self.device, cnn=cnn)
        self.n_classes = self._preimage.n_classes

        # These are populated by build()
        self.bounds: Optional[Dict] = None
        self.bvh_indices: Optional[Dict[int, BVHIndex]] = None
        self.eps: Optional[float] = None
        self.norm: Optional[int] = None

        # Optional: Shapely polygon unions (only for 2D visualization)
        self._class_unions: Optional[Dict] = None

    def build(
        self,
        eps: float = 0.1,
        norm: int = 2,
        max_samples_per_class: Optional[int] = None,
        batch_size: Optional[int] = None,
        build_unions: bool = False,
        verbose: bool = True
    ) -> 'CertifiedAtlas':
        """
        Build the certified atlas by computing LiRPA bounds and spatial indices.

        This is the "offline" phase that should be run once before generating
        counterfactuals.

        Parameters
        ----------
        eps : float, optional
            Perturbation radius for Lp ball (default: 0.1).
        norm : int, optional
            Lp norm for perturbation: 1, 2, or np.inf (default: 2).
        max_samples_per_class : int, optional
            Maximum samples per class. If None, use all available.
        batch_size : int, optional
            Process samples in batches (for GPU memory). If None, process all at once.
        build_unions : bool, optional
            Build Shapely polygon unions for visualization (default: False).
            Only works for 2D data.
        verbose : bool, optional
            Print progress information (default: True).

        Returns
        -------
        self
            Returns self for method chaining.
        """
        self.eps = eps
        self.norm = norm

        if verbose:
            print(f"Building certified atlas (eps={eps}, L{norm} norm)...")

        # Step 1: Compute LiRPA bounds for all classes
        if verbose:
            print("  Computing LiRPA bounds...")

        self.bounds = self._preimage.compute_all_bounds(
            eps=eps,
            norm=norm,
            max_samples_per_class=max_samples_per_class,
            batch_size=batch_size
        )

        # Step 2: Build BVH spatial index for each class
        if verbose:
            print("  Building BVH spatial indices...")

        self.bvh_indices = {}
        for label in range(self.n_classes):
            centers = self.bounds[label]['X']
            self.bvh_indices[label] = BVHIndex(centers, eps)

            if verbose:
                bvh = self.bvh_indices[label]
                print(f"    Class {label}: {bvh.n_polytopes} polytopes, "
                      f"tree depth {bvh.tree_depth}")

        # Step 3: Optionally build polygon unions (for 2D visualization)
        if build_unions:
            if self.bounds[0]['X'].shape[1] != 2:
                print("  Warning: build_unions=True only works for 2D data, skipping.")
            else:
                if verbose:
                    print("  Building polygon unions...")
                self._class_unions = {}
                for label in range(self.n_classes):
                    self._class_unions[label] = build_class_union(
                        label, self.bounds, eps
                    )

        if verbose:
            total_polytopes = sum(
                self.bvh_indices[l].n_polytopes for l in range(self.n_classes)
            )
            print(f"Done! Total: {total_polytopes} polytopes across {self.n_classes} classes")

        return self

    def _project_onto_polytope(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        center: np.ndarray,
        maxiter: Optional[int] = None,
        tol: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Project point x0 onto the polytope {x : A @ x + b >= 0} ∩ B(center, eps).

        Solves: min ||x - x0||^2  s.t.  A @ x + b >= 0, ||x - center||_p <= eps

        Parameters
        ----------
        x0 : np.ndarray
            Query point to project.
        A : np.ndarray
            LiRPA constraint matrix.
        b : np.ndarray
            LiRPA constraint bias.
        center : np.ndarray
            Center of the polytope (anchor point).
        maxiter : int, optional
            Maximum solver iterations. If None, uses self.solver_maxiter.
        tol : float, optional
            Solver tolerance. If None, uses self.solver_tol.
        """
        # Use provided params or fall back to instance defaults
        maxiter = maxiter if maxiter is not None else self.solver_maxiter
        tol = tol if tol is not None else self.solver_tol

        d = len(x0)

        # Combine LiRPA constraints with box constraints (box is always valid)
        A_box, b_box = ball_box_constraints(center, self.eps)
        A_full = np.vstack([A, A_box])
        b_full = np.concatenate([b, b_box])

        # Objective: minimize ||x - x0||^2
        def objective(x):
            return np.sum((x - x0) ** 2)

        def gradient(x):
            return 2 * (x - x0)

        # Linear constraints: A_full @ x + b_full >= 0
        constraints = [{
            'type': 'ineq',
            'fun': lambda x: A_full @ x + b_full,
            'jac': lambda x: A_full
        }]

        # Add norm-specific ball constraint
        # LiRPA bounds are only valid within the Lp ball used during computation
        if self.norm == 2:
            # L2 ball: ||x - center||_2 <= eps  →  eps^2 - ||x - center||_2^2 >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: self.eps**2 - np.sum((x - center)**2),
                'jac': lambda x: -2 * (x - center)
            })
        elif self.norm == 1:
            # L1 ball: ||x - center||_1 <= eps  →  eps - sum(|x - center|) >= 0
            # This is non-smooth, so we use the box as approximation and verify later
            pass
        # For L∞, the box constraint already handles it

        # Bounds for the box (always apply as outer bound)
        bounds = [(center[i] - self.eps, center[i] + self.eps) for i in range(d)]

        # Start from center (guaranteed feasible if polytope is non-empty)
        result = minimize(
            objective,
            center,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': tol, 'maxiter': maxiter}
        )

        if result.success:
            x_proj = result.x

            # CRITICAL: Verify constraints are actually satisfied
            # The solver may return points that slightly violate constraints
            margins = A_full @ x_proj + b_full
            min_margin = np.min(margins)

            # Reject if any constraint is violated (with small tolerance)
            if min_margin < -1e-7:
                return None, np.inf

            # Also verify box constraints explicitly
            if np.any(x_proj < center - self.eps - 1e-7) or \
               np.any(x_proj > center + self.eps + 1e-7):
                return None, np.inf

            # CRITICAL: If using L2 norm, verify point is within L2 ball
            # The QP uses L∞ box which is LARGER than L2 ball
            # LiRPA bounds are only valid within the Lp ball used during computation
            if self.norm == 2:
                l2_dist = np.linalg.norm(x_proj - center)
                if l2_dist > self.eps + 1e-7:
                    return None, np.inf
            elif self.norm == 1:
                l1_dist = np.sum(np.abs(x_proj - center))
                if l1_dist > self.eps + 1e-7:
                    return None, np.inf
            # For L∞ norm, box constraint already covers it

            dist = np.linalg.norm(x_proj - x0)
            return x_proj, dist
        else:
            return None, np.inf

    def find_counterfactual(
        self,
        x_query: np.ndarray,
        target_class: int,
        method: str = 'bvh',
        k: int = 10,
        solver_maxiter: Optional[int] = None,
        solver_tol: Optional[float] = None
    ) -> CounterfactualResult:
        """
        Find the closest counterfactual for a query point.

        Parameters
        ----------
        x_query : np.ndarray
            The query point, shape (d,).
        target_class : int
            The target class for the counterfactual.
        method : str, optional
            Search method: 'bvh' (branch-and-bound) or 'knn' (k-nearest neighbors).
            Default: 'bvh'.
        k : int, optional
            For 'knn' method: number of nearest neighbors to try.
            Default: 10.
        solver_maxiter : int, optional
            Maximum iterations for QP solver. If None, uses instance default.
        solver_tol : float, optional
            Tolerance for QP solver. If None, uses instance default.

        Returns
        -------
        CounterfactualResult
            Result containing the counterfactual point and metadata.

        Raises
        ------
        ValueError
            If atlas hasn't been built yet.
        """
        if self.bounds is None:
            raise ValueError("Atlas not built. Call build() first.")

        x_query = np.asarray(x_query).flatten()
        bd = self.bounds[target_class]

        if method == 'bvh':
            # Branch-and-bound search using BVH
            def project_fn(idx: int) -> Tuple[Optional[np.ndarray], float]:
                return self._project_onto_polytope(
                    x_query,
                    bd['lA'][idx],
                    bd['lbias'][idx],
                    bd['X'][idx],
                    maxiter=solver_maxiter,
                    tol=solver_tol
                )

            bvh = self.bvh_indices[target_class]
            x_cf, dist, anchor_idx, n_qp = bvh.query_nearest(x_query, project_fn)

        elif method == 'knn':
            # K-nearest neighbors heuristic
            anchors = bd['X']
            dists_to_anchors = np.linalg.norm(anchors - x_query, axis=1)
            nearest_indices = np.argsort(dists_to_anchors)[:k]

            x_cf = None
            dist = np.inf
            anchor_idx = None
            n_qp = 0

            for idx in nearest_indices:
                proj, d = self._project_onto_polytope(
                    x_query,
                    bd['lA'][idx],
                    bd['lbias'][idx],
                    bd['X'][idx],
                    maxiter=solver_maxiter,
                    tol=solver_tol
                )
                n_qp += 1

                if d < dist:
                    x_cf = proj
                    dist = d
                    anchor_idx = idx

        else:
            raise ValueError(f"Unknown method: {method}. Use 'bvh' or 'knn'.")

        return CounterfactualResult(
            x_cf=x_cf,
            distance=dist,
            target_class=target_class,
            anchor_idx=anchor_idx,
            n_qp_solved=n_qp,
            success=x_cf is not None
        )

    def find_counterfactual_batch(
        self,
        X_query: np.ndarray,
        target_class: int,
        method: str = 'bvh',
        k: int = 10,
        solver_maxiter: Optional[int] = None,
        solver_tol: Optional[float] = None
    ) -> List[CounterfactualResult]:
        """
        Find counterfactuals for a batch of query points.

        Parameters
        ----------
        X_query : np.ndarray
            Array of query points, shape (n_queries, d).
        target_class : int
            The target class for all counterfactuals.
        method : str, optional
            Search method: 'bvh' or 'knn'. Default: 'bvh'.
        k : int, optional
            For 'knn' method: number of nearest neighbors. Default: 10.
        solver_maxiter : int, optional
            Maximum iterations for QP solver. If None, uses instance default.
        solver_tol : float, optional
            Tolerance for QP solver. If None, uses instance default.

        Returns
        -------
        list[CounterfactualResult]
            List of results, one per query point.
        """
        results = []
        for x in X_query:
            results.append(self.find_counterfactual(
                x, target_class, method, k,
                solver_maxiter=solver_maxiter,
                solver_tol=solver_tol
            ))
        return results

    def verify_counterfactual(
        self,
        x_cf: np.ndarray,
        target_class: int
    ) -> Dict:
        """
        Verify that a counterfactual is correctly classified by the model.

        Parameters
        ----------
        x_cf : np.ndarray
            The counterfactual point, shape (d,).
        target_class : int
            The expected target class.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'predicted': int, model's prediction
            - 'target': int, expected class
            - 'valid': bool, whether prediction matches target
            - 'logits': np.ndarray, raw model outputs
        """
        self.model.eval()
        x_tensor = torch.tensor(x_cf, dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.cnn:
            # Reshape for CNN if needed
            x_tensor = x_tensor.view(-1, 1, 28, 28)

        with torch.no_grad():
            logits = self.model(x_tensor).cpu().numpy()[0]
            pred = np.argmax(logits)

        return {
            'predicted': int(pred),
            'target': target_class,
            'valid': pred == target_class,
            'logits': logits
        }

    def get_class_union(self, label: int):
        """
        Get the Shapely polygon union for a class (2D only).

        Only available if build() was called with build_unions=True.

        Parameters
        ----------
        label : int
            The class label.

        Returns
        -------
        Polygon or MultiPolygon
            The union of all certified polytopes for this class.

        Raises
        ------
        ValueError
            If polygon unions weren't built.
        """
        if self._class_unions is None:
            raise ValueError(
                "Polygon unions not built. Call build() with build_unions=True."
            )
        return self._class_unions[label]

    def get_refined_unions(self, priority_order: Optional[List[int]] = None) -> Dict:
        """
        Get refined polygon unions with overlaps removed (2D only).

        Parameters
        ----------
        priority_order : list[int], optional
            Priority order for resolving overlaps (highest priority first).
            If None, uses sorted label order.

        Returns
        -------
        dict
            Dictionary mapping labels to refined Polygon/MultiPolygon.
        """
        if self._class_unions is None:
            raise ValueError(
                "Polygon unions not built. Call build() with build_unions=True."
            )
        return refine_unions_by_priority(self._class_unions, priority_order)

    def summary(self) -> str:
        """Return a summary string of the atlas."""
        if self.bounds is None:
            return "CertifiedAtlas (not built)"

        lines = [
            f"CertifiedAtlas Summary",
            f"=" * 40,
            f"Classes: {self.n_classes}",
            f"Perturbation: eps={self.eps}, L{self.norm} norm",
            f"Device: {self.device}",
            f""
        ]

        total_polytopes = 0
        for label in range(self.n_classes):
            n = self.bvh_indices[label].n_polytopes
            total_polytopes += n
            depth = self.bvh_indices[label].tree_depth
            lines.append(f"  Class {label}: {n:4d} polytopes (BVH depth {depth})")

        lines.append(f"")
        lines.append(f"Total polytopes: {total_polytopes}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "built" if self.bounds is not None else "not built"
        return f"CertifiedAtlas(n_classes={self.n_classes}, {status})"
