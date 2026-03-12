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
from dataclasses import dataclass, field

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .certification.lirpa import PreimageApproximation
from .eps_strategies import EpsStrategy, ConstantEpsStrategy
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
    profiling : dict
        Query-time profiling metadata.
    """
    x_cf: Optional[np.ndarray]
    distance: float
    target_class: int
    anchor_idx: Optional[int]
    n_qp_solved: int
    success: bool
    profiling: Dict[str, float] = field(default_factory=dict)


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
        Each entry contains an 'eps' key with per-sample epsilon values.
    bvh_indices : dict or None
        BVH spatial indices for each class (after calling build()).
    eps : float or None
        Perturbation radius if a constant strategy was used; None otherwise.
    eps_strategy : EpsStrategy or None
        The strategy used to compute per-sample epsilon (after calling build()).
    norm : int or None
        Lp norm used for building (after calling build()).
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        device: Union[torch.device, str],
        cnn: bool = False,
        default_query_method: str = "sorted",
        cvxpy_solver_policy: str = "auto",
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
        default_query_method : str, optional
            Default search method used by ``find_counterfactual`` when
            ``method`` is not explicitly provided. One of
            ``{'sorted', 'bvh', 'knn'}`` (default: "sorted").
        cvxpy_solver_policy : str, optional
            Solver ordering policy for CVXPY projections.
            - ``"auto"``: norm-aware ordering (L2: CLARABEL→OSQP→SCS,
              L1: OSQP→CLARABEL→SCS)
            - ``"legacy"``: OSQP→CLARABEL→SCS for all norms
            Default: "auto".
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

        allowed_methods = {"sorted", "bvh", "knn"}
        if default_query_method not in allowed_methods:
            raise ValueError(
                f"default_query_method must be one of {allowed_methods}, got {default_query_method!r}"
            )
        self.default_query_method = default_query_method

        allowed_solver_policies = {"auto", "legacy"}
        if cvxpy_solver_policy not in allowed_solver_policies:
            raise ValueError(
                f"cvxpy_solver_policy must be one of {allowed_solver_policies}, got {cvxpy_solver_policy!r}"
            )
        self.cvxpy_solver_policy = cvxpy_solver_policy

        # Initialize preimage approximation handler
        self._preimage = PreimageApproximation(model, dataset, self.device, cnn=cnn)
        self.n_classes = self._preimage.n_classes

        # These are populated by build()
        self.bounds: Optional[Dict] = None
        self.bvh_indices: Optional[Dict[int, BVHIndex]] = None
        self.eps: Optional[float] = None          # scalar iff ConstantEpsStrategy
        self.eps_strategy: Optional[EpsStrategy] = None
        self.norm: Optional[int] = None

        # Optional: Shapely polygon unions (only for 2D visualization)
        self._class_unions: Optional[Dict] = None

        # OHE categorical block slices: list of (start, end) index pairs.
        # When set (e.g. for TabularClassifier), the QP enforces sum(x[s:e])==1 per block.
        self.ohe_slices: Optional[List[Tuple[int, int]]] = None

    def build(
        self,
        eps: Optional[float] = None,
        norm: int = 2,
        eps_strategy: Optional[EpsStrategy] = None,
        max_samples_per_class: Optional[int] = None,
        batch_size: Optional[int] = None,
        build_unions: bool = False,
        verbose: bool = True,
        dtype=None
    ) -> 'CertifiedAtlas':
        """
        Build the certified atlas by computing LiRPA bounds and spatial indices.

        This is the "offline" phase that should be run once before generating
        counterfactuals.

        Parameters
        ----------
        eps : float, optional
            Constant perturbation radius for all samples.  Mutually exclusive
            with ``eps_strategy``.  Kept for backward compatibility.
        norm : int, optional
            Lp norm for perturbation: 1, 2, or np.inf (default: 2).
        eps_strategy : EpsStrategy, optional
            A pluggable strategy that returns a per-sample epsilon array.
            Mutually exclusive with ``eps``.  When omitted and ``eps`` is also
            omitted, defaults to ``ConstantEpsStrategy(0.1)``.
        max_samples_per_class : int, optional
            Maximum samples per class. If None, use all available.
        batch_size : int, optional
            Process samples in batches (for GPU memory). If None, process all at once.
            Ignored when the resolved strategy produces varying epsilon values
            (samples are then processed individually).
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
        # --- Resolve eps strategy ---
        if eps is not None and eps_strategy is not None:
            raise ValueError("Specify eps or eps_strategy, not both.")
        if eps is not None:
            eps_strategy = ConstantEpsStrategy(eps)
        elif eps_strategy is None:
            eps_strategy = ConstantEpsStrategy(0.1)

        self.eps_strategy = eps_strategy
        self.eps = eps  # scalar for ConstantEpsStrategy, None otherwise
        self.norm = norm

        # Compute per-sample epsilon for the full dataset
        X_all = self._preimage.dataset.tensors[0].numpy()
        y_all = self._preimage.dataset.tensors[1].numpy()
        eps_array = eps_strategy.compute_eps(X_all, y_all)

        if verbose:
            eps_min, eps_max = eps_array.min(), eps_array.max()
            if eps_min == eps_max:
                eps_desc = f"eps={eps_min:.4g}"
            else:
                eps_desc = f"eps in [{eps_min:.4g}, {eps_max:.4g}] ({type(eps_strategy).__name__})"
            print(f"Building certified atlas ({eps_desc}, L{norm} norm)...")

        # Step 1: Compute LiRPA bounds for all classes
        if verbose:
            print("  Computing LiRPA bounds...")

        self.bounds = self._preimage.compute_all_bounds(
            eps=0.1,            # fallback scalar (unused when eps_array is provided)
            norm=norm,
            max_samples_per_class=max_samples_per_class,
            batch_size=batch_size,
            dtype=dtype if dtype is not None else torch.float32,
            eps_array=eps_array,
        )

        # Step 2: Build BVH spatial index for each class
        if verbose:
            print("  Building BVH spatial indices...")

        self.bvh_indices = {}
        for label in range(self.n_classes):
            centers = self.bounds[label]['X']
            eps_class = self.bounds[label]['eps']
            self.bvh_indices[label] = BVHIndex(centers, eps_class)

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
                        label, self.bounds, self.bounds[label]['eps']
                    )

        if verbose:
            total_polytopes = sum(
                self.bvh_indices[l].n_polytopes for l in range(self.n_classes)
            )
            print(f"Done! Total: {total_polytopes} polytopes across {self.n_classes} classes")

        return self

    @staticmethod
    def _dual_norm(q) -> float:
        """Return the dual exponent of Lq: 1/q + 1/q* = 1."""
        if q == 1:
            return np.inf
        elif q == np.inf:
            return 1
        else:
            return q / (q - 1)

    @staticmethod
    def _norm_conversion_factor(d: int, from_norm, to_norm) -> float:
        """
        Return C such that ||x||_{to} <= C * ||x||_{from} for all x in R^d.

        Used to bound how much an Lq-ball of radius delta can extend in the
        Lp norm of the certified region.
        """
        inv_to = 0.0 if to_norm == np.inf else 1.0 / to_norm
        inv_from = 0.0 if from_norm == np.inf else 1.0 / from_norm
        exponent = max(0.0, inv_to - inv_from)
        return d ** exponent

    def _erode_constraints(
        self,
        A: np.ndarray,
        b: np.ndarray,
        center: np.ndarray,
        d: int,
        delta: float,
        robust_norm,
        eps_i: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute eroded constraints for delta-robust projection.

        Returns (A_full, b_full, box_eps, ball_eps) or raises if infeasible.
        """
        box_eps = eps_i - delta
        ball_factor = self._norm_conversion_factor(d, robust_norm, self.norm)
        ball_eps = eps_i - delta * ball_factor

        if box_eps <= 0 or ball_eps <= 0:
            return None, None, 0.0, 0.0

        # Erode LiRPA constraints: A @ x + b >= delta * ||a_i||_{q*}
        if delta > 0:
            q_dual = self._dual_norm(robust_norm)
            if q_dual == np.inf:
                row_dual_norms = np.max(np.abs(A), axis=1)
            elif q_dual == 1:
                row_dual_norms = np.sum(np.abs(A), axis=1)
            else:
                row_dual_norms = np.linalg.norm(A, axis=1, ord=q_dual)
            b_eroded = b - delta * row_dual_norms
        else:
            b_eroded = b

        A_box, b_box = ball_box_constraints(center, box_eps)
        A_full = np.vstack([A, A_box])
        b_full = np.concatenate([b_eroded, b_box])

        return A_full, b_full, box_eps, ball_eps

    def _project_cvxpy(
        self,
        x0: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        box_eps: float,
        ball_eps: float,
        fixed_dims: Optional[np.ndarray] = None,
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Project using CVXPY — handles L2 (SOCP) and L1 ball constraints natively.
        """
        d = len(x0)
        z = cp.Variable(d)

        objective = cp.Minimize(cp.sum_squares(z - x0))

        constraints = [
            A_full @ z + b_full >= 0,
            z >= center - box_eps,
            z <= center + box_eps,
        ]

        # Add the Lp ball constraint (handled natively by CVXPY)
        if self.norm == 2:
            constraints.append(cp.norm(z - center, 2) <= ball_eps)
        elif self.norm == 1:
            constraints.append(cp.norm(z - center, 1) <= ball_eps)
        # For L∞, box constraint already covers it

        # Fix specified dimensions to their query values
        if fixed_dims is not None and len(fixed_dims) > 0:
            constraints.append(z[fixed_dims] == x0[fixed_dims])

        # OHE simplex constraints: each categorical block must sum to 1 and be >= 0
        if ohe_slices is not None:
            for s, e in ohe_slices:
                constraints.append(cp.sum(z[s:e]) == 1.0)
                constraints.append(z[s:e] >= 0)

        problem = cp.Problem(objective, constraints)

        if self.cvxpy_solver_policy == "legacy":
            solvers = ('OSQP', 'CLARABEL', 'SCS')
        else:
            # Norm-aware solver policy:
            # - L2 uses SOCP constraints -> CLARABEL is typically strongest first choice.
            # - L1 is linear-constrained/QP-like -> OSQP is usually fastest first try.
            # - SCS remains the permissive fallback for both.
            if self.norm == 2:
                solvers = ('CLARABEL', 'OSQP', 'SCS')
            elif self.norm == 1:
                solvers = ('OSQP', 'CLARABEL', 'SCS')
            else:
                solvers = ('OSQP', 'CLARABEL', 'SCS')
        solved = False
        for i, _solver in enumerate(solvers):
            last = (i == len(solvers) - 1)
            try:
                problem.solve(solver=_solver, verbose=False)
                # Accept inaccurate only from the last solver (SCS) as a fallback.
                ok_status = ('optimal', 'optimal_inaccurate') if last else ('optimal',)
                if problem.status in ok_status and z.value is not None:
                    solved = True
                    break
            except Exception:
                continue
        if not solved:
            return None, np.inf

        x_proj = z.value
        dist = np.linalg.norm(x_proj - x0)
        return x_proj, dist

    def _project_slsqp(
        self,
        x0: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        box_eps: float,
        maxiter: int,
        tol: float,
        fixed_dims: Optional[np.ndarray] = None,
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Project using SLSQP — fast for L∞ (all-linear constraints).
        """
        d = len(x0)

        def objective(x):
            return np.sum((x - x0) ** 2)

        def gradient(x):
            return 2 * (x - x0)

        constraints = [{
            'type': 'ineq',
            'fun': lambda x: A_full @ x + b_full,
            'jac': lambda x: A_full
        }]

        bounds = [(center[i] - box_eps, center[i] + box_eps) for i in range(d)]

        # Fix specified dimensions: tighten bounds to a single value
        if fixed_dims is not None:
            for i in fixed_dims:
                bounds[i] = (x0[i], x0[i])

        # OHE simplex constraints: clamp categorical dims to [0,1] and enforce sum==1
        if ohe_slices is not None:
            for s, e in ohe_slices:
                for i in range(s, e):
                    lo, hi = bounds[i]
                    bounds[i] = (max(lo, 0.0), min(hi, 1.0))
                s_, e_ = int(s), int(e)
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, s=s_, e=e_: np.sum(x[s:e]) - 1.0,
                    'jac': lambda x, s=s_, e=e_: np.eye(len(x))[s:e].sum(axis=0),
                })

        # Start from center, but snap fixed dims to their required values so the
        # initial point already satisfies the box bounds.
        x_init = center.copy()
        if fixed_dims is not None:
            x_init[fixed_dims] = x0[fixed_dims]

        result = minimize(
            objective,
            x_init,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': tol, 'maxiter': maxiter}
        )

        x_proj = result.x

        # Do NOT trust result.success: SLSQP reports failure whenever the gradient
        # tolerance isn't met (common when the optimum lies on the box boundary).
        # Instead, verify feasibility directly.
        margins = A_full @ x_proj + b_full
        if np.min(margins) < -1e-7:
            return None, np.inf

        if np.any(x_proj < center - box_eps - 1e-7) or \
           np.any(x_proj > center + box_eps + 1e-7):
            return None, np.inf

        dist = np.linalg.norm(x_proj - x0)
        return x_proj, dist

    def _project_onto_polytope(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        center: np.ndarray,
        eps_i: float,
        delta: float = 0.0,
        robust_norm: Optional[int] = None,
        maxiter: Optional[int] = None,
        tol: Optional[float] = None,
        fixed_dims: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Project point x0 onto the (optionally eroded) polytope.

        Solves: min ||x - x0||^2  s.t.  eroded constraints hold

        When delta > 0, the polytope is eroded inward so that the entire
        B_q(x_cf, delta) ball (in the robustness norm q) lies within the
        original certified polytope. This guarantees the counterfactual is
        robust to adversarial perturbations of radius delta in the Lq norm.

        For L∞ norm, uses SLSQP (all constraints are linear).
        For L2/L1 norms, uses CVXPY which handles SOCP / L1 constraints natively.

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
        eps_i : float
            Perturbation radius for this specific polytope.
        delta : float, optional
            Robustness radius for polytope erosion. Default: 0.0 (no erosion).
        robust_norm : int or float, optional
            Lp norm for the robustness ball. Can differ from self.norm (the
            LiRPA certification norm). Default: None (uses self.norm).
        maxiter : int, optional
            Maximum solver iterations. If None, uses self.solver_maxiter.
        tol : float, optional
            Solver tolerance. If None, uses self.solver_tol.
        """
        maxiter = maxiter if maxiter is not None else self.solver_maxiter
        tol = tol if tol is not None else self.solver_tol
        if robust_norm is None:
            robust_norm = self.norm

        d = len(x0)
        A_full, b_full, box_eps, ball_eps = self._erode_constraints(
            A, b, center, d, delta, robust_norm, eps_i
        )
        if A_full is None:
            return None, np.inf

        # Dispatch: CVXPY for L2/L1 (handles SOCP / L1 natively), SLSQP for L∞
        if self.norm in (1, 2) and CVXPY_AVAILABLE:
            return self._project_cvxpy(x0, A_full, b_full, center, box_eps, ball_eps,
                                       fixed_dims=fixed_dims,
                                       ohe_slices=self.ohe_slices)
        else:
            return self._project_slsqp(x0, A_full, b_full, center, box_eps, maxiter, tol,
                                       fixed_dims=fixed_dims,
                                       ohe_slices=self.ohe_slices)

    def find_counterfactual(
        self,
        x_query: np.ndarray,
        target_class: int,
        method: Optional[str] = None,
        k: int = 10,
        delta: float = 0.0,
        robust_norm: Optional[int] = None,
        solver_maxiter: Optional[int] = None,
        solver_tol: Optional[float] = None,
        fixed_dims: Optional[np.ndarray] = None
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
            Search method: 'sorted' (vectorised center-distance lower-bound scan),
            'bvh' (branch-and-bound BVH), or 'knn' (k-nearest neighbors).
            Default: atlas ``default_query_method`` ("sorted" by default).
        k : int, optional
            For 'knn' method: number of nearest neighbors to try.
            Default: 10.
        delta : float, optional
            Robustness radius. When delta > 0, the certified polytopes are eroded
            inward so that the returned counterfactual is guaranteed robust to
            perturbations of radius delta. Default: 0.0 (no robustness margin).
        robust_norm : int or float, optional
            Lp norm for the robustness ball (e.g. 1, 2, np.inf). Can differ from
            the LiRPA certification norm (self.norm). Default: None (uses self.norm).
        solver_maxiter : int, optional
            Maximum iterations for QP solver. If None, uses instance default.
        solver_tol : float, optional
            Tolerance for QP solver. If None, uses instance default.
        fixed_dims : np.ndarray of int, optional
            Indices of embedding dimensions that must remain equal to the query
            value. Use ``get_fixed_dims()`` to convert feature names to indices.

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

        import time

        x_query = np.asarray(x_query).flatten()
        bd = self.bounds[target_class]
        resolved_method = method or self.default_query_method
        profiling: Dict[str, float] = {
            "method": resolved_method,
            "k": float(k),
            "delta": float(delta),
        }
        t_total_start = time.perf_counter()
        projection_time_s = 0.0

        if resolved_method == 'bvh':
            # Branch-and-bound search using BVH
            bvh_stats: Dict[str, float] = {}

            def project_fn(idx: int) -> Tuple[Optional[np.ndarray], float]:
                nonlocal projection_time_s
                t0 = time.perf_counter()
                out = self._project_onto_polytope(
                    x_query,
                    bd['lA'][idx],
                    bd['lbias'][idx],
                    bd['X'][idx],
                    eps_i=float(bd['eps'][idx]),
                    delta=delta,
                    robust_norm=robust_norm,
                    maxiter=solver_maxiter,
                    tol=solver_tol,
                    fixed_dims=fixed_dims
                )
                projection_time_s += (time.perf_counter() - t0)
                return out

            bvh = self.bvh_indices[target_class]
            t_search_start = time.perf_counter()
            x_cf, dist, anchor_idx, n_qp = bvh.query_nearest(
                x_query,
                project_fn,
                stats_out=bvh_stats,
            )
            query_loop_time_s = time.perf_counter() - t_search_start
            search_time_exclusive_s = max(0.0, query_loop_time_s - projection_time_s)

            profiling.update({
                "query_loop_time_ms": 1e3 * query_loop_time_s,
                "search_time_ms": 1e3 * search_time_exclusive_s,
                "projection_time_ms": 1e3 * projection_time_s,
                "n_nodes_popped": bvh_stats.get("n_nodes_popped", np.nan),
                "n_nodes_pruned": bvh_stats.get("n_nodes_pruned", np.nan),
                "n_leaves_visited": bvh_stats.get("n_leaves_visited", np.nan),
                "max_queue_size": bvh_stats.get("max_queue_size", np.nan),
                "n_candidates_considered": bvh_stats.get("n_candidates_considered", np.nan),
            })

        elif resolved_method == 'sorted':
            # Vectorised center-distance lower-bound scan with early stopping.
            # Tighter lower bounds than BVH for L2 atlas norm → fewer QP solves.
            sorted_stats: Dict[str, float] = {}

            def project_fn(idx: int) -> Tuple[Optional[np.ndarray], float]:
                nonlocal projection_time_s
                t0 = time.perf_counter()
                out = self._project_onto_polytope(
                    x_query,
                    bd['lA'][idx],
                    bd['lbias'][idx],
                    bd['X'][idx],
                    eps_i=float(bd['eps'][idx]),
                    delta=delta,
                    robust_norm=robust_norm,
                    maxiter=solver_maxiter,
                    tol=solver_tol,
                    fixed_dims=fixed_dims
                )
                projection_time_s += (time.perf_counter() - t0)
                return out

            bvh = self.bvh_indices[target_class]
            atlas_norm = self.norm if self.norm is not None else 2
            t_search_start = time.perf_counter()
            x_cf, dist, anchor_idx, n_qp = bvh.query_sorted_lower_bounds(
                x_query,
                eps_array=bd['eps'],
                project_fn=project_fn,
                atlas_norm=atlas_norm,
                stats_out=sorted_stats,
            )
            query_loop_time_s = time.perf_counter() - t_search_start
            search_time_exclusive_s = max(0.0, query_loop_time_s - projection_time_s)

            profiling.update({
                "query_loop_time_ms": 1e3 * query_loop_time_s,
                "search_time_ms": 1e3 * search_time_exclusive_s,
                "projection_time_ms": 1e3 * projection_time_s,
                "n_candidates_considered": sorted_stats.get("n_candidates_considered", np.nan),
                "n_candidates_total": sorted_stats.get("n_candidates_total", np.nan),
                "n_candidates_pruned_by_bound": sorted_stats.get("n_candidates_pruned_by_bound", np.nan),
                "best_lower_bound_at_termination": sorted_stats.get("best_lower_bound_at_termination", np.nan),
            })

        elif resolved_method == 'knn':
            # K-nearest neighbors heuristic
            t_search_start = time.perf_counter()
            anchors = bd['X']
            dists_to_anchors = np.linalg.norm(anchors - x_query, axis=1)
            nearest_indices = np.argsort(dists_to_anchors)[:k]
            search_time_s = time.perf_counter() - t_search_start

            x_cf = None
            dist = np.inf
            anchor_idx = None
            n_qp = 0

            for idx in nearest_indices:
                t0 = time.perf_counter()
                proj, d = self._project_onto_polytope(
                    x_query,
                    bd['lA'][idx],
                    bd['lbias'][idx],
                    bd['X'][idx],
                    eps_i=float(bd['eps'][idx]),
                    delta=delta,
                    robust_norm=robust_norm,
                    maxiter=solver_maxiter,
                    tol=solver_tol,
                    fixed_dims=fixed_dims
                )
                projection_time_s += (time.perf_counter() - t0)
                n_qp += 1

                if d < dist:
                    x_cf = proj
                    dist = d
                    anchor_idx = idx

            profiling.update({
                "query_loop_time_ms": 1e3 * (search_time_s + projection_time_s),
                "search_time_ms": 1e3 * search_time_s,
                "projection_time_ms": 1e3 * projection_time_s,
                "n_candidates_considered": float(len(nearest_indices)),
                "n_candidates_total": float(len(anchors)),
            })

        else:
            raise ValueError(f"Unknown method: {resolved_method}. Use 'sorted', 'bvh', or 'knn'.")

        total_time_s = time.perf_counter() - t_total_start
        profiling.update({
            "total_time_ms": 1e3 * total_time_s,
            "n_qp_solved": float(n_qp),
            "success": float(x_cf is not None),
            "distance": float(dist),
        })

        return CounterfactualResult(
            x_cf=x_cf,
            distance=dist,
            target_class=target_class,
            anchor_idx=anchor_idx,
            n_qp_solved=n_qp,
            success=x_cf is not None,
            profiling=profiling,
        )

    def find_counterfactual_batch(
        self,
        X_query: np.ndarray,
        target_class: int,
        method: Optional[str] = None,
        k: int = 10,
        delta: float = 0.0,
        robust_norm: Optional[int] = None,
        solver_maxiter: Optional[int] = None,
        solver_tol: Optional[float] = None,
        fixed_dims: Optional[np.ndarray] = None
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
            Search method: 'sorted', 'bvh' or 'knn'.
            Default: atlas ``default_query_method`` ("sorted" by default).
        k : int, optional
            For 'knn' method: number of nearest neighbors. Default: 10.
        delta : float, optional
            Robustness radius for polytope erosion. Default: 0.0.
        robust_norm : int or float, optional
            Lp norm for the robustness ball. Default: None (uses self.norm).
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
                delta=delta,
                robust_norm=robust_norm,
                solver_maxiter=solver_maxiter,
                solver_tol=solver_tol,
                fixed_dims=fixed_dims
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

        if self.eps is not None:
            eps_desc = f"eps={self.eps} (constant)"
        else:
            all_eps = np.concatenate([self.bounds[l]['eps'] for l in range(self.n_classes)])
            eps_desc = (
                f"eps in [{all_eps.min():.4g}, {all_eps.max():.4g}]"
                f" ({type(self.eps_strategy).__name__})"
            )

        lines = [
            f"CertifiedAtlas Summary",
            f"=" * 40,
            f"Classes: {self.n_classes}",
            f"Perturbation: {eps_desc}, L{self.norm} norm",
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
