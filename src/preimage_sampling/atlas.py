"""
CertifiedAtlas: High-level API for certified counterfactual generation.

This module provides a simple, user-friendly interface that combines:
- LiRPA bound propagation for preimage certification
- Polytope construction and union operations
- BVH spatial indexing for efficient queries
- QP-based counterfactual projection

Example usage:
    >>> from preimage_sampling import CertifiedAtlas
    >>> atlas = CertifiedAtlas(model, dataset, device, eps=0.1, norm=2)
    >>> atlas.build()
    >>> cf = atlas.find_counterfactual(x_query, target_class=3)
"""

import time
import numpy as np

_SOLVER_TOL = 1e-9  # QP convergence tolerance (fixed)
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
        # Build configuration
        norm: int = 2,
        eps_strategy: Optional[EpsStrategy] = None,
        batch_size: Optional[int] = None,
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
        # Query configuration
        default_query_method: str = "sorted",
        solver_maxiter: int = 500,
    ):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cnn = cnn

        # Build config — stored here, consumed by build()
        self.norm = norm
        self.eps_strategy = eps_strategy
        self.batch_size = batch_size
        self.ohe_slices = ohe_slices

        self.solver_maxiter = solver_maxiter

        allowed_methods = {"sorted", "bvh"}
        if default_query_method not in allowed_methods:
            raise ValueError(
                f"default_query_method must be one of {allowed_methods}, got {default_query_method!r}"
            )
        self.default_query_method = default_query_method


        # Initialize preimage approximation handler
        self._preimage = PreimageApproximation(model, dataset, self.device, cnn=cnn)
        self.n_classes = self._preimage.n_classes

        # These are populated by build()
        self.bounds: Optional[Dict] = None
        self.bvh_indices: Optional[Dict[int, BVHIndex]] = None

        # Optional: Shapely polygon unions (only for 2D visualization)
        self._class_unions: Optional[Dict] = None

    def build(self, build_unions: bool = False, verbose: bool = True) -> 'CertifiedAtlas':
        """Compute LiRPA bounds and build BVH spatial indices from the dataset."""
        # Resolve eps strategy
        eps_strategy = self.eps_strategy
        if eps_strategy is None:
            eps_strategy = ConstantEpsStrategy(0.1)
        self.eps_strategy = eps_strategy

        norm = self.norm

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
            batch_size=self.batch_size,
            dtype=torch.float32,
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

    def _projection_initial_guess(
        self,
        x0: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        box_eps: float,
        ball_eps: float,
        fixed_dims: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build a geometry-aware warm start for polytope projection.

        The initializer starts from the anchor center, pins any fixed query
        coordinates, moves toward the query along the corresponding segment
        until it reaches the simple trust region (box / ball), and retracts
        along that segment if the reference point is already halfspace-feasible.
        """
        ref = center.astype(np.float64, copy=True)
        if fixed_dims is not None and len(fixed_dims) > 0:
            ref[fixed_dims] = x0[fixed_dims]

        direction = x0 - ref
        ref_offset = ref - center

        if self.norm == np.inf:
            if np.max(np.abs(ref_offset)) > box_eps + 1e-12:
                return ref
            step_norm = float(np.max(np.abs(direction)))
            t_region = 0.0 if step_norm <= 1e-15 else min(1.0, box_eps / step_norm)
        elif self.norm == 2:
            ref_norm = float(np.linalg.norm(ref_offset, ord=2))
            remaining_sq = ball_eps ** 2 - ref_norm ** 2
            step_norm = float(np.linalg.norm(direction, ord=2))
            if remaining_sq <= 0.0 or step_norm <= 1e-15:
                t_region = 0.0
            else:
                t_region = min(1.0, np.sqrt(max(0.0, remaining_sq)) / step_norm)
        elif self.norm == 1:
            ref_norm = float(np.linalg.norm(ref_offset, ord=1))
            remaining = ball_eps - ref_norm
            step_norm = float(np.linalg.norm(direction, ord=1))
            if remaining <= 0.0 or step_norm <= 1e-15:
                t_region = 0.0
            else:
                t_region = min(1.0, remaining / step_norm)
        else:
            ref_norm = float(np.linalg.norm(ref_offset, ord=self.norm))
            remaining = ball_eps - ref_norm
            step_norm = float(np.linalg.norm(direction, ord=self.norm))
            if remaining <= 0.0 or step_norm <= 1e-15:
                t_region = 0.0
            else:
                t_region = min(1.0, remaining / step_norm)

        candidate = ref + t_region * direction
        if fixed_dims is not None and len(fixed_dims) > 0:
            candidate[fixed_dims] = x0[fixed_dims]

        margins_ref = A_full @ ref + b_full
        if np.min(margins_ref) >= -1e-9:
            step = candidate - ref
            slopes = A_full @ step
            harmful = slopes < 0.0
            alpha = 1.0
            if np.any(harmful):
                alpha = min(1.0, float(np.min(margins_ref[harmful] / (-slopes[harmful]))))
            alpha = max(0.0, alpha)
            if alpha < 1.0:
                alpha *= 0.999  # stay slightly inside the active halfspace
            candidate = ref + alpha * step
            if fixed_dims is not None and len(fixed_dims) > 0:
                candidate[fixed_dims] = x0[fixed_dims]
            return candidate

        if np.min(A_full @ candidate + b_full) >= -1e-9:
            return candidate

        return ref

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
        z.value = self._projection_initial_guess(
            x0, A_full, b_full, center, box_eps, ball_eps, fixed_dims=fixed_dims
        )

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

        # Norm-aware solver ordering: CLARABEL first for L2 (SOCP), OSQP first for L1 (QP-like).
        if self.norm == 2:
            solvers = ('CLARABEL', 'OSQP', 'SCS')
        else:
            solvers = ('OSQP', 'CLARABEL', 'SCS')
        solved = False
        for i, _solver in enumerate(solvers):
            last = (i == len(solvers) - 1)
            try:
                problem.solve(solver=_solver, verbose=False, warm_start=True)
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

        x_init = self._projection_initial_guess(
            x0, A_full, b_full, center, box_eps, box_eps, fixed_dims=fixed_dims
        )

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

    def _polytope_aware_decode(
        self,
        x_star: np.ndarray,
        x_query: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        ball_eps: float,
        top_k: int = 3,
        tol: float = 1e-6,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Snap the continuous QP solution to the nearest certified OHE vertex.

        Generates up to 1 + top_k discrete candidates (argmax baseline +
        variants where the most uncertain categorical blocks use their 2nd-best
        category) and returns the closest one to x_query that satisfies the
        eroded polytope certificate. Returns (None, inf) if none pass.
        """
        ohe_slices = self.ohe_slices  # list of (start, end) per categorical block
        d = len(x_star)

        # Build the argmax baseline
        x_base = x_star.copy()
        for s, e in ohe_slices:
            block = x_star[s:e]
            snap = np.zeros(e - s)
            snap[int(np.argmax(block))] = 1.0
            x_base[s:e] = snap

        candidates = [x_base]

        # Identify the top_k most uncertain blocks (smallest top1 - top2 margin)
        margins = []
        for s, e in ohe_slices:
            block = x_star[s:e]
            if len(block) < 2:
                continue
            sorted_vals = np.sort(block)[::-1]
            margins.append((sorted_vals[0] - sorted_vals[1], s, e))
        margins.sort(key=lambda t: t[0])  # ascending: most uncertain first

        for _, s, e in margins[:top_k]:
            block = x_star[s:e]
            order = np.argsort(block)[::-1]  # indices sorted by value descending
            if len(order) < 2:
                continue
            x_alt = x_base.copy()
            snap_alt = np.zeros(e - s)
            snap_alt[order[1]] = 1.0  # use 2nd-best category
            x_alt[s:e] = snap_alt
            candidates.append(x_alt)

        def _check(x_cand: np.ndarray) -> bool:
            if np.any(A_full @ x_cand + b_full < -tol):
                return False
            if self.norm == np.inf:
                return bool(np.max(np.abs(x_cand - center)) <= ball_eps + tol)
            return bool(np.linalg.norm(x_cand - center, ord=self.norm) <= ball_eps + tol)

        best_x, best_dist = None, np.inf
        for cand in candidates:
            if _check(cand):
                d_cand = float(np.linalg.norm(cand - x_query))
                if d_cand < best_dist:
                    best_x, best_dist = cand, d_cand

        return best_x, best_dist

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
            Solver tolerance. Defaults to _SOLVER_TOL (1e-9).
        """
        maxiter = maxiter if maxiter is not None else self.solver_maxiter
        tol = _SOLVER_TOL
        if robust_norm is None:
            robust_norm = self.norm

        d = len(x0)
        A_full, b_full, box_eps, ball_eps = self._erode_constraints(
            A, b, center, d, delta, robust_norm, eps_i
        )
        if A_full is None:
            return None, np.inf

        # Dispatch: CVXPY for L2/L1, or SLSQP for L∞
        if self.norm in (1, 2) and CVXPY_AVAILABLE:
            x_proj, dist = self._project_cvxpy(x0, A_full, b_full, center, box_eps, ball_eps,
                                               fixed_dims=fixed_dims,
                                               ohe_slices=self.ohe_slices)
        else:
            x_proj, dist = self._project_slsqp(x0, A_full, b_full, center, box_eps, maxiter, tol,
                                               fixed_dims=fixed_dims,
                                               ohe_slices=self.ohe_slices)

        # If OHE slices are set, snap the continuous solution to the nearest
        # certified discrete vertex. Reject the polytope if none exists.
        if x_proj is not None and self.ohe_slices:
            x_proj, dist = self._polytope_aware_decode(
                x_proj, x0, A_full, b_full, center, ball_eps
            )

        return x_proj, dist

    def _make_project_fn(self, x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims):
        """Return a timed projection closure and a mutable [time_s] accumulator."""
        projection_time_s = [0.0]

        def project_fn(idx: int) -> Tuple[Optional[np.ndarray], float]:
            t0 = time.perf_counter()
            out = self._project_onto_polytope(
                x_query, bd['lA'][idx], bd['lbias'][idx], bd['X'][idx],
                eps_i=float(bd['eps'][idx]),
                delta=delta, robust_norm=robust_norm,
                maxiter=solver_maxiter, fixed_dims=fixed_dims,
            )
            projection_time_s[0] += time.perf_counter() - t0
            return out

        return project_fn, projection_time_s

    def _search_bvh(self, x_query, bd, target_class, delta, robust_norm, solver_maxiter, fixed_dims):
        """Branch-and-bound BVH search. Returns (x_cf, dist, anchor_idx, n_qp, profiling_dict)."""
        project_fn, projection_time_s = self._make_project_fn(
            x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims)
        bvh_stats: Dict[str, float] = {}
        bvh = self.bvh_indices[target_class]
        t0 = time.perf_counter()
        x_cf, dist, anchor_idx, n_qp = bvh.query_nearest(x_query, project_fn, stats_out=bvh_stats)
        query_loop_time_s = time.perf_counter() - t0
        profiling = {
            "query_loop_time_ms": 1e3 * query_loop_time_s,
            "search_time_ms": 1e3 * max(0.0, query_loop_time_s - projection_time_s[0]),
            "projection_time_ms": 1e3 * projection_time_s[0],
            "n_nodes_popped": bvh_stats.get("n_nodes_popped", np.nan),
            "n_nodes_pruned": bvh_stats.get("n_nodes_pruned", np.nan),
            "n_leaves_visited": bvh_stats.get("n_leaves_visited", np.nan),
            "max_queue_size": bvh_stats.get("max_queue_size", np.nan),
            "n_candidates_considered": bvh_stats.get("n_candidates_considered", np.nan),
        }
        return x_cf, dist, anchor_idx, n_qp, profiling

    def _search_sorted(self, x_query, bd, target_class, delta, robust_norm, solver_maxiter, fixed_dims):
        """Sorted lower-bound scan. Returns (x_cf, dist, anchor_idx, n_qp, profiling_dict)."""
        project_fn, projection_time_s = self._make_project_fn(
            x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims)
        sorted_stats: Dict[str, float] = {}
        bvh = self.bvh_indices[target_class]
        t0 = time.perf_counter()
        x_cf, dist, anchor_idx, n_qp = bvh.query_sorted_lower_bounds(
            x_query, eps_array=bd['eps'], project_fn=project_fn,
            atlas_norm=self.norm if self.norm is not None else 2,
            stats_out=sorted_stats,
        )
        query_loop_time_s = time.perf_counter() - t0
        profiling = {
            "query_loop_time_ms": 1e3 * query_loop_time_s,
            "search_time_ms": 1e3 * max(0.0, query_loop_time_s - projection_time_s[0]),
            "projection_time_ms": 1e3 * projection_time_s[0],
            "n_candidates_considered": sorted_stats.get("n_candidates_considered", np.nan),
            "n_candidates_total": sorted_stats.get("n_candidates_total", np.nan),
            "n_candidates_pruned_by_bound": sorted_stats.get("n_candidates_pruned_by_bound", np.nan),
            "best_lower_bound_at_termination": sorted_stats.get("best_lower_bound_at_termination", np.nan),
        }
        return x_cf, dist, anchor_idx, n_qp, profiling

    def find_counterfactual(
        self,
        x_query: np.ndarray,
        target_class: int,
        method: Optional[str] = None,
        delta: float = 0.0,
        robust_norm: Optional[int] = None,
        solver_maxiter: Optional[int] = None,
        fixed_dims: Optional[np.ndarray] = None
    ) -> CounterfactualResult:
        """Find the closest counterfactual for a query point."""

        if self.bounds is None:
            raise ValueError("Atlas not built. Call build() first.")

        x_query = np.asarray(x_query).flatten()
        bd = self.bounds[target_class]
        resolved_method = method or self.default_query_method
        profiling: Dict[str, float] = {
            "method": resolved_method,
            "delta": float(delta),
        }
        t_total_start = time.perf_counter()

        if resolved_method == 'bvh':
            x_cf, dist, anchor_idx, n_qp, search_profiling = self._search_bvh(
                x_query, bd, target_class, delta, robust_norm, solver_maxiter, fixed_dims)
        elif resolved_method == 'sorted':
            x_cf, dist, anchor_idx, n_qp, search_profiling = self._search_sorted(
                x_query, bd, target_class, delta, robust_norm, solver_maxiter, fixed_dims)
        else:
            raise ValueError(f"Unknown method: {resolved_method}. Use 'sorted' or 'bvh'.")

        profiling.update(search_profiling)

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
        delta: float = 0.0,
        robust_norm: Optional[int] = None,
        solver_maxiter: Optional[int] = None,
        fixed_dims: Optional[np.ndarray] = None
    ) -> List[CounterfactualResult]:
        """Find counterfactuals for a batch of query points."""
        results = []
        for x in X_query:
            results.append(self.find_counterfactual(
                x, target_class, method,
                delta=delta,
                robust_norm=robust_norm,
                solver_maxiter=solver_maxiter,
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
