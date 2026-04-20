"""
CertCFAtlas: High-level API for certified counterfactual generation.

This module provides a simple, user-friendly interface that combines:
- LiRPA bound propagation for preimage certification
- Polytope construction and union operations
- BVH spatial indexing for efficient queries
- QP-based counterfactual projection

Example usage:
    >>> from certcf import CertCFAtlas
    >>> atlas = CertCFAtlas(model, dataset, device, eps=0.1, norm=2)
    >>> atlas.build()
    >>> cf = atlas.find_counterfactual(x_query, target_class=3)
"""

import time
import heapq
from itertools import count, product
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


_EXACT_ENUM_PRODUCT_THRESHOLD = 4096
ProfileValue = Union[float, str, int, bool]


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
    profiling: Dict[str, ProfileValue] = field(default_factory=dict)


class CertCFAtlas:
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
        distance_norm: Optional[Union[int, float]] = None,
        lirpa_method: str = "backward",
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
        self.norm = self._normalize_lp_norm(norm)
        self.distance_norm = self._normalize_lp_norm(distance_norm if distance_norm is not None else norm)
        self.lirpa_method = str(lirpa_method)
        self.eps_strategy = eps_strategy
        self.batch_size = batch_size
        self.ohe_slices = ohe_slices

        self.solver_maxiter = solver_maxiter

        allowed_methods = {"sorted", "bvh", "nearest_anchor"}
        if default_query_method not in allowed_methods:
            raise ValueError(
                f"default_query_method must be one of {allowed_methods}, got {default_query_method!r}"
            )
        self.default_query_method = default_query_method


        # Initialize preimage approximation handler
        self._preimage = PreimageApproximation(model, dataset, self.device, cnn=cnn)
        self.n_classes = self._preimage.n_classes
        self.class_labels = list(self._preimage.class_labels)
        self.label_to_index = dict(self._preimage.label_to_index)
        sample_shape = tuple(self._preimage.dataset.tensors[0][0].shape)
        if self.cnn and len(sample_shape) == 1:
            flat_dim = int(np.prod(sample_shape))
            side = int(round(np.sqrt(flat_dim)))
            self.model_input_shape = (1, side, side) if side * side == flat_dim else sample_shape
        else:
            self.model_input_shape = sample_shape

        # These are populated by build()
        self.bounds: Optional[Dict] = None
        self.bvh_indices: Optional[Dict[int, BVHIndex]] = None

        # Optional: Shapely polygon unions (only for 2D visualization)
        self._class_unions: Optional[Dict] = None

    @staticmethod
    def _normalize_lp_norm(norm_value: Union[int, float, str]) -> Union[int, float]:
        """Normalize Lp norm values so callers can use ints or inf-like strings."""
        if norm_value == np.inf:
            return np.inf
        if isinstance(norm_value, str):
            lowered = norm_value.strip().lower()
            if lowered in {"inf", "infinity"}:
                return np.inf
            return int(lowered)
        return int(norm_value)

    def build(self, build_unions: bool = False, verbose: bool = True) -> 'CertCFAtlas':
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
            lirpa_method=self.lirpa_method,
        )

        # Step 2: Build BVH spatial index for each class
        if verbose:
            print("  Building BVH spatial indices...")

        self.bvh_indices = {}
        for label in self.class_labels:
            centers = self.bounds[label]['X']
            eps_class = self.bounds[label]['eps']
            self.bvh_indices[label] = BVHIndex(centers, eps_class)

            if verbose:
                bvh = self.bvh_indices[label]
                print(f"    Class {label}: {bvh.n_polytopes} polytopes, "
                      f"tree depth {bvh.tree_depth}")

        # Step 3: Optionally build polygon unions (for 2D visualization)
        if build_unions:
            first_label = self.class_labels[0]
            if self.bounds[first_label]['X'].shape[1] != 2:
                print("  Warning: build_unions=True only works for 2D data, skipping.")
            else:
                if verbose:
                    print("  Building polygon unions...")
                self._class_unions = {}
                for label in self.class_labels:
                    self._class_unions[label] = build_class_union(
                        label, self.bounds, self.bounds[label]['eps']
                    )

        if verbose:
            total_polytopes = sum(
                self.bvh_indices[l].n_polytopes for l in self.class_labels
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
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
        fixed_ohe_assignments: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """
        Build a geometry-aware warm start for polytope projection.

        The initializer starts from the anchor center, pins any fixed query
        coordinates, moves toward the query along the corresponding segment
        until it reaches the simple trust region (box / ball), and retracts
        along that segment if the reference point is already halfspace-feasible.
        """
        ref = center.astype(np.float64, copy=True)
        if ohe_slices is not None and fixed_ohe_assignments:
            ref = self._apply_fixed_ohe_assignments(ref, ohe_slices, fixed_ohe_assignments)
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
            if ref_norm >= ball_eps - 1e-15:
                t_region = 0.0
            else:
                t_region = self._largest_l2_ray_step(ref_offset, direction, ball_eps)
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
        if ohe_slices is not None and fixed_ohe_assignments:
            candidate = self._apply_fixed_ohe_assignments(candidate, ohe_slices, fixed_ohe_assignments)
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
            if ohe_slices is not None and fixed_ohe_assignments:
                candidate = self._apply_fixed_ohe_assignments(candidate, ohe_slices, fixed_ohe_assignments)
            if fixed_dims is not None and len(fixed_dims) > 0:
                candidate[fixed_dims] = x0[fixed_dims]
            return candidate

        if np.min(A_full @ candidate + b_full) >= -1e-9:
            return candidate

        return ref

    @staticmethod
    def _ohe_product_size(ohe_slices: Optional[List[Tuple[int, int]]]) -> int:
        if not ohe_slices:
            return 1
        size = 1
        for s, e in ohe_slices:
            size *= max(1, int(e - s))
        return int(size)

    @staticmethod
    def _apply_fixed_ohe_assignments(
        x: np.ndarray,
        ohe_slices: List[Tuple[int, int]],
        fixed_ohe_assignments: Dict[int, int],
    ) -> np.ndarray:
        out = np.asarray(x, dtype=np.float64).copy()
        for block_idx, cat_idx in fixed_ohe_assignments.items():
            s, e = ohe_slices[block_idx]
            out[s:e] = 0.0
            out[s + int(cat_idx)] = 1.0
        return out

    @staticmethod
    def _largest_l2_ray_step(
        ref_offset: np.ndarray,
        direction: np.ndarray,
        ball_eps: float,
    ) -> float:
        """Return the largest t in [0, 1] such that ||ref_offset + t*direction||_2 <= ball_eps."""
        a = float(np.dot(direction, direction))
        if a <= 1e-15:
            return 0.0
        b = 2.0 * float(np.dot(ref_offset, direction))
        c = float(np.dot(ref_offset, ref_offset) - ball_eps ** 2)
        discriminant = b * b - 4.0 * a * c
        if discriminant <= 0.0:
            return 0.0
        root = (-b + np.sqrt(discriminant)) / (2.0 * a)
        return float(np.clip(root, 0.0, 1.0))

    def _reshape_model_input(self, x: np.ndarray) -> torch.Tensor:
        """Convert a flat numpy point into the tensor shape expected by the wrapped model."""
        x_arr = np.asarray(x, dtype=np.float32).reshape(1, -1)
        x_tensor = torch.from_numpy(x_arr).to(self.device)
        if not self.cnn:
            return x_tensor
        expected_size = int(np.prod(self.model_input_shape))
        if x_tensor.shape[1] != expected_size:
            raise ValueError(
                f"Counterfactual has dim={x_tensor.shape[1]}, but CNN expects flattened dim={expected_size}."
            )
        return x_tensor.view((1,) + tuple(self.model_input_shape))

    def _is_ohe_valid(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """Check that each configured OHE block is exactly one-hot up to tolerance."""
        if not self.ohe_slices:
            return True
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        for start, end in self.ohe_slices:
            block = x_arr[start:end]
            if np.any(block < -tol) or np.any(block > 1.0 + tol):
                return False
            if not np.isclose(np.sum(block), 1.0, atol=tol):
                return False
            active = block >= 1.0 - tol
            if int(np.sum(active)) != 1:
                return False
            if np.any(block[~active] > tol):
                return False
        return True

    def _polytope_membership_for_anchor(
        self,
        x_cf: np.ndarray,
        bd: Dict[str, np.ndarray],
        anchor_idx: int,
        *,
        delta: float,
        robust_norm: Union[int, float],
        tol: float = 1e-6,
    ) -> Tuple[bool, bool]:
        """Return nominal and robust membership for one anchor polytope."""
        center = bd['X'][anchor_idx]
        eps_i = float(bd['eps'][anchor_idx])
        d = len(x_cf)

        A_nominal, b_nominal, _, nominal_ball_eps = self._erode_constraints(
            bd['lA'][anchor_idx], bd['lbias'][anchor_idx], center, d, 0.0, self.norm, eps_i
        )
        in_nominal = self._is_certified_candidate(
            x_cf, A_nominal, b_nominal, center, nominal_ball_eps, tol=tol
        )
        if delta <= 0.0:
            return in_nominal, in_nominal

        A_robust, b_robust, _, robust_ball_eps = self._erode_constraints(
            bd['lA'][anchor_idx], bd['lbias'][anchor_idx], center, d, delta, robust_norm, eps_i
        )
        if A_robust is None:
            return in_nominal, False
        robust = self._is_certified_candidate(
            x_cf, A_robust, b_robust, center, robust_ball_eps, tol=tol
        )
        return in_nominal, robust

    def _build_decode_profile(
        self,
        *,
        mode: str,
        exact_fallback_used: bool,
        nodes_visited: int,
        nodes_pruned: int,
        solver_calls: int,
        product_size: int,
        heuristic_success: bool,
    ) -> Dict[str, ProfileValue]:
        return {
            "decode_mode": mode,
            "decode_exact_fallback_used": bool(exact_fallback_used),
            "decode_nodes_visited": int(nodes_visited),
            "decode_nodes_pruned": int(nodes_pruned),
            "decode_solver_calls": int(solver_calls),
            "decode_product_size": int(product_size),
            "decode_heuristic_success": bool(heuristic_success),
        }

    def _is_certified_candidate(
        self,
        x_cand: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        ball_eps: float,
        tol: float = 1e-6,
    ) -> bool:
        if np.any(A_full @ x_cand + b_full < -tol):
            return False
        if self.norm == np.inf:
            return bool(np.max(np.abs(x_cand - center)) <= ball_eps + tol)
        return bool(np.linalg.norm(x_cand - center, ord=self.norm) <= ball_eps + tol)

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
        fixed_ohe_assignments: Optional[Dict[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Project using CVXPY — handles L2 (SOCP) and L1 ball constraints natively.
        """
        d = len(x0)
        z = cp.Variable(d)
        z.value = self._projection_initial_guess(
            x0,
            A_full,
            b_full,
            center,
            box_eps,
            ball_eps,
            fixed_dims=fixed_dims,
            ohe_slices=ohe_slices,
            fixed_ohe_assignments=fixed_ohe_assignments,
        )

        diff = z - x0
        if self.distance_norm == 1:
            objective = cp.Minimize(cp.norm(diff, 1))
        elif self.distance_norm == 2:
            objective = cp.Minimize(cp.sum_squares(diff))
        elif self.distance_norm == np.inf:
            objective = cp.Minimize(cp.norm(diff, np.inf))
        else:
            objective = cp.Minimize(cp.norm(diff, self.distance_norm))

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
            for block_idx, (s, e) in enumerate(ohe_slices):
                fixed_cat = None if fixed_ohe_assignments is None else fixed_ohe_assignments.get(block_idx)
                if fixed_cat is None:
                    constraints.append(cp.sum(z[s:e]) == 1.0)
                    constraints.append(z[s:e] >= 0)
                else:
                    target = np.zeros(e - s, dtype=np.float64)
                    target[int(fixed_cat)] = 1.0
                    constraints.append(z[s:e] == target)

        problem = cp.Problem(objective, constraints)

        # Norm-aware solver ordering.
        if self.norm == np.inf and self.distance_norm == 2:
            solvers = ('OSQP', 'CLARABEL', 'SCS')
        else:
            solvers = ('CLARABEL', 'SCS', 'OSQP')
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
        if np.min(A_full @ x_proj + b_full) < -1e-7:
            return None, np.inf
        if np.any(x_proj < center - box_eps - 1e-7) or np.any(x_proj > center + box_eps + 1e-7):
            return None, np.inf
        if self.norm == np.inf:
            in_ball = np.max(np.abs(x_proj - center)) <= ball_eps + 1e-7
        else:
            in_ball = np.linalg.norm(x_proj - center, ord=self.norm) <= ball_eps + 1e-7
        if not in_ball:
            return None, np.inf
        dist = float(np.linalg.norm(x_proj - x0, ord=self.distance_norm))
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
        fixed_ohe_assignments: Optional[Dict[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Project using SLSQP — fast for L∞ (all-linear constraints).
        """
        if self.distance_norm != 2:
            raise RuntimeError(
                "SLSQP fallback only supports L2 distance minimization. "
                "Install CVXPY to use distance_norm != 2."
            )
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
            for block_idx, (s, e) in enumerate(ohe_slices):
                fixed_cat = None if fixed_ohe_assignments is None else fixed_ohe_assignments.get(block_idx)
                if fixed_cat is None:
                    for i in range(s, e):
                        lo, hi = bounds[i]
                        bounds[i] = (max(lo, 0.0), min(hi, 1.0))
                    s_, e_ = int(s), int(e)
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda x, s=s_, e=e_: np.sum(x[s:e]) - 1.0,
                        'jac': lambda x, s=s_, e=e_: np.eye(len(x))[s:e].sum(axis=0),
                    })
                else:
                    fixed_cat = int(fixed_cat)
                    for i in range(s, e):
                        target = 1.0 if i == s + fixed_cat else 0.0
                        bounds[i] = (target, target)

        x_init = self._projection_initial_guess(
            x0,
            A_full,
            b_full,
            center,
            box_eps,
            box_eps,
            fixed_dims=fixed_dims,
            ohe_slices=ohe_slices,
            fixed_ohe_assignments=fixed_ohe_assignments,
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

        dist = float(np.linalg.norm(x_proj - x0, ord=self.distance_norm))
        return x_proj, dist

    def _solve_projection_subproblem(
        self,
        x0: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        box_eps: float,
        ball_eps: float,
        maxiter: int,
        tol: float,
        fixed_dims: Optional[np.ndarray] = None,
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
        fixed_ohe_assignments: Optional[Dict[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        use_cvxpy = CVXPY_AVAILABLE and (self.norm in (1, 2) or self.distance_norm != 2)
        if use_cvxpy:
            return self._project_cvxpy(
                x0,
                A_full,
                b_full,
                center,
                box_eps,
                ball_eps,
                fixed_dims=fixed_dims,
                ohe_slices=ohe_slices,
                fixed_ohe_assignments=fixed_ohe_assignments,
            )
        return self._project_slsqp(
            x0,
            A_full,
            b_full,
            center,
            box_eps,
            maxiter,
            tol,
            fixed_dims=fixed_dims,
            ohe_slices=ohe_slices,
            fixed_ohe_assignments=fixed_ohe_assignments,
        )

    def _heuristic_polytope_decode(
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
        ohe_slices = self.ohe_slices or []

        x_base = x_star.copy()
        for s, e in ohe_slices:
            block = x_star[s:e]
            snap = np.zeros(e - s)
            snap[int(np.argmax(block))] = 1.0
            x_base[s:e] = snap

        candidates = [x_base]

        margins = []
        for s, e in ohe_slices:
            block = x_star[s:e]
            if len(block) < 2:
                continue
            sorted_vals = np.sort(block)[::-1]
            margins.append((sorted_vals[0] - sorted_vals[1], s, e))
        margins.sort(key=lambda t: t[0])

        for _, s, e in margins[:top_k]:
            block = x_star[s:e]
            order = np.argsort(block)[::-1]
            if len(order) < 2:
                continue
            x_alt = x_base.copy()
            snap_alt = np.zeros(e - s)
            snap_alt[int(order[1])] = 1.0
            x_alt[s:e] = snap_alt
            candidates.append(x_alt)

        best_x, best_dist = None, np.inf
        for cand in candidates:
            if self._is_certified_candidate(cand, A_full, b_full, center, ball_eps, tol=tol):
                d_cand = float(np.linalg.norm(cand - x_query, ord=self.distance_norm))
                if d_cand < best_dist:
                    best_x, best_dist = cand, d_cand

        return best_x, best_dist

    def _polytope_aware_decode(
        self,
        x_star: np.ndarray,
        x_query: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        box_eps: float,
        ball_eps: float,
        maxiter: int,
        tol: float,
        fixed_dims: Optional[np.ndarray],
        incumbent_upper_bound: float = np.inf,
        top_k: int = 3,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, ProfileValue]]:
        """
        Decode a relaxed OHE solution to a certified discrete vertex.

        The fast path uses the existing heuristic snap. If that fails, an exact
        per-polytope discrete search is launched: full enumeration for tiny
        categorical products and branch-and-bound otherwise.
        """
        ohe_slices = self.ohe_slices or []
        product_size = self._ohe_product_size(ohe_slices)

        best_x, best_dist = self._heuristic_polytope_decode(
            x_star,
            x_query,
            A_full,
            b_full,
            center,
            ball_eps,
            top_k=top_k,
            tol=tol,
        )
        if best_x is not None:
            profile = self._build_decode_profile(
                mode="heuristic",
                exact_fallback_used=False,
                nodes_visited=0,
                nodes_pruned=0,
                solver_calls=0,
                product_size=product_size,
                heuristic_success=True,
            )
            return best_x, best_dist, profile

        incumbent = float(incumbent_upper_bound)
        if product_size <= _EXACT_ENUM_PRODUCT_THRESHOLD:
            return self._exact_polytope_decode_enumeration(
                x_star,
                x_query,
                A_full,
                b_full,
                center,
                box_eps,
                ball_eps,
                maxiter=maxiter,
                tol=tol,
                fixed_dims=fixed_dims,
                incumbent_upper_bound=incumbent,
            )
        return self._exact_polytope_decode_branch_and_bound(
            x_star,
            x_query,
            A_full,
            b_full,
            center,
            box_eps,
            ball_eps,
            maxiter=maxiter,
            tol=tol,
            fixed_dims=fixed_dims,
            incumbent_upper_bound=incumbent,
        )

    def _exact_polytope_decode_enumeration(
        self,
        x_star: np.ndarray,
        x_query: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        box_eps: float,
        ball_eps: float,
        maxiter: int,
        tol: float,
        fixed_dims: Optional[np.ndarray],
        incumbent_upper_bound: float,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, ProfileValue]]:
        ohe_slices = self.ohe_slices or []
        product_size = self._ohe_product_size(ohe_slices)
        profile = self._build_decode_profile(
            mode="exact_enum",
            exact_fallback_used=True,
            nodes_visited=0,
            nodes_pruned=0,
            solver_calls=0,
            product_size=product_size,
            heuristic_success=False,
        )

        root_dist = float(np.linalg.norm(x_star - x_query, ord=self.distance_norm))
        if root_dist >= incumbent_upper_bound:
            profile["decode_nodes_pruned"] = 1
            return None, np.inf, profile

        block_indices = list(range(len(ohe_slices)))
        category_orders = []
        for block_idx in block_indices:
            s, e = ohe_slices[block_idx]
            block = x_star[s:e]
            category_orders.append([int(i) for i in np.argsort(block)[::-1]])

        best_x = None
        best_dist = float(incumbent_upper_bound)
        for assignment in product(*category_orders):
            fixed_ohe_assignments = {block_idx: int(cat) for block_idx, cat in zip(block_indices, assignment)}
            profile["decode_nodes_visited"] = int(profile["decode_nodes_visited"]) + 1
            profile["decode_solver_calls"] = int(profile["decode_solver_calls"]) + 1
            x_leaf, dist_leaf = self._solve_projection_subproblem(
                x_query,
                A_full,
                b_full,
                center,
                box_eps,
                ball_eps,
                maxiter=maxiter,
                tol=tol,
                fixed_dims=fixed_dims,
                ohe_slices=ohe_slices,
                fixed_ohe_assignments=fixed_ohe_assignments,
            )
            if x_leaf is None or dist_leaf >= best_dist:
                profile["decode_nodes_pruned"] = int(profile["decode_nodes_pruned"]) + 1
                continue
            best_x = x_leaf
            best_dist = float(dist_leaf)

        if best_x is None:
            return None, np.inf, profile
        return best_x, best_dist, profile

    def _exact_polytope_decode_branch_and_bound(
        self,
        x_star: np.ndarray,
        x_query: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        box_eps: float,
        ball_eps: float,
        maxiter: int,
        tol: float,
        fixed_dims: Optional[np.ndarray],
        incumbent_upper_bound: float,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, ProfileValue]]:
        ohe_slices = self.ohe_slices or []
        product_size = self._ohe_product_size(ohe_slices)
        profile = self._build_decode_profile(
            mode="exact_bnb",
            exact_fallback_used=True,
            nodes_visited=0,
            nodes_pruned=0,
            solver_calls=0,
            product_size=product_size,
            heuristic_success=False,
        )

        block_indices = [i for i, (s, e) in enumerate(ohe_slices) if e - s > 1]
        if not block_indices:
            root_dist = float(np.linalg.norm(x_star - x_query, ord=self.distance_norm))
            if root_dist >= incumbent_upper_bound:
                profile["decode_nodes_pruned"] = 1
                return None, np.inf, profile
            return x_star.copy(), root_dist, profile

        root_dist = float(np.linalg.norm(x_star - x_query, ord=self.distance_norm))
        if root_dist >= incumbent_upper_bound:
            profile["decode_nodes_pruned"] = 1
            return None, np.inf, profile

        pq: List[Tuple[float, int, Dict[int, int], np.ndarray]] = []
        ticket = count()
        heapq.heappush(pq, (root_dist, next(ticket), {}, x_star.copy()))

        best_x = None
        best_dist = float(incumbent_upper_bound)

        while pq:
            lower_bound, _, fixed_assignments, relaxed_point = heapq.heappop(pq)
            if lower_bound >= best_dist:
                profile["decode_nodes_pruned"] = int(profile["decode_nodes_pruned"]) + 1
                continue

            if len(fixed_assignments) == len(block_indices):
                best_x = relaxed_point
                best_dist = float(lower_bound)
                continue

            branch_block = None
            branch_margin = np.inf
            for block_idx in block_indices:
                if block_idx in fixed_assignments:
                    continue
                s, e = ohe_slices[block_idx]
                block = relaxed_point[s:e]
                if len(block) < 2:
                    branch_block = block_idx
                    branch_margin = -np.inf
                    break
                sorted_vals = np.sort(block)[::-1]
                margin = float(sorted_vals[0] - sorted_vals[1])
                if margin < branch_margin:
                    branch_margin = margin
                    branch_block = block_idx

            if branch_block is None:
                profile["decode_nodes_pruned"] = int(profile["decode_nodes_pruned"]) + 1
                continue

            s, e = ohe_slices[branch_block]
            category_order = [int(i) for i in np.argsort(relaxed_point[s:e])[::-1]]
            for category in category_order:
                child_assignments = dict(fixed_assignments)
                child_assignments[branch_block] = category
                profile["decode_nodes_visited"] = int(profile["decode_nodes_visited"]) + 1
                profile["decode_solver_calls"] = int(profile["decode_solver_calls"]) + 1
                child_x, child_dist = self._solve_projection_subproblem(
                    x_query,
                    A_full,
                    b_full,
                    center,
                    box_eps,
                    ball_eps,
                    maxiter=maxiter,
                    tol=tol,
                    fixed_dims=fixed_dims,
                    ohe_slices=ohe_slices,
                    fixed_ohe_assignments=child_assignments,
                )
                if child_x is None or child_dist >= best_dist:
                    profile["decode_nodes_pruned"] = int(profile["decode_nodes_pruned"]) + 1
                    continue
                heapq.heappush(
                    pq,
                    (float(child_dist), next(ticket), child_assignments, child_x),
                )

        if best_x is None:
            return None, np.inf, profile
        return best_x, best_dist, profile

    def _project_onto_polytope(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        center: np.ndarray,
        eps_i: float,
        delta: float = 0.0,
        robust_norm: Optional[Union[int, float]] = None,
        maxiter: Optional[int] = None,
        tol: Optional[float] = None,
        fixed_dims: Optional[np.ndarray] = None,
        incumbent_upper_bound: float = np.inf,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, ProfileValue]]:
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
            profile = self._build_decode_profile(
                mode="heuristic",
                exact_fallback_used=False,
                nodes_visited=0,
                nodes_pruned=0,
                solver_calls=0,
                product_size=self._ohe_product_size(self.ohe_slices),
                heuristic_success=False,
            )
            return None, np.inf, profile

        x_proj, dist = self._solve_projection_subproblem(
            x0,
            A_full,
            b_full,
            center,
            box_eps,
            ball_eps,
            maxiter=maxiter,
            tol=tol,
            fixed_dims=fixed_dims,
            ohe_slices=self.ohe_slices,
            fixed_ohe_assignments=None,
        )

        if x_proj is None:
            profile = self._build_decode_profile(
                mode="heuristic",
                exact_fallback_used=False,
                nodes_visited=0,
                nodes_pruned=0,
                solver_calls=0,
                product_size=self._ohe_product_size(self.ohe_slices),
                heuristic_success=False,
            )
            return None, np.inf, profile

        if self.ohe_slices:
            x_proj, dist, decode_profile = self._polytope_aware_decode(
                x_proj,
                x0,
                A_full,
                b_full,
                center,
                box_eps,
                ball_eps,
                maxiter=maxiter,
                tol=tol,
                fixed_dims=fixed_dims,
                incumbent_upper_bound=incumbent_upper_bound,
            )
        else:
            decode_profile = self._build_decode_profile(
                mode="heuristic",
                exact_fallback_used=False,
                nodes_visited=0,
                nodes_pruned=0,
                solver_calls=0,
                product_size=1,
                heuristic_success=True,
            )

        return x_proj, dist, decode_profile

    def _make_project_fn(self, x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims):
        """Return a timed projection closure plus decode profiling accumulators."""
        projection_time_s = [0.0]
        best_projection_dist = [np.inf]
        profile_recorded = [False]
        best_decode_profile = [self._build_decode_profile(
            mode="heuristic",
            exact_fallback_used=False,
            nodes_visited=0,
            nodes_pruned=0,
            solver_calls=0,
            product_size=self._ohe_product_size(self.ohe_slices),
            heuristic_success=not bool(self.ohe_slices),
        )]

        def project_fn(idx: int, incumbent_upper_bound: float) -> Tuple[Optional[np.ndarray], float]:
            t0 = time.perf_counter()
            x_proj, dist, decode_profile = self._project_onto_polytope(
                x_query, bd['lA'][idx], bd['lbias'][idx], bd['X'][idx],
                eps_i=float(bd['eps'][idx]),
                delta=delta, robust_norm=robust_norm,
                maxiter=solver_maxiter, fixed_dims=fixed_dims,
                incumbent_upper_bound=incumbent_upper_bound,
            )
            projection_time_s[0] += time.perf_counter() - t0
            improved_dist = dist < best_projection_dist[0]
            if (
                improved_dist
                or not profile_recorded[0]
                or (
                    not np.isfinite(best_projection_dist[0])
                    and bool(decode_profile.get("decode_exact_fallback_used", False))
                    and (
                        not bool(best_decode_profile[0].get("decode_exact_fallback_used", False))
                        or int(decode_profile.get("decode_solver_calls", 0))
                        > int(best_decode_profile[0].get("decode_solver_calls", 0))
                    )
                )
            ):
                best_decode_profile[0] = decode_profile
                profile_recorded[0] = True
            if improved_dist:
                best_projection_dist[0] = dist
            return x_proj, dist

        return project_fn, projection_time_s, best_decode_profile

    def _search_bvh(self, x_query, bd, target_class, delta, robust_norm, solver_maxiter, fixed_dims):
        """Branch-and-bound BVH search. Returns (x_cf, dist, anchor_idx, n_qp, profiling_dict)."""
        project_fn, projection_time_s, best_decode_profile = self._make_project_fn(
            x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims)
        bvh_stats: Dict[str, float] = {}
        bvh = self.bvh_indices[target_class]
        t0 = time.perf_counter()
        x_cf, dist, anchor_idx, n_qp = bvh.query_nearest(
            x_query, project_fn, distance_norm=self.distance_norm, stats_out=bvh_stats
        )
        query_loop_time_s = time.perf_counter() - t0
        profiling = {
            "query_loop_time_ms": 1e3 * query_loop_time_s,
            "search_time_ms": 1e3 * max(0.0, query_loop_time_s - projection_time_s[0]),
            "projection_time_ms": 1e3 * projection_time_s[0],
            "distance_norm": float(self.distance_norm) if self.distance_norm == np.inf else int(self.distance_norm),
            "n_nodes_popped": bvh_stats.get("n_nodes_popped", np.nan),
            "n_nodes_pruned": bvh_stats.get("n_nodes_pruned", np.nan),
            "n_leaves_visited": bvh_stats.get("n_leaves_visited", np.nan),
            "max_queue_size": bvh_stats.get("max_queue_size", np.nan),
            "n_candidates_considered": bvh_stats.get("n_candidates_considered", np.nan),
        }
        profiling.update(best_decode_profile[0])
        return x_cf, dist, anchor_idx, n_qp, profiling

    def _search_sorted(self, x_query, bd, target_class, delta, robust_norm, solver_maxiter, fixed_dims):
        """Sorted lower-bound scan. Returns (x_cf, dist, anchor_idx, n_qp, profiling_dict)."""
        project_fn, projection_time_s, best_decode_profile = self._make_project_fn(
            x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims)
        sorted_stats: Dict[str, float] = {}
        bvh = self.bvh_indices[target_class]
        t0 = time.perf_counter()
        x_cf, dist, anchor_idx, n_qp = bvh.query_sorted_lower_bounds(
            x_query, eps_array=bd['eps'], project_fn=project_fn,
            distance_norm=self.distance_norm,
            stats_out=sorted_stats,
        )
        query_loop_time_s = time.perf_counter() - t0
        profiling = {
            "query_loop_time_ms": 1e3 * query_loop_time_s,
            "search_time_ms": 1e3 * max(0.0, query_loop_time_s - projection_time_s[0]),
            "projection_time_ms": 1e3 * projection_time_s[0],
            "distance_norm": float(self.distance_norm) if self.distance_norm == np.inf else int(self.distance_norm),
            "n_candidates_considered": sorted_stats.get("n_candidates_considered", np.nan),
            "n_candidates_total": sorted_stats.get("n_candidates_total", np.nan),
            "n_candidates_pruned_by_bound": sorted_stats.get("n_candidates_pruned_by_bound", np.nan),
            "best_lower_bound_at_termination": sorted_stats.get("best_lower_bound_at_termination", np.nan),
        }
        profiling.update(best_decode_profile[0])
        return x_cf, dist, anchor_idx, n_qp, profiling

    def _search_nearest_anchor(
        self,
        x_query,
        bd,
        target_class,
        delta,
        robust_norm,
        solver_maxiter,
        fixed_dims,
        query_k_candidates: int,
    ):
        """Project only onto the top-k nearest target-class anchors.

        This is an approximate, fixed-budget query strategy. It preserves
        certified validity for any returned point because projection is still
        performed against certified target-class polytopes, but it does not
        guarantee the closest point over the full atlas.
        """
        project_fn, projection_time_s, best_decode_profile = self._make_project_fn(
            x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims)
        bvh = self.bvh_indices[target_class]
        k = int(query_k_candidates)
        if k <= 0:
            raise ValueError("query_k_candidates must be positive for method='nearest_anchor'.")

        t0 = time.perf_counter()
        sorted_candidate_indices = bvh.query_k_nearest_candidates(
            x_query,
            k=bvh.n_polytopes,
            distance_norm=self.distance_norm,
        )
        primary_indices = sorted_candidate_indices[:k]
        fallback_indices = sorted_candidate_indices[k:]

        best_point: Optional[np.ndarray] = None
        best_dist = np.inf
        best_idx: Optional[int] = None
        n_qp = 0
        fallback_used = False
        for idx in primary_indices:
            point, dist = project_fn(int(idx), best_dist)
            n_qp += 1
            if dist < best_dist:
                best_point = point
                best_dist = dist
                best_idx = int(idx)

        # If the fixed top-k budget finds no feasible certified projection,
        # keep the method useful by scanning remaining anchors in nearest-anchor
        # order until the first feasible projection is found. This fallback is
        # only paid on failures; successful top-k queries keep a fixed QP budget.
        if best_point is None:
            fallback_used = True
            for idx in fallback_indices:
                point, dist = project_fn(int(idx), best_dist)
                n_qp += 1
                if point is not None and np.isfinite(dist):
                    best_point = point
                    best_dist = dist
                    best_idx = int(idx)
                    break

        query_loop_time_s = time.perf_counter() - t0
        profiling = {
            "query_loop_time_ms": 1e3 * query_loop_time_s,
            "search_time_ms": 1e3 * max(0.0, query_loop_time_s - projection_time_s[0]),
            "projection_time_ms": 1e3 * projection_time_s[0],
            "distance_norm": float(self.distance_norm) if self.distance_norm == np.inf else int(self.distance_norm),
            "query_k_candidates": float(k),
            "nearest_anchor_fallback_used": float(fallback_used),
            "n_candidates_considered": float(n_qp),
            "n_candidates_total": float(bvh.n_polytopes),
            "n_candidates_pruned_by_top_k": float(max(0, bvh.n_polytopes - n_qp)),
        }
        profiling.update(best_decode_profile[0])
        return best_point, best_dist, best_idx, n_qp, profiling

    def find_counterfactual(
        self,
        x_query: np.ndarray,
        target_class: int,
        method: Optional[str] = None,
        delta: float = 0.0,
        robust_norm: Optional[Union[int, float, str]] = None,
        solver_maxiter: Optional[int] = None,
        fixed_dims: Optional[np.ndarray] = None,
        query_k_candidates: int = 1,
    ) -> CounterfactualResult:
        """Find the closest counterfactual for a query point."""

        if self.bounds is None:
            raise ValueError("Atlas not built. Call build() first.")

        target_class = int(target_class)
        if target_class not in self.bounds:
            raise ValueError(
                f"Unknown target_class {target_class}. Available classes: {self.class_labels}"
            )
        x_query = np.asarray(x_query).flatten()
        if robust_norm is not None:
            robust_norm = self._normalize_lp_norm(robust_norm)
        bd = self.bounds[target_class]
        resolved_method = method or self.default_query_method
        profiling: Dict[str, ProfileValue] = {
            "method": resolved_method,
            "delta": float(delta),
            "robust_norm": (
                float(robust_norm) if robust_norm == np.inf else int(robust_norm)
            ) if robust_norm is not None else (
                float(self.norm) if self.norm == np.inf else int(self.norm)
            ),
        }
        t_total_start = time.perf_counter()

        if resolved_method == 'bvh':
            x_cf, dist, anchor_idx, n_qp, search_profiling = self._search_bvh(
                x_query, bd, target_class, delta, robust_norm, solver_maxiter, fixed_dims)
        elif resolved_method == 'sorted':
            x_cf, dist, anchor_idx, n_qp, search_profiling = self._search_sorted(
                x_query, bd, target_class, delta, robust_norm, solver_maxiter, fixed_dims)
        elif resolved_method == 'nearest_anchor':
            x_cf, dist, anchor_idx, n_qp, search_profiling = self._search_nearest_anchor(
                x_query, bd, target_class, delta, robust_norm, solver_maxiter,
                fixed_dims, query_k_candidates)
        else:
            raise ValueError(f"Unknown method: {resolved_method}. Use 'sorted', 'bvh', or 'nearest_anchor'.")

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
        robust_norm: Optional[Union[int, float, str]] = None,
        solver_maxiter: Optional[int] = None,
        fixed_dims: Optional[np.ndarray] = None,
        query_k_candidates: int = 1,
    ) -> List[CounterfactualResult]:
        """Find counterfactuals for a batch of query points."""
        results = []
        for x in X_query:
            results.append(self.find_counterfactual(
                x, target_class, method,
                delta=delta,
                robust_norm=robust_norm,
                solver_maxiter=solver_maxiter,
                fixed_dims=fixed_dims,
                query_k_candidates=query_k_candidates,
            ))
        return results

    def verify_counterfactual(
        self,
        x_cf: np.ndarray,
        target_class: int,
        *,
        delta: float = 0.0,
        robust_norm: Optional[Union[int, float]] = None,
        anchor_idx: Optional[int] = None,
    ) -> Dict:
        """
        Verify model prediction, structural validity, and certified polytope membership.

        Parameters
        ----------
        x_cf : np.ndarray
            The counterfactual point, shape (d,).
        target_class : int
            The expected target class.

        Returns
        -------
        dict
            Dictionary with the model prediction plus certification diagnostics.
        """
        if self.bounds is None:
            raise ValueError("Atlas not built. Call build() first.")

        target_class = int(target_class)
        if target_class not in self.bounds:
            raise ValueError(
                f"Unknown target_class {target_class}. Available classes: {self.class_labels}"
            )
        if robust_norm is None:
            robust_norm = self.norm
        else:
            robust_norm = self._normalize_lp_norm(robust_norm)

        self.model.eval()
        x_arr = np.asarray(x_cf, dtype=np.float32).reshape(-1)
        x_tensor = self._reshape_model_input(x_arr)

        with torch.no_grad():
            logits = self.model(x_tensor).cpu().numpy()[0]
            pred_index = int(np.argmax(logits))
            predicted = int(self.class_labels[pred_index])

        prediction_valid = predicted == target_class
        ohe_valid = self._is_ohe_valid(x_arr)
        bd = self.bounds[target_class]

        if anchor_idx is not None:
            anchor_idx = int(anchor_idx)
            n_anchors = int(len(bd['X']))
            if anchor_idx < 0 or anchor_idx >= n_anchors:
                raise ValueError(
                    f"anchor_idx={anchor_idx} is out of bounds for target_class {target_class} "
                    f"(n_anchors={n_anchors})."
                )
            in_polytope, robust_polytope = self._polytope_membership_for_anchor(
                x_arr,
                bd,
                anchor_idx,
                delta=delta,
                robust_norm=robust_norm,
            )
            verified_anchor_idx = anchor_idx if (robust_polytope if delta > 0.0 else in_polytope) else None
        else:
            in_polytope = False
            robust_polytope = False
            verified_anchor_idx = None
            for idx in range(len(bd['X'])):
                nominal_ok, robust_ok = self._polytope_membership_for_anchor(
                    x_arr,
                    bd,
                    idx,
                    delta=delta,
                    robust_norm=robust_norm,
                )
                if nominal_ok:
                    in_polytope = True
                active_ok = robust_ok if delta > 0.0 else nominal_ok
                if active_ok:
                    robust_polytope = robust_ok if delta > 0.0 else nominal_ok
                    verified_anchor_idx = idx
                    break
            if delta <= 0.0:
                robust_polytope = in_polytope

        overall_valid = prediction_valid and ohe_valid and in_polytope and robust_polytope

        return {
            'predicted': predicted,
            'target': target_class,
            'valid': overall_valid,
            'logits': logits,
            'prediction_valid': prediction_valid,
            'ohe_valid': ohe_valid,
            'in_certified_polytope': in_polytope,
            'robustly_certified': robust_polytope,
            'overall_valid': overall_valid,
            'verified_anchor_idx': verified_anchor_idx,
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
            return "CertCFAtlas (not built)"

        all_eps = np.concatenate([self.bounds[label]['eps'] for label in self.class_labels])
        eps_desc = (
            f"eps in [{all_eps.min():.4g}, {all_eps.max():.4g}]"
            f" ({type(self.eps_strategy).__name__})"
        )

        lines = [
            f"CertCFAtlas Summary",
            f"=" * 40,
            f"Classes: {self.n_classes}",
            f"Perturbation: {eps_desc}, L{self.norm} norm",
            f"Device: {self.device}",
            f""
        ]

        total_polytopes = 0
        for label in self.class_labels:
            n = self.bvh_indices[label].n_polytopes
            total_polytopes += n
            depth = self.bvh_indices[label].tree_depth
            lines.append(f"  Class {label}: {n:4d} polytopes (BVH depth {depth})")

        lines.append(f"")
        lines.append(f"Total polytopes: {total_polytopes}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "built" if self.bounds is not None else "not built"
        return f"CertCFAtlas(n_classes={self.n_classes}, {status})"
