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
import json
import multiprocessing as mp
import os
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from itertools import count, product
import numpy as np
from copy import deepcopy

_SOLVER_TOL = 1e-9  # QP convergence tolerance (fixed)
_PROJECTION_FEASIBILITY_TOL = 1e-6  # Numerical post-solve feasibility tolerance.
import torch
import torch.nn as nn
from scipy.optimize import minimize
from typing import Any, Optional, Dict, List, Sequence, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .certification.lirpa import PreimageApproximation
from .eps_strategies import EpsStrategy, ConstantEpsStrategy
from .geometry.polytopes import ball_box_constraints
from .geometry.operations import build_class_union, assert_no_cross_class_overlap
from .indexing.bvh import BVHIndex


_EXACT_ENUM_PRODUCT_THRESHOLD = 4096
_ALLOWED_OHE_DECODE_MODES = {"exact", "beam_then_exact", "beam_only"}
ProfileValue = Union[float, str, int, bool]
_CANDIDATE_PARALLEL_BACKENDS = {"thread", "process"}
_PROCESS_PROJECTION_ATLAS: Optional["CertCFAtlas"] = None


def _run_candidate_projection_in_process(task: Tuple[Any, ...]) -> "_CandidateProjection":
    """Process-pool entry point using a fork-inherited projection-only atlas."""
    if _PROCESS_PROJECTION_ATLAS is None:
        raise RuntimeError("Candidate projection worker was not initialized")
    (
        x_query,
        target_class,
        idx,
        delta,
        robust_norm,
        solver_maxiter,
        fixed_dims,
        nondecreasing_dims,
        nonincreasing_dims,
        incumbent_upper_bound,
    ) = task
    atlas = _PROCESS_PROJECTION_ATLAS
    return atlas._project_candidate(
        x_query=x_query,
        bd=atlas.bounds[int(target_class)],
        idx=int(idx),
        delta=float(delta),
        robust_norm=robust_norm,
        solver_maxiter=solver_maxiter,
        fixed_dims=fixed_dims,
        nondecreasing_dims=nondecreasing_dims,
        nonincreasing_dims=nonincreasing_dims,
        incumbent_upper_bound=float(incumbent_upper_bound),
    )


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


@dataclass
class _CandidateProjection:
    """Internal result of one independently timed candidate projection."""

    index: int
    point: Optional[np.ndarray]
    selection_score: float
    profile: Dict[str, ProfileValue]
    elapsed_s: float


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
        model_input_shape: Optional[Sequence[int]] = None,
        reuse_lirpa_graph: bool = False,
        bounds_checkpoint_dir: Optional[Union[str, Path]] = None,
        # Build configuration
        norm: int = 2,
        distance_norm: Optional[Union[int, float]] = None,
        lirpa_method: str = "backward",
        eps_strategy: Optional[EpsStrategy] = None,
        batch_size: Optional[int] = None,
        epsilon_parallelism: int = 1,
        build_parallelism: int = 1,
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
        input_bounds: Optional[Sequence[float]] = None,
        # Query configuration
        default_query_method: str = "sorted",
        solver_maxiter: int = 500,
        query_parallelism: int = 1,
        candidate_parallelism: int = 1,
        candidate_parallel_backend: str = "thread",
        cvxpy_solvers: Optional[List[str]] = None,
        cvxpy_solver_options: Optional[Dict[str, Dict[str, Any]]] = None,
        cvxpy_accept_statuses: Optional[Dict[str, List[str]]] = None,
        classification_margin: float = 0.0,
        adaptive_eps: bool = False,
        adaptive_eps_shrink_factor: float = 0.5,
        adaptive_eps_max_shrinks: int = 8,
        adaptive_eps_min: float = 1.0e-6,
        adaptive_eps_center_tol: float = 1.0e-6,
        adaptive_eps_binary_search_steps: int = 0,
        ohe_decode_mode: str = "exact",
        decode_beam_width: int = 8,
        decode_beam_branch_top_k: int = 3,
        decode_beam_max_solver_calls: int = 32,
        sparsity_penalty: str = "none",
        sparsity_lambda: float = 0.0,
        sparsity_reweight_iters: int = 0,
        sparsity_eps: float = 1.0e-3,
        sparsity_group_ohe: bool = True,
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
        self.epsilon_parallelism = int(epsilon_parallelism)
        if self.epsilon_parallelism <= 0:
            raise ValueError("epsilon_parallelism must be positive")
        self.build_parallelism = int(build_parallelism)
        if self.build_parallelism <= 0:
            raise ValueError("build_parallelism must be positive")
        self.ohe_slices = ohe_slices
        self.input_bounds = self._normalize_input_bounds(input_bounds)
        self.bounds_checkpoint_dir = (
            None if bounds_checkpoint_dir is None else Path(bounds_checkpoint_dir)
        )

        self.solver_maxiter = solver_maxiter
        self.query_parallelism = int(query_parallelism)
        if self.query_parallelism <= 0:
            raise ValueError("query_parallelism must be positive")
        self.candidate_parallelism = int(candidate_parallelism)
        if self.candidate_parallelism <= 0:
            raise ValueError("candidate_parallelism must be positive")
        if self.query_parallelism > 1 and self.candidate_parallelism > 1:
            raise ValueError(
                "query_parallelism and candidate_parallelism cannot both exceed 1; "
                "choose inter-query or intra-query parallelism to avoid nested worker pools"
            )
        self.candidate_parallel_backend = str(candidate_parallel_backend).lower()
        if self.candidate_parallel_backend not in _CANDIDATE_PARALLEL_BACKENDS:
            raise ValueError(
                "candidate_parallel_backend must be one of {'thread', 'process'}"
            )
        self._candidate_process_pool = None
        self._candidate_process_pool_workers = 0
        (
            self.cvxpy_solvers,
            self.cvxpy_solver_options,
            self.cvxpy_accept_statuses,
        ) = self._normalize_cvxpy_solver_config(
            cvxpy_solvers=cvxpy_solvers,
            cvxpy_solver_options=cvxpy_solver_options,
            cvxpy_accept_statuses=cvxpy_accept_statuses,
        )
        self.classification_margin = float(classification_margin)
        if self.classification_margin < 0.0:
            raise ValueError("classification_margin must be non-negative")
        self.adaptive_eps = bool(adaptive_eps)
        self.adaptive_eps_shrink_factor = float(adaptive_eps_shrink_factor)
        if not (0.0 < self.adaptive_eps_shrink_factor < 1.0):
            raise ValueError("adaptive_eps_shrink_factor must be in (0, 1)")
        self.adaptive_eps_max_shrinks = int(adaptive_eps_max_shrinks)
        if self.adaptive_eps_max_shrinks < 0:
            raise ValueError("adaptive_eps_max_shrinks must be non-negative")
        self.adaptive_eps_min = float(adaptive_eps_min)
        if self.adaptive_eps_min < 0.0:
            raise ValueError("adaptive_eps_min must be non-negative")
        self.adaptive_eps_center_tol = float(adaptive_eps_center_tol)
        if self.adaptive_eps_center_tol < 0.0:
            raise ValueError("adaptive_eps_center_tol must be non-negative")
        self.adaptive_eps_binary_search_steps = int(adaptive_eps_binary_search_steps)
        if self.adaptive_eps_binary_search_steps < 0:
            raise ValueError("adaptive_eps_binary_search_steps must be non-negative")
        (
            self.ohe_decode_mode,
            self.decode_beam_width,
            self.decode_beam_branch_top_k,
            self.decode_beam_max_solver_calls,
        ) = self._normalize_ohe_decode_config(
            ohe_decode_mode=ohe_decode_mode,
            decode_beam_width=decode_beam_width,
            decode_beam_branch_top_k=decode_beam_branch_top_k,
            decode_beam_max_solver_calls=decode_beam_max_solver_calls,
        )
        (
            self.sparsity_penalty,
            self.sparsity_lambda,
            self.sparsity_reweight_iters,
            self.sparsity_eps,
            self.sparsity_group_ohe,
        ) = self._normalize_sparsity_config(
            sparsity_penalty=sparsity_penalty,
            sparsity_lambda=sparsity_lambda,
            sparsity_reweight_iters=sparsity_reweight_iters,
            sparsity_eps=sparsity_eps,
            sparsity_group_ohe=sparsity_group_ohe,
            distance_norm=self.distance_norm,
        )

        allowed_methods = {"sorted", "bvh", "nearest_anchor"}
        if default_query_method not in allowed_methods:
            raise ValueError(
                f"default_query_method must be one of {allowed_methods}, got {default_query_method!r}"
            )
        self.default_query_method = default_query_method


        # Initialize preimage approximation handler
        self._preimage = PreimageApproximation(
            model,
            dataset,
            self.device,
            cnn=cnn,
            model_input_shape=model_input_shape,
            reuse_lirpa_graph=reuse_lirpa_graph,
        )
        self.n_classes = self._preimage.n_classes
        self.class_labels = list(self._preimage.class_labels)
        self.label_to_index = dict(self._preimage.label_to_index)
        sample_shape = tuple(self._preimage.dataset.tensors[0][0].shape)
        if model_input_shape is not None:
            self.model_input_shape = tuple(int(v) for v in model_input_shape)
            if int(np.prod(self.model_input_shape)) != int(np.prod(sample_shape)):
                raise ValueError("model_input_shape does not match the sample size")
        elif self.cnn and len(sample_shape) == 1:
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
    def _default_cvxpy_solvers() -> List[str]:
        return ["CLARABEL", "SCS"]

    @staticmethod
    def _default_cvxpy_accept_statuses() -> Dict[str, List[str]]:
        return {
            "CLARABEL": ["optimal"],
            "SCS": ["optimal", "optimal_inaccurate"],
        }

    @staticmethod
    def _normalize_cvxpy_solver_name(solver_name: str) -> str:
        if not isinstance(solver_name, str):
            raise ValueError("cvxpy solver names must be strings")
        normalized = solver_name.strip().upper()
        if not normalized:
            raise ValueError("cvxpy solver names must be non-empty strings")
        return normalized

    @classmethod
    def _normalize_cvxpy_solver_config(
        cls,
        *,
        cvxpy_solvers: Optional[List[str]],
        cvxpy_solver_options: Optional[Dict[str, Dict[str, Any]]],
        cvxpy_accept_statuses: Optional[Dict[str, List[str]]],
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
        raw_solvers = cls._default_cvxpy_solvers() if cvxpy_solvers is None else list(cvxpy_solvers)
        normalized_solvers: List[str] = []
        seen_solvers = set()
        for solver_name in raw_solvers:
            normalized = cls._normalize_cvxpy_solver_name(solver_name)
            if normalized in seen_solvers:
                raise ValueError("cvxpy_solvers must be unique after normalization")
            seen_solvers.add(normalized)
            normalized_solvers.append(normalized)

        normalized_options: Dict[str, Dict[str, Any]] = {}
        if cvxpy_solver_options is not None:
            if not isinstance(cvxpy_solver_options, dict):
                raise ValueError("cvxpy_solver_options must be a dictionary")
            for solver_name, solver_options in cvxpy_solver_options.items():
                normalized_solver = cls._normalize_cvxpy_solver_name(solver_name)
                if normalized_solver not in seen_solvers:
                    raise ValueError("cvxpy_solver_options may only reference configured solvers")
                if not isinstance(solver_options, dict):
                    raise ValueError("cvxpy_solver_options entries must be dictionaries")
                normalized_options[normalized_solver] = deepcopy(solver_options)

        default_statuses = {
            solver: list(cls._default_cvxpy_accept_statuses().get(solver, ["optimal"]))
            for solver in normalized_solvers
        }
        if cvxpy_accept_statuses is not None:
            if not isinstance(cvxpy_accept_statuses, dict):
                raise ValueError("cvxpy_accept_statuses must be a dictionary")
            for solver_name, statuses in cvxpy_accept_statuses.items():
                normalized_solver = cls._normalize_cvxpy_solver_name(solver_name)
                if normalized_solver not in seen_solvers:
                    raise ValueError("cvxpy_accept_statuses may only reference configured solvers")
                if not isinstance(statuses, (list, tuple)) or len(statuses) == 0:
                    raise ValueError("cvxpy_accept_statuses entries must be non-empty lists of strings")
                normalized_statuses = []
                for status in statuses:
                    if not isinstance(status, str) or not status.strip():
                        raise ValueError("cvxpy_accept_statuses entries must be non-empty lists of strings")
                    normalized_statuses.append(status.strip())
                default_statuses[normalized_solver] = normalized_statuses

        return normalized_solvers, normalized_options, default_statuses

    @staticmethod
    def _default_cvxpy_solver_profile() -> Dict[str, ProfileValue]:
        return {
            "cvxpy_solver_used": "",
            "cvxpy_solver_status": "",
            "cvxpy_solver_attempts": 0,
            "cvxpy_fallback_used": False,
        }

    @classmethod
    def _normalize_ohe_decode_config(
        cls,
        *,
        ohe_decode_mode: str,
        decode_beam_width: int,
        decode_beam_branch_top_k: int,
        decode_beam_max_solver_calls: int,
    ) -> Tuple[str, int, int, int]:
        if not isinstance(ohe_decode_mode, str):
            raise ValueError("ohe_decode_mode must be a string")
        normalized_mode = ohe_decode_mode.strip().lower()
        if normalized_mode not in _ALLOWED_OHE_DECODE_MODES:
            raise ValueError(
                f"ohe_decode_mode must be one of {_ALLOWED_OHE_DECODE_MODES}, got {ohe_decode_mode!r}"
            )

        normalized_ints = []
        for field_name, raw_value in (
            ("decode_beam_width", decode_beam_width),
            ("decode_beam_branch_top_k", decode_beam_branch_top_k),
            ("decode_beam_max_solver_calls", decode_beam_max_solver_calls),
        ):
            value = int(raw_value)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            normalized_ints.append(value)

        return normalized_mode, normalized_ints[0], normalized_ints[1], normalized_ints[2]

    @staticmethod
    def _normalize_sparsity_config(
        *,
        sparsity_penalty: str,
        sparsity_lambda: float,
        sparsity_reweight_iters: int,
        sparsity_eps: float,
        sparsity_group_ohe: bool,
        distance_norm: Union[int, float],
    ) -> Tuple[str, float, int, float, bool]:
        if not isinstance(sparsity_penalty, str):
            raise ValueError("sparsity_penalty must be a string")
        penalty = sparsity_penalty.strip().lower()
        if penalty in {"", "none", "off", "false"}:
            penalty = "none"
        if penalty not in {"none", "reweighted_l1"}:
            raise ValueError("sparsity_penalty must be one of {'none', 'reweighted_l1'}")

        lambda_value = float(sparsity_lambda)
        if lambda_value < 0.0:
            raise ValueError("sparsity_lambda must be non-negative")
        reweight_iters = int(sparsity_reweight_iters)
        if reweight_iters < 0:
            raise ValueError("sparsity_reweight_iters must be non-negative")
        eps_value = float(sparsity_eps)
        if eps_value <= 0.0:
            raise ValueError("sparsity_eps must be positive")
        group_ohe = bool(sparsity_group_ohe)

        if penalty == "reweighted_l1":
            if distance_norm != 1:
                raise ValueError("sparsity_penalty='reweighted_l1' requires distance_norm=1")
            if not CVXPY_AVAILABLE:
                raise ValueError("sparsity_penalty='reweighted_l1' requires CVXPY")

        return penalty, lambda_value, reweight_iters, eps_value, group_ohe

    def _sparsity_active(self) -> bool:
        return (
            getattr(self, "sparsity_penalty", "none") == "reweighted_l1"
            and getattr(self, "sparsity_lambda", 0.0) > 0.0
            and getattr(self, "sparsity_reweight_iters", 0) > 0
        )

    def _sparsity_metadata_defaults(self) -> Dict[str, ProfileValue]:
        return {
            "sparsity_penalty": getattr(self, "sparsity_penalty", "none"),
            "sparsity_lambda": float(getattr(self, "sparsity_lambda", 0.0)),
            "sparsity_reweight_iters": int(getattr(self, "sparsity_reweight_iters", 0)),
            "sparsity_eps": float(getattr(self, "sparsity_eps", 1.0e-3)),
            "sparsity_group_count": 0,
            "sparsity_active_groups": 0,
            "sparsity_selection_score": np.inf,
            "sparsity_solver_calls": 0,
        }

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

    @staticmethod
    def _normalize_input_bounds(
        input_bounds: Optional[Sequence[float]],
    ) -> Optional[Tuple[float, float]]:
        """Validate and normalize optional global bounds for every input dimension."""
        if input_bounds is None:
            return None
        if isinstance(input_bounds, (str, bytes)):
            raise ValueError("input_bounds must be a two-element sequence [lower, upper]")
        try:
            values = list(input_bounds)
        except TypeError as exc:
            raise ValueError(
                "input_bounds must be a two-element sequence [lower, upper]"
            ) from exc
        if len(values) != 2:
            raise ValueError("input_bounds must contain exactly two values [lower, upper]")
        try:
            lower, upper = (float(values[0]), float(values[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError("input_bounds values must be finite numbers") from exc
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("input_bounds values must be finite numbers")
        if lower > upper:
            raise ValueError("input_bounds lower value must not exceed upper value")
        return lower, upper

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
        epsilon_started = time.perf_counter()
        eps_array = eps_strategy.compute_eps(
            X_all,
            y_all,
            norm=norm,
            parallelism=self.epsilon_parallelism,
        )
        epsilon_time_s = time.perf_counter() - epsilon_started

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

        precomputed_bounds: Dict[int, Dict[str, np.ndarray]] = {}
        if self.bounds_checkpoint_dir is not None:
            self.bounds_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for label in self.class_labels:
                checkpoint = self.bounds_checkpoint_dir / f"class_{int(label)}.npz"
                if checkpoint.exists():
                    with np.load(checkpoint, allow_pickle=False) as data:
                        precomputed_bounds[int(label)] = {
                            key: data[key] for key in data.files
                        }
        remaining_bound_class_count = sum(
            int(label) not in precomputed_bounds for label in self.class_labels
        )

        def checkpoint_class(label: int, values: Dict[str, np.ndarray]) -> None:
            if self.bounds_checkpoint_dir is None:
                return
            target = self.bounds_checkpoint_dir / f"class_{int(label)}.npz"
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            with temporary.open("wb") as handle:
                np.savez(handle, **values)
            os.replace(temporary, target)

        lirpa_started = time.perf_counter()
        self.bounds = self._preimage.compute_all_bounds(
            eps=0.1,            # fallback scalar (unused when eps_array is provided)
            norm=norm,
            batch_size=self.batch_size,
            dtype=torch.float32,
            eps_array=eps_array,
            lirpa_method=self.lirpa_method,
            classification_margin=self.classification_margin,
            adaptive_eps=self.adaptive_eps,
            adaptive_eps_shrink_factor=self.adaptive_eps_shrink_factor,
            adaptive_eps_max_shrinks=self.adaptive_eps_max_shrinks,
            adaptive_eps_min=self.adaptive_eps_min,
            adaptive_eps_center_tol=self.adaptive_eps_center_tol,
            adaptive_eps_binary_search_steps=self.adaptive_eps_binary_search_steps,
            precomputed_bounds=precomputed_bounds,
            class_completed_callback=checkpoint_class,
            build_parallelism=self.build_parallelism,
        )
        lirpa_time_s = time.perf_counter() - lirpa_started

        # Only center-certified regions may enter the atlas.
        for label in self.class_labels:
            class_bounds = self.bounds[label]
            certified = np.asarray(
                class_bounds.get(
                    "adaptive_eps_center_certified",
                    np.ones(len(class_bounds["X"]), dtype=bool),
                ),
                dtype=bool,
            )
            original_count = int(len(certified))
            class_bounds["attempted_anchor_count"] = original_count
            class_bounds["uncertified_anchor_count"] = int(np.count_nonzero(~certified))
            if not np.all(certified):
                for key, value in list(class_bounds.items()):
                    if (
                        isinstance(value, np.ndarray)
                        and value.ndim > 0
                        and len(value) == original_count
                    ):
                        class_bounds[key] = value[certified]
            if len(class_bounds["X"]) == 0:
                raise RuntimeError(f"No certified atlas anchors remain for class {label}")

        # Step 2: Build BVH spatial index for each class
        if verbose:
            print("  Building BVH spatial indices...")

        bvh_started = time.perf_counter()
        self.bvh_indices = {}
        for label in self.class_labels:
            centers = self.bounds[label]['X']
            eps_class = self.bounds[label]['eps']
            self.bvh_indices[label] = BVHIndex(centers, eps_class)

            if verbose:
                bvh = self.bvh_indices[label]
                print(f"    Class {label}: {bvh.n_polytopes} polytopes, "
                      f"tree depth {bvh.tree_depth}")
        bvh_time_s = time.perf_counter() - bvh_started
        self.build_profiling = {
            "epsilon_time_s": float(epsilon_time_s),
            "lirpa_time_s": float(lirpa_time_s),
            "bvh_time_s": float(bvh_time_s),
            "epsilon_parallelism": int(self.epsilon_parallelism),
            "build_parallelism": int(self.build_parallelism),
            "lirpa_workers_used": int(
                min(self.build_parallelism, remaining_bound_class_count)
            ),
        }

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
                        label, self.bounds, self.bounds[label]['eps'], norm=self.norm
                    )
                self._validate_class_unions(tol=1e-9)

        if verbose:
            total_polytopes = sum(
                self.bvh_indices[l].n_polytopes for l in self.class_labels
            )
            print(f"Done! Total: {total_polytopes} polytopes across {self.n_classes} classes")

        return self

    def save_bounds(self, directory: Union[str, Path]) -> Path:
        """Persist built numeric atlas bounds without unsafe pickle payloads."""
        if self.bounds is None:
            raise RuntimeError("Cannot save an atlas before build()")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        files = {}
        for label in self.class_labels:
            target = root / f"class_{int(label)}.npz"
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            arrays = {
                key: np.asarray(value)
                for key, value in self.bounds[label].items()
                if isinstance(value, (np.ndarray, np.number, int, float, bool))
            }
            with temporary.open("wb") as handle:
                np.savez(handle, **arrays)
            os.replace(temporary, target)
            files[str(label)] = target.name
        manifest = {
            "format_version": 1,
            "class_labels": [int(label) for label in self.class_labels],
            "norm": "inf" if self.norm == np.inf else int(self.norm),
            "distance_norm": "inf" if self.distance_norm == np.inf else int(self.distance_norm),
            "model_input_shape": list(self.model_input_shape),
            "files": files,
        }
        target = root / "manifest.json"
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)
        return target

    def load_bounds(self, directory: Union[str, Path]) -> 'CertCFAtlas':
        """Load bounds saved by :meth:`save_bounds` and rebuild BVH indices."""
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        labels = [int(label) for label in manifest["class_labels"]]
        if labels != [int(label) for label in self.class_labels]:
            raise ValueError("Serialized atlas class labels do not match the attached dataset")
        expected_shape = tuple(int(v) for v in manifest["model_input_shape"])
        if expected_shape != tuple(self.model_input_shape):
            raise ValueError("Serialized atlas input shape does not match")
        self.bounds = {}
        self.bvh_indices = {}
        for label in labels:
            with np.load(root / manifest["files"][str(label)], allow_pickle=False) as data:
                self.bounds[label] = {key: data[key] for key in data.files}
            centers = self.bounds[label]["X"]
            eps = self.bounds[label]["eps"]
            self.bvh_indices[label] = BVHIndex(centers, eps)
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

        classification_margin = float(getattr(self, "classification_margin", 0.0))

        # Erode LiRPA constraints: A @ x + b >= margin + delta * ||a_i||_{q*}
        if delta > 0:
            q_dual = self._dual_norm(robust_norm)
            if q_dual == np.inf:
                row_dual_norms = np.max(np.abs(A), axis=1)
            elif q_dual == 1:
                row_dual_norms = np.sum(np.abs(A), axis=1)
            else:
                row_dual_norms = np.linalg.norm(A, axis=1, ord=q_dual)
            b_eroded = b - classification_margin - delta * row_dual_norms
        else:
            b_eroded = b - classification_margin

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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
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
        ref = self._apply_directional_bounds_to_point(
            ref,
            x0,
            nondecreasing_dims=nondecreasing_dims,
            nonincreasing_dims=nonincreasing_dims,
        )

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
        candidate = self._apply_directional_bounds_to_point(
            candidate,
            x0,
            nondecreasing_dims=nondecreasing_dims,
            nonincreasing_dims=nonincreasing_dims,
        )

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
            candidate = self._apply_directional_bounds_to_point(
                candidate,
                x0,
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
            )
            return candidate

        if np.min(A_full @ candidate + b_full) >= -1e-9:
            return candidate

        return ref

    @staticmethod
    def _apply_directional_bounds_to_point(
        point: np.ndarray,
        x_query: np.ndarray,
        *,
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Clip one point to query-relative directional halfspaces."""
        if nondecreasing_dims is None and nonincreasing_dims is None:
            return point
        clipped = np.asarray(point, dtype=np.float64).copy()
        x_query = np.asarray(x_query, dtype=np.float64).reshape(-1)
        if nondecreasing_dims is not None and len(nondecreasing_dims) > 0:
            clipped[nondecreasing_dims] = np.maximum(
                clipped[nondecreasing_dims],
                x_query[nondecreasing_dims],
            )
        if nonincreasing_dims is not None and len(nonincreasing_dims) > 0:
            clipped[nonincreasing_dims] = np.minimum(
                clipped[nonincreasing_dims],
                x_query[nonincreasing_dims],
            )
        return clipped

    @staticmethod
    def _directional_constraints_satisfied(
        point: np.ndarray,
        x_query: np.ndarray,
        *,
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        tol: float = _PROJECTION_FEASIBILITY_TOL,
    ) -> bool:
        """Check query-relative directional feasibility for one point."""
        point = np.asarray(point, dtype=np.float64).reshape(-1)
        x_query = np.asarray(x_query, dtype=np.float64).reshape(-1)
        if nondecreasing_dims is not None and len(nondecreasing_dims) > 0:
            if np.any(point[nondecreasing_dims] < x_query[nondecreasing_dims] - tol):
                return False
        if nonincreasing_dims is not None and len(nonincreasing_dims) > 0:
            if np.any(point[nonincreasing_dims] > x_query[nonincreasing_dims] + tol):
                return False
        return True

    def _build_sparsity_groups(
        self,
        d: int,
        fixed_dims: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        fixed_set = set() if fixed_dims is None else {int(i) for i in fixed_dims}
        grouped_dims: set[int] = set()
        groups: List[np.ndarray] = []

        if getattr(self, "sparsity_group_ohe", True) and self.ohe_slices:
            for start, end in self.ohe_slices:
                dims = [i for i in range(int(start), int(end)) if i not in fixed_set]
                grouped_dims.update(range(int(start), int(end)))
                if dims:
                    groups.append(np.asarray(dims, dtype=np.int64))

        for dim in range(int(d)):
            if dim in fixed_set or dim in grouped_dims:
                continue
            groups.append(np.asarray([dim], dtype=np.int64))

        return groups

    def _sparsity_group_changes(
        self,
        x: np.ndarray,
        x_query: np.ndarray,
        groups: Sequence[np.ndarray],
    ) -> np.ndarray:
        if not groups:
            return np.empty((0,), dtype=np.float64)
        diff = np.asarray(x, dtype=np.float64).reshape(-1) - np.asarray(x_query, dtype=np.float64).reshape(-1)
        return np.asarray([float(np.sum(np.abs(diff[group]))) for group in groups], dtype=np.float64)

    def _sparsity_surrogate_score(
        self,
        x: Optional[np.ndarray],
        x_query: np.ndarray,
        groups: Optional[Sequence[np.ndarray]] = None,
    ) -> float:
        if x is None:
            return np.inf
        x_arr = np.asarray(x, dtype=np.float64)
        x_query_arr = np.asarray(x_query, dtype=np.float64)
        true_l1 = float(np.linalg.norm(x_arr - x_query_arr, ord=1))
        if not self._sparsity_active():
            return float(np.linalg.norm(x_arr - x_query_arr, ord=getattr(self, "distance_norm", 1)))
        groups = groups or self._build_sparsity_groups(len(np.asarray(x).reshape(-1)))
        changes = self._sparsity_group_changes(x, x_query, groups)
        lambda_value = float(getattr(self, "sparsity_lambda", 0.0))
        eps_value = float(getattr(self, "sparsity_eps", 1.0e-3))
        return float(true_l1 + lambda_value * np.sum(changes / (changes + eps_value)))

    def _sparsity_group_weights(
        self,
        x: np.ndarray,
        x_query: np.ndarray,
        groups: Sequence[np.ndarray],
    ) -> np.ndarray:
        changes = self._sparsity_group_changes(x, x_query, groups)
        lambda_value = float(getattr(self, "sparsity_lambda", 0.0))
        eps_value = float(getattr(self, "sparsity_eps", 1.0e-3))
        return 1.0 + lambda_value / (changes + eps_value)

    @staticmethod
    def _score_from_profile(profile: Optional[Dict[str, ProfileValue]], fallback_dist: float) -> float:
        if profile is None:
            return float(fallback_dist)
        score = profile.get("sparsity_selection_score", fallback_dist)
        try:
            return float(score)
        except (TypeError, ValueError):
            return float(fallback_dist)

    @staticmethod
    def _ohe_product_size(ohe_slices: Optional[List[Tuple[int, int]]]) -> int:
        if not ohe_slices:
            return 1
        size = 1
        for s, e in ohe_slices:
            size *= max(1, int(e - s))
        return int(size)

    @staticmethod
    def _anchor_bbox_lower_bounds(
        x_query: np.ndarray,
        centers: np.ndarray,
        eps_array: np.ndarray,
        distance_norm: int | float,
    ) -> np.ndarray:
        diff = np.maximum(0.0, np.abs(x_query[None, :] - centers) - eps_array[:, None])
        if distance_norm == 1:
            return np.sum(diff, axis=1)
        if distance_norm == np.inf:
            return np.max(diff, axis=1)
        return np.linalg.norm(diff, ord=distance_norm, axis=1)

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
        if A_nominal is None:
            return False, False
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
        beam_attempted: bool = False,
        beam_budget_exhausted: bool = False,
        beam_fallback_to_exact: bool = False,
    ) -> Dict[str, ProfileValue]:
        return {
            "decode_mode": mode,
            "decode_exact_fallback_used": bool(exact_fallback_used),
            "decode_nodes_visited": int(nodes_visited),
            "decode_nodes_pruned": int(nodes_pruned),
            "decode_solver_calls": int(solver_calls),
            "decode_product_size": int(product_size),
            "decode_heuristic_success": bool(heuristic_success),
            "decode_beam_attempted": bool(beam_attempted),
            "decode_beam_width": int(self.decode_beam_width),
            "decode_beam_branch_top_k": int(self.decode_beam_branch_top_k),
            "decode_beam_max_solver_calls": int(self.decode_beam_max_solver_calls),
            "decode_beam_budget_exhausted": bool(beam_budget_exhausted),
            "decode_beam_fallback_to_exact": bool(beam_fallback_to_exact),
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
        input_bounds = getattr(self, "input_bounds", None)
        if input_bounds is not None:
            lower, upper = input_bounds
            if np.any(x_cand < lower - tol) or np.any(x_cand > upper + tol):
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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
        fixed_ohe_assignments: Optional[Dict[int, int]] = None,
        sparsity_groups: Optional[Sequence[np.ndarray]] = None,
        sparsity_group_weights: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, ProfileValue]]:
        """
        Project using CVXPY — handles L2 (SOCP) and L1 ball constraints natively.
        """
        d = len(x0)
        z = cp.Variable(d)
        initial_guess = self._projection_initial_guess(
            x0,
            A_full,
            b_full,
            center,
            box_eps,
            ball_eps,
            fixed_dims=fixed_dims,
            nondecreasing_dims=nondecreasing_dims,
            nonincreasing_dims=nonincreasing_dims,
            ohe_slices=ohe_slices,
            fixed_ohe_assignments=fixed_ohe_assignments,
        )
        input_bounds = getattr(self, "input_bounds", None)
        if input_bounds is not None:
            initial_guess = np.clip(initial_guess, *input_bounds)
        z.value = initial_guess

        diff = z - x0
        if sparsity_groups is not None and sparsity_group_weights is not None:
            terms = [
                float(weight) * cp.norm(diff[np.asarray(group, dtype=np.int64)], 1)
                for group, weight in zip(sparsity_groups, sparsity_group_weights)
                if len(group) > 0
            ]
            objective = cp.Minimize(cp.sum(terms) if terms else cp.Constant(0.0))
        elif self.distance_norm == 1:
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
        if input_bounds is not None:
            lower, upper = input_bounds
            constraints.extend([z >= lower, z <= upper])

        # Add the Lp ball constraint (handled natively by CVXPY)
        if self.norm == 2:
            constraints.append(cp.norm(z - center, 2) <= ball_eps)
        elif self.norm == 1:
            constraints.append(cp.norm(z - center, 1) <= ball_eps)
        # For L∞, box constraint already covers it

        # Fix specified dimensions to their query values
        if fixed_dims is not None and len(fixed_dims) > 0:
            constraints.append(z[fixed_dims] == x0[fixed_dims])

        # Query-relative directional constraints for numerical features.
        if nondecreasing_dims is not None and len(nondecreasing_dims) > 0:
            constraints.append(z[nondecreasing_dims] >= x0[nondecreasing_dims])
        if nonincreasing_dims is not None and len(nonincreasing_dims) > 0:
            constraints.append(z[nonincreasing_dims] <= x0[nonincreasing_dims])

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

        solver_profile = self._default_cvxpy_solver_profile()
        solved = False
        for attempt_idx, solver_name in enumerate(self.cvxpy_solvers, start=1):
            solver_profile["cvxpy_solver_attempts"] = int(attempt_idx)
            try:
                problem.solve(
                    solver=solver_name,
                    verbose=False,
                    warm_start=True,
                    **self.cvxpy_solver_options.get(solver_name, {}),
                )
                solver_profile["cvxpy_solver_used"] = solver_name
                solver_profile["cvxpy_solver_status"] = str(problem.status or "")
                accepted_statuses = self.cvxpy_accept_statuses[solver_name]
                if problem.status in accepted_statuses and z.value is not None:
                    solved = True
                    break
            except Exception:
                solver_profile["cvxpy_solver_used"] = solver_name
                solver_profile["cvxpy_solver_status"] = "exception"
                continue
        solver_profile["cvxpy_fallback_used"] = bool(int(solver_profile["cvxpy_solver_attempts"]) > 1)
        if not solved:
            return None, np.inf, solver_profile

        x_proj = z.value
        tol = _PROJECTION_FEASIBILITY_TOL
        if np.min(A_full @ x_proj + b_full) < -tol:
            return None, np.inf, solver_profile
        if np.any(x_proj < center - box_eps - tol) or np.any(x_proj > center + box_eps + tol):
            return None, np.inf, solver_profile
        if input_bounds is not None:
            lower, upper = input_bounds
            if np.any(x_proj < lower - tol) or np.any(x_proj > upper + tol):
                return None, np.inf, solver_profile
        if not self._directional_constraints_satisfied(
            x_proj,
            x0,
            nondecreasing_dims=nondecreasing_dims,
            nonincreasing_dims=nonincreasing_dims,
            tol=tol,
        ):
            return None, np.inf, solver_profile
        if self.norm == np.inf:
            in_ball = np.max(np.abs(x_proj - center)) <= ball_eps + tol
        else:
            in_ball = np.linalg.norm(x_proj - center, ord=self.norm) <= ball_eps + tol
        if not in_ball:
            return None, np.inf, solver_profile
        dist = float(np.linalg.norm(x_proj - x0, ord=self.distance_norm))
        solver_profile["sparsity_selection_score"] = self._sparsity_surrogate_score(
            x_proj,
            x0,
            sparsity_groups,
        )
        return x_proj, dist, solver_profile

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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
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

        input_bounds = getattr(self, "input_bounds", None)
        if input_bounds is None:
            bounds = [(center[i] - box_eps, center[i] + box_eps) for i in range(d)]
        else:
            input_lower, input_upper = input_bounds
            bounds = [
                (
                    max(center[i] - box_eps, input_lower),
                    min(center[i] + box_eps, input_upper),
                )
                for i in range(d)
            ]

        # Fix specified dimensions: tighten bounds to a single value
        if fixed_dims is not None:
            for i in fixed_dims:
                i = int(i)
                lo, hi = bounds[i]
                value = float(x0[i])
                if value < lo - _PROJECTION_FEASIBILITY_TOL or value > hi + _PROJECTION_FEASIBILITY_TOL:
                    return None, np.inf
                bounds[i] = (value, value)

        if nondecreasing_dims is not None:
            for i in nondecreasing_dims:
                lo, hi = bounds[int(i)]
                bounds[int(i)] = (max(lo, float(x0[int(i)])), hi)
        if nonincreasing_dims is not None:
            for i in nonincreasing_dims:
                lo, hi = bounds[int(i)]
                bounds[int(i)] = (lo, min(hi, float(x0[int(i)])))

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
                        lo, hi = bounds[i]
                        if (
                            target < lo - _PROJECTION_FEASIBILITY_TOL
                            or target > hi + _PROJECTION_FEASIBILITY_TOL
                        ):
                            return None, np.inf
                        bounds[i] = (target, target)

        if any(lo > hi + _PROJECTION_FEASIBILITY_TOL for lo, hi in bounds):
            return None, np.inf

        x_init = self._projection_initial_guess(
            x0,
            A_full,
            b_full,
            center,
            box_eps,
            box_eps,
            fixed_dims=fixed_dims,
            nondecreasing_dims=nondecreasing_dims,
            nonincreasing_dims=nonincreasing_dims,
            ohe_slices=ohe_slices,
            fixed_ohe_assignments=fixed_ohe_assignments,
        )
        if input_bounds is not None:
            x_init = np.clip(x_init, *input_bounds)

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
        tol = _PROJECTION_FEASIBILITY_TOL
        if np.min(margins) < -tol:
            return None, np.inf

        if np.any(x_proj < center - box_eps - tol) or \
           np.any(x_proj > center + box_eps + tol):
            return None, np.inf
        if input_bounds is not None:
            lower, upper = input_bounds
            if np.any(x_proj < lower - tol) or np.any(x_proj > upper + tol):
                return None, np.inf
        if not self._directional_constraints_satisfied(
            x_proj,
            x0,
            nondecreasing_dims=nondecreasing_dims,
            nonincreasing_dims=nonincreasing_dims,
            tol=tol,
        ):
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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
        fixed_ohe_assignments: Optional[Dict[int, int]] = None,
        sparsity_groups: Optional[Sequence[np.ndarray]] = None,
        sparsity_group_weights: Optional[np.ndarray] = None,
        profile_out: Optional[Dict[str, ProfileValue]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        use_cvxpy = CVXPY_AVAILABLE and (self.norm in (1, 2) or self.distance_norm != 2)
        if sparsity_group_weights is not None and not use_cvxpy:
            raise RuntimeError("Sparsity-weighted projection requires CVXPY")
        if use_cvxpy:
            x_proj, dist, solver_profile = self._project_cvxpy(
                x0,
                A_full,
                b_full,
                center,
                box_eps,
                ball_eps,
                fixed_dims=fixed_dims,
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
                ohe_slices=ohe_slices,
                fixed_ohe_assignments=fixed_ohe_assignments,
                sparsity_groups=sparsity_groups,
                sparsity_group_weights=sparsity_group_weights,
            )
            if profile_out is not None:
                profile_out.clear()
                profile_out.update(solver_profile)
            return x_proj, dist
        if profile_out is not None:
            profile_out.clear()
            profile_out.update(self._default_cvxpy_solver_profile())
        slsqp_kwargs: Dict[str, Any] = {
            "fixed_dims": fixed_dims,
            "ohe_slices": ohe_slices,
            "fixed_ohe_assignments": fixed_ohe_assignments,
        }
        has_directional = (
            (nondecreasing_dims is not None and len(nondecreasing_dims) > 0)
            or (nonincreasing_dims is not None and len(nonincreasing_dims) > 0)
        )
        if has_directional:
            slsqp_kwargs["nondecreasing_dims"] = nondecreasing_dims
            slsqp_kwargs["nonincreasing_dims"] = nonincreasing_dims
        return self._project_slsqp(
            x0,
            A_full,
            b_full,
            center,
            box_eps,
            maxiter,
            tol,
            **slsqp_kwargs,
        )

    def _solve_projection_subproblem_for_constraints(
        self,
        x0: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        box_eps: float,
        ball_eps: float,
        *,
        maxiter: int,
        tol: float,
        fixed_dims: Optional[np.ndarray] = None,
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
        fixed_ohe_assignments: Optional[Dict[int, int]] = None,
        sparsity_groups: Optional[Sequence[np.ndarray]] = None,
        sparsity_group_weights: Optional[np.ndarray] = None,
        profile_out: Optional[Dict[str, ProfileValue]] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Call projection with old-compatible kwargs when directional dims are absent."""
        kwargs: Dict[str, Any] = {
            "maxiter": maxiter,
            "tol": tol,
            "fixed_dims": fixed_dims,
            "ohe_slices": ohe_slices,
            "fixed_ohe_assignments": fixed_ohe_assignments,
            "profile_out": profile_out,
        }
        if sparsity_groups is not None or sparsity_group_weights is not None:
            kwargs["sparsity_groups"] = sparsity_groups
            kwargs["sparsity_group_weights"] = sparsity_group_weights
        has_directional = (
            (nondecreasing_dims is not None and len(nondecreasing_dims) > 0)
            or (nonincreasing_dims is not None and len(nonincreasing_dims) > 0)
        )
        if has_directional:
            kwargs["nondecreasing_dims"] = nondecreasing_dims
            kwargs["nonincreasing_dims"] = nonincreasing_dims
        return self._solve_projection_subproblem(
            x0,
            A_full,
            b_full,
            center,
            box_eps,
            ball_eps,
            **kwargs,
        )

    def _heuristic_polytope_decode(
        self,
        x_star: np.ndarray,
        x_query: np.ndarray,
        A_full: np.ndarray,
        b_full: np.ndarray,
        center: np.ndarray,
        ball_eps: float,
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        sparsity_groups: Optional[Sequence[np.ndarray]] = None,
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
        best_score = np.inf
        for cand in candidates:
            if not self._directional_constraints_satisfied(
                cand,
                x_query,
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
                tol=tol,
            ):
                continue
            if self._is_certified_candidate(cand, A_full, b_full, center, ball_eps, tol=tol):
                d_cand = float(np.linalg.norm(cand - x_query, ord=self.distance_norm))
                score_cand = self._sparsity_surrogate_score(cand, x_query, sparsity_groups)
                if score_cand < best_score:
                    best_x, best_dist = cand, d_cand
                    best_score = score_cand

        return best_x, best_dist

    def _beam_polytope_decode(
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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        sparsity_groups: Optional[Sequence[np.ndarray]] = None,
        sparsity_group_weights: Optional[np.ndarray] = None,
        incumbent_upper_bound: float = np.inf,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, ProfileValue]]:
        ohe_slices = self.ohe_slices or []
        product_size = self._ohe_product_size(ohe_slices)
        profile = self._build_decode_profile(
            mode="beam",
            exact_fallback_used=False,
            nodes_visited=0,
            nodes_pruned=0,
            solver_calls=0,
            product_size=product_size,
            heuristic_success=False,
            beam_attempted=True,
        )

        block_indices = [i for i, (s, e) in enumerate(ohe_slices) if e - s > 1]
        root_dist = float(np.linalg.norm(x_star - x_query, ord=self.distance_norm))
        root_score = self._sparsity_surrogate_score(x_star, x_query, sparsity_groups)
        if not block_indices:
            if root_score >= incumbent_upper_bound:
                profile["decode_nodes_pruned"] = 1
                return None, np.inf, profile
            profile["sparsity_selection_score"] = root_score
            return x_star.copy(), root_dist, profile

        if root_score >= incumbent_upper_bound:
            profile["decode_nodes_pruned"] = 1
            return None, np.inf, profile

        beam: List[Tuple[Dict[int, int], np.ndarray, float, float]] = [({}, x_star.copy(), root_dist, root_score)]
        best_x: Optional[np.ndarray] = None
        best_dist = float(incumbent_upper_bound)
        best_true_dist = np.inf

        while beam:
            next_beam: List[Tuple[Dict[int, int], np.ndarray, float, float]] = []
            for fixed_assignments, relaxed_point, true_dist, lower_bound in beam:
                if lower_bound >= best_dist:
                    profile["decode_nodes_pruned"] = int(profile["decode_nodes_pruned"]) + 1
                    continue

                if len(fixed_assignments) == len(block_indices):
                    best_x = relaxed_point
                    best_dist = float(lower_bound)
                    best_true_dist = float(true_dist)
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
                category_order = category_order[: self.decode_beam_branch_top_k]
                for category in category_order:
                    if int(profile["decode_solver_calls"]) >= self.decode_beam_max_solver_calls:
                        profile["decode_beam_budget_exhausted"] = True
                        break
                    child_assignments = dict(fixed_assignments)
                    child_assignments[branch_block] = category
                    profile["decode_nodes_visited"] = int(profile["decode_nodes_visited"]) + 1
                    profile["decode_solver_calls"] = int(profile["decode_solver_calls"]) + 1
                    child_profile: Dict[str, ProfileValue] = {}
                    child_x, child_dist = self._solve_projection_subproblem_for_constraints(
                        x_query,
                        A_full,
                        b_full,
                        center,
                        box_eps,
                        ball_eps,
                        maxiter=maxiter,
                        tol=tol,
                        fixed_dims=fixed_dims,
                        nondecreasing_dims=nondecreasing_dims,
                        nonincreasing_dims=nonincreasing_dims,
                        ohe_slices=ohe_slices,
                        fixed_ohe_assignments=child_assignments,
                        sparsity_groups=sparsity_groups,
                        sparsity_group_weights=sparsity_group_weights,
                        profile_out=child_profile,
                    )
                    child_score = self._score_from_profile(child_profile, child_dist)
                    if child_x is None or child_score >= best_dist:
                        profile["decode_nodes_pruned"] = int(profile["decode_nodes_pruned"]) + 1
                        continue
                    if len(child_assignments) == len(block_indices):
                        best_x = child_x
                        best_dist = float(child_score)
                        best_true_dist = float(child_dist)
                        continue
                    next_beam.append((child_assignments, child_x, float(child_dist), float(child_score)))
                if bool(profile["decode_beam_budget_exhausted"]):
                    break

            if bool(profile["decode_beam_budget_exhausted"]):
                break
            if not next_beam:
                break
            next_beam.sort(key=lambda item: item[3])
            beam = next_beam[: self.decode_beam_width]

        if best_x is None:
            return None, np.inf, profile
        profile["sparsity_selection_score"] = float(best_dist)
        return best_x, best_true_dist, profile

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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        sparsity_groups: Optional[Sequence[np.ndarray]] = None,
        sparsity_group_weights: Optional[np.ndarray] = None,
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
            nondecreasing_dims=nondecreasing_dims,
            nonincreasing_dims=nonincreasing_dims,
            sparsity_groups=sparsity_groups,
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
            profile["sparsity_selection_score"] = self._sparsity_surrogate_score(
                best_x,
                x_query,
                sparsity_groups,
            )
            return best_x, best_dist, profile

        incumbent = float(incumbent_upper_bound)
        if self.ohe_decode_mode in {"beam_then_exact", "beam_only"}:
            beam_x, beam_dist, beam_profile = self._beam_polytope_decode(
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
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
                sparsity_groups=sparsity_groups,
                sparsity_group_weights=sparsity_group_weights,
                incumbent_upper_bound=incumbent,
            )
            if beam_x is not None:
                return beam_x, beam_dist, beam_profile
            if self.ohe_decode_mode == "beam_only":
                return None, np.inf, beam_profile

        if product_size <= _EXACT_ENUM_PRODUCT_THRESHOLD:
            exact_x, exact_dist, exact_profile = self._exact_polytope_decode_enumeration(
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
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
                sparsity_groups=sparsity_groups,
                sparsity_group_weights=sparsity_group_weights,
                incumbent_upper_bound=incumbent,
            )
        else:
            exact_x, exact_dist, exact_profile = self._exact_polytope_decode_branch_and_bound(
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
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
                sparsity_groups=sparsity_groups,
                sparsity_group_weights=sparsity_group_weights,
                incumbent_upper_bound=incumbent,
            )
        if self.ohe_decode_mode == "beam_then_exact":
            exact_profile["decode_beam_attempted"] = True
            exact_profile["decode_beam_fallback_to_exact"] = True
            exact_profile["decode_beam_budget_exhausted"] = beam_profile["decode_beam_budget_exhausted"]
        return exact_x, exact_dist, exact_profile

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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        sparsity_groups: Optional[Sequence[np.ndarray]] = None,
        sparsity_group_weights: Optional[np.ndarray] = None,
        incumbent_upper_bound: float = np.inf,
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
        root_score = self._sparsity_surrogate_score(x_star, x_query, sparsity_groups)
        if root_score >= incumbent_upper_bound:
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
        best_true_dist = np.inf
        for assignment in product(*category_orders):
            fixed_ohe_assignments = {block_idx: int(cat) for block_idx, cat in zip(block_indices, assignment)}
            profile["decode_nodes_visited"] = int(profile["decode_nodes_visited"]) + 1
            profile["decode_solver_calls"] = int(profile["decode_solver_calls"]) + 1
            leaf_profile: Dict[str, ProfileValue] = {}
            x_leaf, dist_leaf = self._solve_projection_subproblem_for_constraints(
                x_query,
                A_full,
                b_full,
                center,
                box_eps,
                ball_eps,
                maxiter=maxiter,
                tol=tol,
                fixed_dims=fixed_dims,
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
                ohe_slices=ohe_slices,
                fixed_ohe_assignments=fixed_ohe_assignments,
                sparsity_groups=sparsity_groups,
                sparsity_group_weights=sparsity_group_weights,
                profile_out=leaf_profile,
            )
            score_leaf = self._score_from_profile(leaf_profile, dist_leaf)
            if x_leaf is None or score_leaf >= best_dist:
                profile["decode_nodes_pruned"] = int(profile["decode_nodes_pruned"]) + 1
                continue
            best_x = x_leaf
            best_dist = float(score_leaf)
            best_true_dist = float(dist_leaf)

        if best_x is None:
            return None, np.inf, profile
        profile["sparsity_selection_score"] = float(best_dist)
        return best_x, best_true_dist, profile

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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        sparsity_groups: Optional[Sequence[np.ndarray]] = None,
        sparsity_group_weights: Optional[np.ndarray] = None,
        incumbent_upper_bound: float = np.inf,
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
            root_score = self._sparsity_surrogate_score(x_star, x_query, sparsity_groups)
            if root_score >= incumbent_upper_bound:
                profile["decode_nodes_pruned"] = 1
                return None, np.inf, profile
            profile["sparsity_selection_score"] = root_score
            return x_star.copy(), root_dist, profile

        root_dist = float(np.linalg.norm(x_star - x_query, ord=self.distance_norm))
        root_score = self._sparsity_surrogate_score(x_star, x_query, sparsity_groups)
        if root_score >= incumbent_upper_bound:
            profile["decode_nodes_pruned"] = 1
            return None, np.inf, profile

        pq: List[Tuple[float, int, Dict[int, int], np.ndarray, float]] = []
        ticket = count()
        heapq.heappush(pq, (root_score, next(ticket), {}, x_star.copy(), root_dist))

        best_x = None
        best_dist = float(incumbent_upper_bound)
        best_true_dist = np.inf

        while pq:
            lower_bound, _, fixed_assignments, relaxed_point, true_dist = heapq.heappop(pq)
            if lower_bound >= best_dist:
                profile["decode_nodes_pruned"] = int(profile["decode_nodes_pruned"]) + 1
                continue

            if len(fixed_assignments) == len(block_indices):
                best_x = relaxed_point
                best_dist = float(lower_bound)
                best_true_dist = float(true_dist)
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
                child_profile: Dict[str, ProfileValue] = {}
                child_x, child_dist = self._solve_projection_subproblem_for_constraints(
                    x_query,
                    A_full,
                    b_full,
                    center,
                    box_eps,
                    ball_eps,
                    maxiter=maxiter,
                    tol=tol,
                    fixed_dims=fixed_dims,
                    nondecreasing_dims=nondecreasing_dims,
                    nonincreasing_dims=nonincreasing_dims,
                    ohe_slices=ohe_slices,
                    fixed_ohe_assignments=child_assignments,
                    sparsity_groups=sparsity_groups,
                    sparsity_group_weights=sparsity_group_weights,
                    profile_out=child_profile,
                )
                child_score = self._score_from_profile(child_profile, child_dist)
                if child_x is None or child_score >= best_dist:
                    profile["decode_nodes_pruned"] = int(profile["decode_nodes_pruned"]) + 1
                    continue
                heapq.heappush(
                    pq,
                    (float(child_score), next(ticket), child_assignments, child_x, float(child_dist)),
                )

        if best_x is None:
            return None, np.inf, profile
        profile["sparsity_selection_score"] = float(best_dist)
        return best_x, best_true_dist, profile

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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
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
            profile.update(self._default_cvxpy_solver_profile())
            return None, np.inf, profile

        sparsity_active = self._sparsity_active()
        sparsity_groups = self._build_sparsity_groups(d, fixed_dims) if sparsity_active else None
        n_projection_solves = 0
        best_x: Optional[np.ndarray] = None
        best_dist = np.inf
        best_score = np.inf
        best_solver_profile = self._default_cvxpy_solver_profile()
        best_decode_profile: Optional[Dict[str, ProfileValue]] = None
        current_weights: Optional[np.ndarray] = None
        n_attempts = 1 + (self.sparsity_reweight_iters if sparsity_active else 0)

        for attempt_idx in range(n_attempts):
            solver_profile = self._default_cvxpy_solver_profile()
            x_relaxed, _ = self._solve_projection_subproblem_for_constraints(
                x0,
                A_full,
                b_full,
                center,
                box_eps,
                ball_eps,
                maxiter=maxiter,
                tol=tol,
                fixed_dims=fixed_dims,
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
                ohe_slices=self.ohe_slices,
                fixed_ohe_assignments=None,
                sparsity_groups=sparsity_groups,
                sparsity_group_weights=current_weights,
                profile_out=solver_profile,
            )
            n_projection_solves += 1
            if x_relaxed is None:
                if attempt_idx == 0:
                    best_solver_profile = solver_profile
                break

            if self.ohe_slices:
                x_candidate, dist_candidate, decode_profile = self._polytope_aware_decode(
                    x_relaxed,
                    x0,
                    A_full,
                    b_full,
                    center,
                    box_eps,
                    ball_eps,
                    maxiter=maxiter,
                    tol=tol,
                    fixed_dims=fixed_dims,
                    nondecreasing_dims=nondecreasing_dims,
                    nonincreasing_dims=nonincreasing_dims,
                    sparsity_groups=sparsity_groups,
                    sparsity_group_weights=current_weights,
                    incumbent_upper_bound=best_score,
                )
            else:
                x_candidate = x_relaxed
                dist_candidate = float(np.linalg.norm(x_candidate - x0, ord=self.distance_norm))
                decode_profile = self._build_decode_profile(
                    mode="heuristic",
                    exact_fallback_used=False,
                    nodes_visited=0,
                    nodes_pruned=0,
                    solver_calls=0,
                    product_size=1,
                    heuristic_success=True,
                )

            if x_candidate is not None and np.isfinite(dist_candidate):
                candidate_score = self._sparsity_surrogate_score(x_candidate, x0, sparsity_groups)
                decode_profile["sparsity_selection_score"] = candidate_score
                if candidate_score < best_score:
                    best_x = x_candidate
                    best_dist = float(dist_candidate)
                    best_score = float(candidate_score)
                    best_solver_profile = solver_profile
                    best_decode_profile = decode_profile

            if not sparsity_active:
                break

            weight_source = x_candidate if x_candidate is not None else x_relaxed
            current_weights = self._sparsity_group_weights(weight_source, x0, sparsity_groups or [])

        if best_x is None:
            profile = self._build_decode_profile(
                mode="heuristic",
                exact_fallback_used=False,
                nodes_visited=0,
                nodes_pruned=0,
                solver_calls=0,
                product_size=self._ohe_product_size(self.ohe_slices),
                heuristic_success=False,
            )
            profile.update(self._sparsity_metadata_defaults())
            profile.update(best_solver_profile)
            profile["sparsity_group_count"] = int(0 if sparsity_groups is None else len(sparsity_groups))
            profile["sparsity_solver_calls"] = int(n_projection_solves)
            return None, np.inf, profile

        decode_profile = best_decode_profile or self._build_decode_profile(
            mode="heuristic",
            exact_fallback_used=False,
            nodes_visited=0,
            nodes_pruned=0,
            solver_calls=0,
            product_size=1,
            heuristic_success=True,
        )
        changes = self._sparsity_group_changes(best_x, x0, sparsity_groups or [])
        decode_profile.update(self._sparsity_metadata_defaults())
        decode_profile.update(best_solver_profile)
        decode_profile["sparsity_group_count"] = int(0 if sparsity_groups is None else len(sparsity_groups))
        decode_profile["sparsity_active_groups"] = int(
            np.sum(changes > float(getattr(self, "sparsity_eps", 1.0e-3)))
        )
        decode_profile["sparsity_selection_score"] = float(best_score)
        decode_profile["sparsity_solver_calls"] = int(n_projection_solves)
        return best_x, best_dist, decode_profile

    def _project_candidate(
        self,
        x_query: np.ndarray,
        bd: Dict[str, np.ndarray],
        idx: int,
        delta: float,
        robust_norm: Optional[Union[int, float]],
        solver_maxiter: Optional[int],
        fixed_dims: Optional[np.ndarray],
        nondecreasing_dims: Optional[np.ndarray],
        nonincreasing_dims: Optional[np.ndarray],
        incumbent_upper_bound: float,
    ) -> _CandidateProjection:
        """Project onto one atlas region without mutating shared search state."""
        started = time.perf_counter()
        point, distance, profile = self._project_onto_polytope(
            x_query,
            bd["lA"][idx],
            bd["lbias"][idx],
            bd["X"][idx],
            eps_i=float(bd["eps"][idx]),
            delta=delta,
            robust_norm=robust_norm,
            maxiter=solver_maxiter,
            fixed_dims=fixed_dims,
            nondecreasing_dims=nondecreasing_dims,
            nonincreasing_dims=nonincreasing_dims,
            incumbent_upper_bound=incumbent_upper_bound,
        )
        return _CandidateProjection(
            index=int(idx),
            point=point,
            selection_score=float(self._score_from_profile(profile, distance)),
            profile=profile,
            elapsed_s=float(time.perf_counter() - started),
        )

    def _projection_worker_clone(self) -> "CertCFAtlas":
        """Create the minimal CPU-only state needed by projection workers."""
        worker = CertCFAtlas.__new__(CertCFAtlas)
        worker.bounds = self.bounds
        for name in (
            "norm",
            "distance_norm",
            "solver_maxiter",
            "ohe_slices",
            "input_bounds",
            "cvxpy_solvers",
            "cvxpy_solver_options",
            "cvxpy_accept_statuses",
            "ohe_decode_mode",
            "decode_beam_width",
            "decode_beam_branch_top_k",
            "decode_beam_max_solver_calls",
            "sparsity_penalty",
            "sparsity_lambda",
            "sparsity_reweight_iters",
            "sparsity_eps",
            "sparsity_group_ohe",
        ):
            setattr(worker, name, getattr(self, name))
        return worker

    def _get_candidate_process_pool(self, workers: int):
        """Return a persistent fork pool large enough for candidate projection."""
        workers = int(workers)
        existing = getattr(self, "_candidate_process_pool", None)
        existing_workers = int(getattr(self, "_candidate_process_pool_workers", 0))
        if existing is not None and existing_workers >= workers:
            return existing
        self.close_candidate_process_pool()
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError(
                "candidate_parallel_backend='process' requires a platform with fork support"
            )
        global _PROCESS_PROJECTION_ATLAS
        _PROCESS_PROJECTION_ATLAS = self._projection_worker_clone()
        context = mp.get_context("fork")
        pool = context.Pool(processes=workers)
        self._candidate_process_pool = pool
        self._candidate_process_pool_workers = workers
        return pool

    def close_candidate_process_pool(self) -> None:
        """Close the optional persistent projection process pool."""
        pool = getattr(self, "_candidate_process_pool", None)
        if pool is not None:
            pool.close()
            pool.join()
        self._candidate_process_pool = None
        self._candidate_process_pool_workers = 0
        global _PROCESS_PROJECTION_ATLAS
        _PROCESS_PROJECTION_ATLAS = None

    def _make_project_fn(
        self,
        x_query,
        bd,
        delta,
        robust_norm,
        solver_maxiter,
        fixed_dims,
        nondecreasing_dims=None,
        nonincreasing_dims=None,
    ):
        """Return a timed projection closure plus decode profiling accumulators."""
        projection_time_s = [0.0]
        best_projection_dist = [np.inf]
        profile_recorded = [False]
        initial_profile = self._build_decode_profile(
            mode="heuristic",
            exact_fallback_used=False,
            nodes_visited=0,
            nodes_pruned=0,
            solver_calls=0,
            product_size=self._ohe_product_size(self.ohe_slices),
            heuristic_success=not bool(self.ohe_slices),
        )
        initial_profile.update(self._sparsity_metadata_defaults())
        best_decode_profile = [initial_profile]

        def project_fn(idx: int, incumbent_upper_bound: float) -> Tuple[Optional[np.ndarray], float]:
            result = self._project_candidate(
                x_query=x_query,
                bd=bd,
                idx=int(idx),
                delta=delta,
                robust_norm=robust_norm,
                solver_maxiter=solver_maxiter,
                fixed_dims=fixed_dims,
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
                incumbent_upper_bound=incumbent_upper_bound,
            )
            projection_time_s[0] += result.elapsed_s
            selection_score = result.selection_score
            decode_profile = result.profile
            improved_dist = selection_score < best_projection_dist[0]
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
                best_projection_dist[0] = selection_score
            return result.point, selection_score

        return project_fn, projection_time_s, best_decode_profile

    def _make_project_fn_for_constraints(
        self,
        x_query,
        bd,
        delta,
        robust_norm,
        solver_maxiter,
        fixed_dims,
        nondecreasing_dims=None,
        nonincreasing_dims=None,
    ):
        has_directional = (
            (nondecreasing_dims is not None and len(nondecreasing_dims) > 0)
            or (nonincreasing_dims is not None and len(nonincreasing_dims) > 0)
        )
        if not has_directional:
            return self._make_project_fn(
                x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims
            )
        return self._make_project_fn(
            x_query,
            bd,
            delta,
            robust_norm,
            solver_maxiter,
            fixed_dims,
            nondecreasing_dims,
            nonincreasing_dims,
        )

    def _search_bvh(
        self,
        x_query,
        bd,
        target_class,
        delta,
        robust_norm,
        solver_maxiter,
        fixed_dims,
        nondecreasing_dims=None,
        nonincreasing_dims=None,
    ):
        """Branch-and-bound BVH search. Returns (x_cf, dist, anchor_idx, n_qp, profiling_dict)."""
        project_fn, projection_time_s, best_decode_profile = self._make_project_fn_for_constraints(
            x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims,
            nondecreasing_dims, nonincreasing_dims)
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

    def _search_sorted(
        self,
        x_query,
        bd,
        target_class,
        delta,
        robust_norm,
        solver_maxiter,
        fixed_dims,
        nondecreasing_dims=None,
        nonincreasing_dims=None,
    ):
        """Sorted lower-bound scan. Returns (x_cf, dist, anchor_idx, n_qp, profiling_dict)."""
        project_fn, projection_time_s, best_decode_profile = self._make_project_fn_for_constraints(
            x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims,
            nondecreasing_dims, nonincreasing_dims)
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
        candidate_parallelism: int,
        candidate_parallel_backend: str,
        nondecreasing_dims=None,
        nonincreasing_dims=None,
    ):
        """Project only onto the top-k nearest target-class anchors.

        This is an approximate, fixed-budget query strategy. It preserves
        certified validity for any returned point because projection is still
        performed against certified target-class polytopes, but it does not
        guarantee the closest point over the full atlas.
        """
        project_fn, projection_time_s, best_decode_profile = self._make_project_fn_for_constraints(
            x_query, bd, delta, robust_norm, solver_maxiter, fixed_dims,
            nondecreasing_dims, nonincreasing_dims)
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
        lower_bounds = self._anchor_bbox_lower_bounds(
            np.asarray(x_query, dtype=np.float64),
            np.asarray(bd["X"], dtype=np.float64),
            np.asarray(bd["eps"], dtype=np.float64),
            self.distance_norm,
        )
        primary_indices = sorted_candidate_indices[:k]
        fallback_indices = sorted_candidate_indices[k:]

        centers = np.asarray(bd["X"], dtype=np.float64)
        anchor_distances = np.linalg.norm(
            centers - np.asarray(x_query, dtype=np.float64)[None, :],
            ord=self.distance_norm,
            axis=1,
        )
        nearest_anchor_candidate_idx: Optional[int] = None
        nearest_anchor_candidate_dist = np.inf
        nearest_anchor_candidate_certified = False
        nearest_anchor_idx: Optional[int] = None
        nearest_anchor_dist = np.inf
        membership_robust_norm = self.norm if robust_norm is None else robust_norm
        for candidate_idx in np.argsort(anchor_distances):
            candidate_idx = int(candidate_idx)
            if fixed_dims is not None and len(fixed_dims) > 0:
                if not np.allclose(centers[candidate_idx][fixed_dims], x_query[fixed_dims], atol=1e-6):
                    continue
            if not self._directional_constraints_satisfied(
                centers[candidate_idx],
                x_query,
                nondecreasing_dims=nondecreasing_dims,
                nonincreasing_dims=nonincreasing_dims,
                tol=1e-6,
            ):
                continue
            _, certified = self._polytope_membership_for_anchor(
                centers[candidate_idx],
                bd,
                candidate_idx,
                delta=delta,
                robust_norm=membership_robust_norm,
            )
            if nearest_anchor_candidate_idx is None:
                nearest_anchor_candidate_idx = candidate_idx
                nearest_anchor_candidate_dist = float(anchor_distances[candidate_idx])
                nearest_anchor_candidate_certified = bool(certified)
            if certified:
                nearest_anchor_idx = candidate_idx
                nearest_anchor_dist = float(anchor_distances[candidate_idx])
                break

        if nearest_anchor_idx is None:
            best_point: Optional[np.ndarray] = None
            best_dist = np.inf
            best_idx: Optional[int] = None
            best_source = "projection"
        else:
            best_point = centers[nearest_anchor_idx].copy()
            best_dist = self._sparsity_surrogate_score(best_point, x_query)
            best_idx = nearest_anchor_idx
            best_source = "nearest_anchor"
        initial_best_point = None if best_point is None else best_point.copy()
        initial_best_dist = float(best_dist)
        initial_best_idx = best_idx
        initial_best_source = best_source

        n_qp = 0
        fallback_used = False
        n_pruned_by_bound = 0
        best_lower_bound_at_termination = np.nan
        candidate_parallelism = int(candidate_parallelism)
        if candidate_parallelism <= 0:
            raise ValueError("candidate_parallelism must be positive")
        candidate_parallel_backend = str(candidate_parallel_backend).lower()
        if candidate_parallel_backend not in _CANDIDATE_PARALLEL_BACKENDS:
            raise ValueError(
                "candidate_parallel_backend must be one of {'thread', 'process'}"
            )
        primary_projection_wall_time_s = 0.0
        parallel_projection_work_time_s = 0.0
        parallel_candidate_count = 0
        candidate_workers_used = 1
        candidate_parent_refinement_used = False

        if candidate_parallelism <= 1:
            for idx in primary_indices:
                lower_bound = float(lower_bounds[int(idx)])
                if lower_bound >= best_dist:
                    n_pruned_by_bound += 1
                    if np.isnan(best_lower_bound_at_termination):
                        best_lower_bound_at_termination = lower_bound
                    else:
                        best_lower_bound_at_termination = min(
                            best_lower_bound_at_termination,
                            lower_bound,
                        )
                    continue
                point, dist = project_fn(int(idx), best_dist)
                n_qp += 1
                if dist < best_dist:
                    best_point = point
                    best_dist = dist
                    best_idx = int(idx)
                    best_source = "projection"
        else:
            # All candidates receive the same safe incumbent. This may do
            # more work than the serial search because later projections do
            # not see intermediate improvements, but it makes the k projection
            # problems independent without changing their feasible regions.
            initial_incumbent = float(best_dist)
            eligible_indices: List[int] = []
            for idx in primary_indices:
                candidate_idx = int(idx)
                lower_bound = float(lower_bounds[candidate_idx])
                if lower_bound >= initial_incumbent:
                    n_pruned_by_bound += 1
                    if np.isnan(best_lower_bound_at_termination):
                        best_lower_bound_at_termination = lower_bound
                    else:
                        best_lower_bound_at_termination = min(
                            best_lower_bound_at_termination,
                            lower_bound,
                        )
                    continue
                eligible_indices.append(candidate_idx)

            if len(eligible_indices) == 1:
                candidate_idx = eligible_indices[0]
                point, dist = project_fn(candidate_idx, initial_incumbent)
                n_qp += 1
                if dist < best_dist:
                    best_point = point
                    best_dist = dist
                    best_idx = candidate_idx
                    best_source = "projection"
            elif eligible_indices:
                candidate_workers_used = min(candidate_parallelism, len(eligible_indices))
                projection_started = time.perf_counter()
                if candidate_parallel_backend == "thread":
                    with ThreadPoolExecutor(
                        max_workers=candidate_workers_used,
                        thread_name_prefix="certcf-candidate",
                    ) as executor:
                        futures = [
                            executor.submit(
                                self._project_candidate,
                                x_query=x_query,
                                bd=bd,
                                idx=candidate_idx,
                                delta=delta,
                                robust_norm=robust_norm,
                                solver_maxiter=solver_maxiter,
                                fixed_dims=fixed_dims,
                                nondecreasing_dims=nondecreasing_dims,
                                nonincreasing_dims=nonincreasing_dims,
                                incumbent_upper_bound=initial_incumbent,
                            )
                            for candidate_idx in eligible_indices
                        ]
                        # Consume in anchor order so equal-score ties have the
                        # same deterministic resolution as the serial path.
                        parallel_results = [future.result() for future in futures]
                else:
                    process_pool = self._get_candidate_process_pool(candidate_parallelism)
                    tasks = [
                        (
                            # Preserve the serial path's input dtype.  Promoting
                            # float32 queries to float64 here can perturb the
                            # solver solution and categorical decoding.
                            np.asarray(x_query).copy(),
                            int(target_class),
                            candidate_idx,
                            float(delta),
                            robust_norm,
                            solver_maxiter,
                            fixed_dims,
                            nondecreasing_dims,
                            nonincreasing_dims,
                            initial_incumbent,
                        )
                        for candidate_idx in eligible_indices
                    ]
                    parallel_results = []
                    for start in range(0, len(tasks), candidate_workers_used):
                        parallel_results.extend(
                            process_pool.map(
                                _run_candidate_projection_in_process,
                                tasks[start:start + candidate_workers_used],
                            )
                        )
                primary_projection_wall_time_s = time.perf_counter() - projection_started
                parallel_projection_work_time_s = float(
                    sum(result.elapsed_s for result in parallel_results)
                )
                parallel_candidate_count = len(parallel_results)
                n_qp += parallel_candidate_count

                best_projection_score = np.inf
                profile_recorded = False
                for result in parallel_results:
                    improved_projection = result.selection_score < best_projection_score
                    if improved_projection or not profile_recorded:
                        best_decode_profile[0] = result.profile
                        profile_recorded = True
                    if improved_projection:
                        best_projection_score = result.selection_score
                    if result.selection_score < best_dist:
                        best_point = result.point
                        best_dist = result.selection_score
                        best_idx = result.index
                        best_source = "projection"

                # Process-local solver state can introduce small numerical
                # differences for reweighted sparsity or categorical decoding.
                # Re-solve the winning region in the parent process to reduce
                # process-local numerical differences in the returned point.
                # Candidate ranking can still differ in near-tie cases because
                # the remaining regions are not redundantly re-solved.
                needs_parent_refinement = bool(
                    candidate_parallel_backend == "process"
                    and best_source == "projection"
                    and (self._sparsity_active() or bool(self.ohe_slices))
                )
                if needs_parent_refinement and best_idx is not None:
                    refined_point, refined_score = project_fn(
                        int(best_idx), initial_incumbent
                    )
                    n_qp += 1
                    candidate_parent_refinement_used = True
                    if refined_point is not None and refined_score < initial_best_dist:
                        best_point = refined_point
                        best_dist = refined_score
                        best_source = "projection"
                    else:
                        best_point = initial_best_point
                        best_dist = initial_best_dist
                        best_idx = initial_best_idx
                        best_source = initial_best_source

        # If no certified incumbent is available after the fixed top-k budget,
        # keep the method useful by scanning remaining anchors in nearest-anchor
        # order until a certified projection is found.
        if best_point is None:
            fallback_used = True
            for idx in fallback_indices:
                lower_bound = float(lower_bounds[int(idx)])
                if lower_bound >= best_dist:
                    n_pruned_by_bound += 1
                    continue
                point, dist = project_fn(int(idx), best_dist)
                n_qp += 1
                if point is not None and np.isfinite(dist):
                    best_point = point
                    best_dist = dist
                    best_idx = int(idx)
                    best_source = "projection"
                    break
            best_lower_bound_at_termination = np.nan

        query_loop_time_s = time.perf_counter() - t0
        serial_projection_time_s = float(projection_time_s[0])
        projection_work_time_s = parallel_projection_work_time_s + serial_projection_time_s
        projection_wall_time_s = (
            primary_projection_wall_time_s + serial_projection_time_s
            if candidate_parallelism > 1
            else serial_projection_time_s
        )
        profiling = {
            "query_loop_time_ms": 1e3 * query_loop_time_s,
            "search_time_ms": 1e3 * max(0.0, query_loop_time_s - projection_wall_time_s),
            # Sum of the individual solver durations (computational work).
            "projection_time_ms": 1e3 * projection_work_time_s,
            # Wall-clock time spent waiting for projection batches.
            "projection_wall_time_ms": 1e3 * projection_wall_time_s,
            "distance_norm": float(self.distance_norm) if self.distance_norm == np.inf else int(self.distance_norm),
            "query_k_candidates": float(k),
            "candidate_parallelism": float(candidate_parallelism),
            "candidate_parallel_backend": candidate_parallel_backend,
            "candidate_workers_used": float(candidate_workers_used),
            "parallel_candidate_count": float(parallel_candidate_count),
            "candidate_parent_refinement_used": candidate_parent_refinement_used,
            "nearest_anchor_fallback_used": float(fallback_used),
            "n_candidates_considered": float(n_qp),
            "n_candidates_total": float(bvh.n_polytopes),
            "n_candidates_pruned_by_top_k": float(max(0, bvh.n_polytopes - k) if not fallback_used else 0),
            "n_candidates_pruned_by_bound": float(n_pruned_by_bound),
            "best_lower_bound_at_termination": float(best_lower_bound_at_termination),
            "nearest_anchor_initialization_used": float(nearest_anchor_idx is not None),
            "nearest_anchor_initial_idx": float(nearest_anchor_idx) if nearest_anchor_idx is not None else np.nan,
            "nearest_anchor_initial_distance": float(nearest_anchor_dist),
            "nearest_anchor_candidate_idx": (
                float(nearest_anchor_candidate_idx)
                if nearest_anchor_candidate_idx is not None
                else np.nan
            ),
            "nearest_anchor_candidate_distance": float(nearest_anchor_candidate_dist),
            "nearest_anchor_candidate_certified": float(nearest_anchor_candidate_certified),
            "nearest_anchor_returned": float(best_source == "nearest_anchor"),
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
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        query_k_candidates: int = 1,
        candidate_parallelism: Optional[int] = None,
        candidate_parallel_backend: Optional[str] = None,
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
        resolved_candidate_parallelism = int(
            getattr(self, "candidate_parallelism", 1)
            if candidate_parallelism is None
            else candidate_parallelism
        )
        if resolved_candidate_parallelism <= 0:
            raise ValueError("candidate_parallelism must be positive")
        resolved_candidate_parallel_backend = str(
            getattr(self, "candidate_parallel_backend", "thread")
            if candidate_parallel_backend is None
            else candidate_parallel_backend
        ).lower()
        if resolved_candidate_parallel_backend not in _CANDIDATE_PARALLEL_BACKENDS:
            raise ValueError(
                "candidate_parallel_backend must be one of {'thread', 'process'}"
            )
        profiling: Dict[str, ProfileValue] = {
            "method": resolved_method,
            "delta": float(delta),
            "classification_margin": float(getattr(self, "classification_margin", 0.0)),
            "nondecreasing_dims_count": (
                0 if nondecreasing_dims is None else int(len(nondecreasing_dims))
            ),
            "nonincreasing_dims_count": (
                0 if nonincreasing_dims is None else int(len(nonincreasing_dims))
            ),
            "robust_norm": (
                float(robust_norm) if robust_norm == np.inf else int(robust_norm)
            ) if robust_norm is not None else (
                float(self.norm) if self.norm == np.inf else int(self.norm)
            ),
        }
        t_total_start = time.perf_counter()

        if resolved_method == 'bvh':
            x_cf, selection_score, anchor_idx, n_qp, search_profiling = self._search_bvh(
                x_query, bd, target_class, delta, robust_norm, solver_maxiter,
                fixed_dims, nondecreasing_dims, nonincreasing_dims)
        elif resolved_method == 'sorted':
            x_cf, selection_score, anchor_idx, n_qp, search_profiling = self._search_sorted(
                x_query, bd, target_class, delta, robust_norm, solver_maxiter,
                fixed_dims, nondecreasing_dims, nonincreasing_dims)
        elif resolved_method == 'nearest_anchor':
            x_cf, selection_score, anchor_idx, n_qp, search_profiling = self._search_nearest_anchor(
                x_query, bd, target_class, delta, robust_norm, solver_maxiter,
                fixed_dims, query_k_candidates, resolved_candidate_parallelism,
                resolved_candidate_parallel_backend, nondecreasing_dims,
                nonincreasing_dims)
        else:
            raise ValueError(f"Unknown method: {resolved_method}. Use 'sorted', 'bvh', or 'nearest_anchor'.")

        profiling.update(search_profiling)

        total_time_s = time.perf_counter() - t_total_start
        if x_cf is not None and self._sparsity_active():
            dist = float(np.linalg.norm(np.asarray(x_cf, dtype=np.float64) - x_query, ord=self.distance_norm))
        else:
            dist = float(selection_score)
        profiling.update({
            "total_time_ms": 1e3 * total_time_s,
            "n_qp_solved": float(n_qp),
            "success": float(x_cf is not None),
            "distance": float(dist),
            "sparsity_selection_score": float(selection_score),
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
        target_class: Union[int, Sequence[int], np.ndarray],
        method: Optional[str] = None,
        delta: float = 0.0,
        robust_norm: Optional[Union[int, float, str]] = None,
        solver_maxiter: Optional[int] = None,
        fixed_dims: Optional[np.ndarray] = None,
        nondecreasing_dims: Optional[np.ndarray] = None,
        nonincreasing_dims: Optional[np.ndarray] = None,
        query_k_candidates: int = 1,
        candidate_parallelism: Optional[int] = None,
        candidate_parallel_backend: Optional[str] = None,
        timeout_s_per_query: Optional[float] = None,
    ) -> List[CounterfactualResult]:
        """Find counterfactuals for a batch of query points."""
        X_query_np = np.asarray(X_query, dtype=np.float32)
        if X_query_np.ndim == 1:
            X_query_np = X_query_np.reshape(1, -1)
        if X_query_np.ndim != 2:
            raise ValueError("X_query must be a 2D array or a single query vector")
        n_queries = int(X_query_np.shape[0])
        if n_queries == 0:
            return []

        target_classes = self._normalize_batch_target_classes(
            X_query=X_query_np,
            target_class=target_class,
        )
        resolved_candidate_parallelism = int(
            getattr(self, "candidate_parallelism", 1)
            if candidate_parallelism is None
            else candidate_parallelism
        )
        if resolved_candidate_parallelism <= 0:
            raise ValueError("candidate_parallelism must be positive")
        if self.query_parallelism > 1 and resolved_candidate_parallelism > 1:
            raise ValueError(
                "query_parallelism and candidate_parallelism cannot both exceed 1; "
                "choose inter-query or intra-query parallelism"
            )
        find_kwargs: Dict[str, Any] = {
            "method": method,
            "delta": delta,
            "robust_norm": robust_norm,
            "solver_maxiter": solver_maxiter,
            "fixed_dims": fixed_dims,
            "query_k_candidates": query_k_candidates,
        }
        if candidate_parallelism is not None:
            find_kwargs["candidate_parallelism"] = candidate_parallelism
        if candidate_parallel_backend is not None:
            find_kwargs["candidate_parallel_backend"] = candidate_parallel_backend
        has_directional = (
            (nondecreasing_dims is not None and len(nondecreasing_dims) > 0)
            or (nonincreasing_dims is not None and len(nonincreasing_dims) > 0)
        )
        if has_directional:
            find_kwargs["nondecreasing_dims"] = nondecreasing_dims
            find_kwargs["nonincreasing_dims"] = nonincreasing_dims

        if self.query_parallelism <= 1:
            return [
                self.find_counterfactual(
                    x_query=X_query_np[idx],
                    target_class=int(target_classes[idx]),
                    **find_kwargs,
                )
                for idx in range(n_queries)
            ]

        resolved_method = method or self.default_query_method
        normalized_robust_norm = (
            self._normalize_lp_norm(robust_norm) if robust_norm is not None else None
        )
        results: List[Optional[CounterfactualResult]] = [None] * n_queries
        grouped_indices: Dict[int, List[int]] = {}
        for idx, cls in enumerate(target_classes):
            grouped_indices.setdefault(int(cls), []).append(idx)
        work_items = [
            (idx, cls)
            for cls, indices in grouped_indices.items()
            for idx in indices
        ]

        max_workers = min(self.query_parallelism, n_queries)
        executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="certcf-query",
        )
        try:
            future_to_item = {
                executor.submit(
                    self.find_counterfactual,
                    x_query=X_query_np[idx],
                    target_class=cls,
                    **find_kwargs,
                ): (idx, cls)
                for idx, cls in work_items
            }
            if timeout_s_per_query is None or timeout_s_per_query <= 0:
                done, not_done = wait(future_to_item.keys(), return_when=ALL_COMPLETED)
            else:
                done, not_done = wait(
                    future_to_item.keys(),
                    timeout=float(timeout_s_per_query),
                    return_when=ALL_COMPLETED,
                )

            for future in done:
                idx, _ = future_to_item[future]
                results[idx] = future.result()

            if not_done:
                timeout_s = float(timeout_s_per_query or 0.0)
                for future in not_done:
                    idx, cls = future_to_item[future]
                    results[idx] = self._build_timeout_result(
                        target_class=cls,
                        method=resolved_method,
                        delta=delta,
                        robust_norm=normalized_robust_norm,
                        timeout_s_per_query=timeout_s,
                    )
                executor.shutdown(wait=False, cancel_futures=False)
                executor = None
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

        final_results = [result for result in results if result is not None]
        if len(final_results) != n_queries:
            raise RuntimeError("CertCF batch query failed to produce one result per input query")
        return final_results

    def _normalize_batch_target_classes(
        self,
        X_query: np.ndarray,
        target_class: Union[int, Sequence[int], np.ndarray],
    ) -> np.ndarray:
        """Return one target class per query, broadcasting a scalar if needed."""
        X_query_np = np.asarray(X_query, dtype=np.float32)
        if X_query_np.ndim != 2:
            raise ValueError("X_query must be a 2D array for batched target normalization")
        if np.isscalar(target_class):
            return np.full(X_query_np.shape[0], int(target_class), dtype=np.int64)
        target_classes = np.asarray(target_class, dtype=np.int64).reshape(-1)
        if target_classes.shape[0] != X_query_np.shape[0]:
            raise ValueError("target_class must be a scalar or provide one entry per query")
        return target_classes

    def _build_timeout_result(
        self,
        *,
        target_class: int,
        method: str,
        delta: float,
        robust_norm: Optional[Union[int, float]],
        timeout_s_per_query: float,
    ) -> CounterfactualResult:
        """Return a failure result representing a soft per-query timeout."""
        profiling: Dict[str, ProfileValue] = {
            "method": method,
            "delta": float(delta),
            "robust_norm": (
                float(robust_norm) if robust_norm == np.inf else int(robust_norm)
            ) if robust_norm is not None else (
                float(self.norm) if self.norm == np.inf else int(self.norm)
            ),
            "reason": "timeout",
            "timed_out": True,
            "timeout_s_per_query": float(timeout_s_per_query),
            "total_time_ms": 1e3 * float(timeout_s_per_query),
            "n_qp_solved": 0.0,
            "success": 0.0,
            "distance": float(np.inf),
        }
        return CounterfactualResult(
            x_cf=None,
            distance=float(np.inf),
            target_class=int(target_class),
            anchor_idx=None,
            n_qp_solved=0,
            success=False,
            profiling=profiling,
        )

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

    def _validate_class_unions(self, tol: float = 1e-9) -> None:
        """Raise if built class unions violate certified disjointness."""
        if self._class_unions is None:
            raise ValueError(
                "Polygon unions not built. Call build() with build_unions=True."
            )
        assert_no_cross_class_overlap(self._class_unions, tol=tol)

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
