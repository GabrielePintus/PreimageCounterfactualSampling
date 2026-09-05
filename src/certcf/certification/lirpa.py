"""LiRPA orchestration and preimage approximation."""

import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from tqdm import tqdm

from .wrapping import WrappedModel
from .bounds import get_lower_bound, get_upper_bound


_OPTIMIZED_LIRPA_METHODS = {"crown-optimized", "alpha-crown", "forward-optimized"}


def _lirpa_grad_context(lirpa_method: str):
    """Use gradients only for auto_LiRPA methods that optimize relaxation parameters."""
    method = str(lirpa_method or "").strip().lower()
    return torch.enable_grad() if method in _OPTIMIZED_LIRPA_METHODS else torch.no_grad()


def _center_certification_slack(
    lA: np.ndarray,
    lbias: np.ndarray,
    center: np.ndarray,
    *,
    classification_margin: float = 0.0,
) -> float:
    """Return min center slack for LiRPA lower-bound halfspaces."""
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    A = np.asarray(lA, dtype=np.float64).reshape(-1, center.shape[0])
    b = np.asarray(lbias, dtype=np.float64).reshape(-1)
    if A.shape[0] == 0:
        return float("inf")
    return float(np.min(A @ center + b - classification_margin))


def run_lirpa(
    model: nn.Module,
    label: int,
    X: torch.Tensor,
    n_classes: int,
    device: torch.device,
    eps: float | torch.Tensor = 0.1,
    norm: int = 2,
    dtype: torch.dtype = torch.float32,
    lirpa_method: str = "backward",
) -> tuple:
    """
    Run LiRPA backward-mode bound propagation for a single class.

    This function wraps the model with one-vs-all constraints for the target
    label, creates a bounded input tensor with Lp perturbation, and computes
    linear bounds A and b such that:
        A * x + b <= g(x)  (lower bound)
        A * x + b >= g(x)  (upper bound)
    where g(x) = z_label(x) - z_i(x) for all i != label.

    Parameters
    ----------
    model : nn.Module
        The neural network classifier.
    label : int
        The target class label for certification.
    X : torch.Tensor
        Input samples of shape (N, d) for which to compute bounds.
    n_classes : int
        Total number of classes in the classification problem.
    device : torch.device
        Device to run computations on.
    eps : float, optional
        Perturbation radius for Lp ball (default: 0.1).
    norm : int, optional
        Lp norm for perturbation (default: 2 for L2).

    Returns
    -------
    lA : np.ndarray
        Lower bound A matrices, shape (N, n_classes-1, d).
    lbias : np.ndarray
        Lower bound biases, shape (N, n_classes-1).
    uA : np.ndarray
        Upper bound A matrices, shape (N, n_classes-1, d).
    ubias : np.ndarray
        Upper bound biases, shape (N, n_classes-1).
    """
    # Wrap model with one-vs-all constraint layer and move to device
    wrapped = WrappedModel(model, label, device, n_labels=n_classes).to(device).to(dtype)
    wrapped.eval()

    # Cast input to match model dtype
    X = X.to(dtype)

    # Create perturbation specification
    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    X_bounded = BoundedTensor(X, ptb)

    # Create bounded module for LiRPA
    bounded_model = BoundedModule(wrapped, X_bounded)

    # Forward pass to initialize bounds
    _ = bounded_model(X_bounded)

    # Specify which bounds to compute
    needed_A = defaultdict(set)
    needed_A[bounded_model.output_name[0]].add(bounded_model.input_name[0])

    # Compute bounds using backward mode
    _, _, A_dict = bounded_model.compute_bounds(
        x=(X_bounded,),
        method=lirpa_method,
        # method='alpha-crown',      # alpha-CROWN: slower but much tighter
        # method='crown-optimized',
        return_A=True,
        needed_A_dict=needed_A,
    )

    # Extract A matrices for the input
    A = A_dict[bounded_model.output_name[0]][bounded_model.input_name[0]]

    # Get lower and upper bounds (always convert to float32 for downstream numpy)
    lA = A['lA'].detach().cpu().float().numpy()
    lbias = A['lbias'].detach().cpu().float().numpy()
    uA = A['uA'].detach().cpu().float().numpy()
    ubias = A['ubias'].detach().cpu().float().numpy()

    # Handle both 2D (N, d) and 4D (N, C, H, W) input shapes
    N = X.shape[0]
    k_m1 = n_classes - 1

    if len(X.shape) == 2:
        # Simple feedforward: (N, d)
        d = X.shape[1]
        return (
            lA.reshape(N, k_m1, d),
            lbias.reshape(N, k_m1),
            uA.reshape(N, k_m1, d),
            ubias.reshape(N, k_m1)
        )
    elif len(X.shape) == 4:
        # CNN: (N, C, H, W) - flatten spatial dimensions
        C, H, W = X.shape[1:]
        d = C * H * W
        return (
            lA.reshape(N, k_m1, d),
            lbias.reshape(N, k_m1),
            uA.reshape(N, k_m1, d),
            ubias.reshape(N, k_m1)
        )
    else:
        raise ValueError(f"Unexpected input shape: {X.shape}")


class ReusableLiRPASession:
    """Reuse auto-LiRPA's traced bounded graph for repeated same-class calls."""

    def __init__(
        self,
        model: nn.Module,
        label: int,
        example: torch.Tensor,
        n_classes: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.n_classes = int(n_classes)
        self.device = device
        self.dtype = dtype
        wrapped = WrappedModel(model, label, device, n_labels=n_classes).to(device).to(dtype).eval()
        example = example.to(device=device, dtype=dtype)
        initial = BoundedTensor(example, PerturbationLpNorm(norm=2, eps=0.0))
        self.bounded_model = BoundedModule(wrapped, initial)

    def run(
        self,
        X: torch.Tensor,
        *,
        eps: float | torch.Tensor,
        norm: int,
        lirpa_method: str,
    ) -> tuple:
        X = X.to(device=self.device, dtype=self.dtype)
        # auto-LiRPA computes the L1 dual as 1/(1-1/p), which is correctly
        # infinity for p=1 but emits a NumPy divide-by-zero warning.
        with np.errstate(divide="ignore"):
            eps_value = (
                eps.to(device=self.device, dtype=self.dtype)
                if isinstance(eps, torch.Tensor)
                else float(eps)
            )
            bounded_x = BoundedTensor(X, PerturbationLpNorm(norm=norm, eps=eps_value))
            _ = self.bounded_model(bounded_x)
            needed_A = defaultdict(set)
            output_name = self.bounded_model.output_name[0]
            input_name = self.bounded_model.input_name[0]
            needed_A[output_name].add(input_name)
            _, _, A_dict = self.bounded_model.compute_bounds(
                x=(bounded_x,),
                method=lirpa_method,
                return_A=True,
                needed_A_dict=needed_A,
            )
        A = A_dict[output_name][input_name]
        N = X.shape[0]
        dimension = int(np.prod(X.shape[1:]))
        specifications = self.n_classes - 1
        return (
            A["lA"].detach().cpu().float().numpy().reshape(N, specifications, dimension),
            A["lbias"].detach().cpu().float().numpy().reshape(N, specifications),
            A["uA"].detach().cpu().float().numpy().reshape(N, specifications, dimension),
            A["ubias"].detach().cpu().float().numpy().reshape(N, specifications),
        )


class PreimageApproximation:
    """
    Orchestrates the preimage approximation process using LiRPA.

    This class manages the workflow of computing certified inner polytopes
    for each class in a classification dataset by:
    1. Wrapping the model with one-vs-all constraints
    2. Setting up perturbation specifications
    3. Running LiRPA to compute linear bounds
    4. Returning bounds for all samples in each class

    Attributes
    ----------
    model : nn.Module
        The neural network classifier.
    dataset : torch.utils.data.TensorDataset
        Dataset containing (X, y) tuples.
    n_classes : int
        Number of classes in the dataset.
    cnn : bool
        Whether the model is a CNN (affects reshaping).
    device : torch.device
        Device for computations.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        device: torch.device,
        cnn: bool = False,
        model_input_shape: tuple[int, ...] | None = None,
        reuse_lirpa_graph: bool = False,
    ):
        """
        Initialize the PreimageApproximation class.

        Parameters
        ----------
        model : nn.Module
            The model to generate preimages for.
        dataset : torch.utils.data.TensorDataset or list
            The dataset to use, expected to have (X, y) tensors.
            Can be a TensorDataset or a list of (X, y) tuples.
        device : torch.device
            Device for computation.
        cnn : bool, optional
            Whether the model is a CNN (default: False).
        """
        self.model = model
        self.device = device
        self.cnn = cnn
        self.model_input_shape = (
            None if model_input_shape is None else tuple(int(v) for v in model_input_shape)
        )
        self.reuse_lirpa_graph = bool(reuse_lirpa_graph)
        self._lirpa_sessions: dict[int, ReusableLiRPASession] = {}
        self._empty_cuda_cache = True

        # Handle different dataset formats
        if hasattr(dataset, 'tensors'):
            # TensorDataset format
            self.dataset = dataset
        elif isinstance(dataset, list):
            # List of tuples format - convert to TensorDataset
            X_list, y_list = [], []
            for x, y in dataset:
                X_list.append(x)
                y_list.append(y)
            X_tensor = torch.stack(X_list)
            y_tensor = torch.stack(y_list) if isinstance(y_list[0], torch.Tensor) else torch.tensor(y_list)
            self.dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        else:
            raise ValueError(
                "Dataset must be a TensorDataset or a list of (X, y) tuples"
            )

        labels_tensor = self.dataset.tensors[1]
        unique_labels = torch.unique(labels_tensor).detach().cpu().tolist()
        self.class_labels = [int(label) for label in sorted(unique_labels)]
        self.label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
        self.n_classes = len(self.class_labels)

    def _reshape_cnn_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Restore an explicitly configured CNN shape from flattened inputs."""
        if not self.cnn or batch.ndim != 2:
            return batch
        if self.model_input_shape is not None:
            return batch.view(batch.shape[0], *self.model_input_shape)
        flat_dim = int(batch.shape[1])
        side = int(round(np.sqrt(flat_dim)))
        if side * side != flat_dim:
            raise ValueError(
                "Flattened CNN inputs require model_input_shape unless they are square grayscale images"
            )
        return batch.view(batch.shape[0], 1, side, side)

    def _run_bounds(
        self,
        label: int,
        X: torch.Tensor,
        *,
        eps: float | torch.Tensor,
        norm: int,
        dtype: torch.dtype,
        lirpa_method: str,
    ) -> tuple:
        if not self.reuse_lirpa_graph:
            return run_lirpa(
                self.model,
                label,
                X,
                self.n_classes,
                self.device,
                eps=eps,
                norm=norm,
                dtype=dtype,
                lirpa_method=lirpa_method,
            )
        if label not in self._lirpa_sessions:
            self._lirpa_sessions[label] = ReusableLiRPASession(
                self.model, label, X[:1], self.n_classes, self.device, dtype
            )
        return self._lirpa_sessions[label].run(
            X, eps=eps, norm=norm, lirpa_method=lirpa_method
        )

    def _compute_batched_variable_eps_bounds(
        self,
        *,
        label_index: int,
        X: torch.Tensor,
        eps_label: np.ndarray,
        batch_size: int,
        norm: int,
        dtype: torch.dtype,
        lirpa_method: str,
        classification_margin: float,
        adaptive_eps: bool,
        adaptive_eps_shrink_factor: float,
        adaptive_eps_max_shrinks: int,
        adaptive_eps_min: float,
        adaptive_eps_center_tol: float,
    ) -> tuple:
        """Certify FC anchors in wavefront batches with per-anchor radii."""
        sample_count = len(X)
        eps_current = np.asarray(eps_label, dtype=np.float64).copy()
        n_shrinks = np.zeros(sample_count, dtype=np.int64)
        center_slack = np.full(sample_count, np.nan, dtype=np.float64)
        center_certified = np.zeros(sample_count, dtype=bool)
        lA_by_sample = [None] * sample_count
        lbias_by_sample = [None] * sample_count
        uA_by_sample = [None] * sample_count
        ubias_by_sample = [None] * sample_count
        X_by_sample = [None] * sample_count

        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            active = np.arange(start, stop, dtype=np.int64)
            while len(active):
                X_active = X[active.tolist()].to(device=self.device, dtype=dtype)
                eps_active = torch.as_tensor(
                    eps_current[active],
                    device=self.device,
                    dtype=dtype,
                ).reshape(-1, 1)
                with _lirpa_grad_context(lirpa_method):
                    lA, lbias, uA, ubias = self._run_bounds(
                        label_index,
                        X_active,
                        eps=eps_active,
                        norm=norm,
                        dtype=dtype,
                        lirpa_method=lirpa_method,
                    )
                X_stored = X_active.detach().cpu().numpy()
                retry = []
                for local_index, sample_index in enumerate(active):
                    sample_index = int(sample_index)
                    slack = _center_certification_slack(
                        lA[local_index:local_index + 1],
                        lbias[local_index:local_index + 1],
                        X_stored[local_index],
                        classification_margin=classification_margin,
                    )
                    certified = slack >= -adaptive_eps_center_tol
                    should_retry = (
                        adaptive_eps
                        and not certified
                        and n_shrinks[sample_index] < adaptive_eps_max_shrinks
                        and eps_current[sample_index] * adaptive_eps_shrink_factor
                        >= adaptive_eps_min
                    )
                    if should_retry:
                        eps_current[sample_index] *= adaptive_eps_shrink_factor
                        n_shrinks[sample_index] += 1
                        retry.append(sample_index)
                        continue

                    lA_by_sample[sample_index] = lA[local_index:local_index + 1]
                    lbias_by_sample[sample_index] = lbias[local_index:local_index + 1]
                    uA_by_sample[sample_index] = uA[local_index:local_index + 1]
                    ubias_by_sample[sample_index] = ubias[local_index:local_index + 1]
                    X_by_sample[sample_index] = X_stored[local_index:local_index + 1]
                    center_slack[sample_index] = slack
                    center_certified[sample_index] = certified
                active = np.asarray(retry, dtype=np.int64)

        return (
            np.concatenate(lA_by_sample, axis=0),
            np.concatenate(lbias_by_sample, axis=0),
            np.concatenate(uA_by_sample, axis=0),
            np.concatenate(ubias_by_sample, axis=0),
            np.concatenate(X_by_sample, axis=0),
            eps_current,
            n_shrinks,
            np.zeros(sample_count, dtype=np.int64),
            center_slack,
            center_certified,
        )

    @staticmethod
    def _partition_cnn_radius_buckets(
        eps_values: np.ndarray,
        *,
        maximum_size: int,
        maximum_relative_inflation: float,
    ) -> list[np.ndarray]:
        """Group nearby radii while bounding inflation by the bucket maximum."""
        eps_values = np.asarray(eps_values, dtype=np.float64).reshape(-1)
        maximum_size = int(maximum_size)
        maximum_relative_inflation = float(maximum_relative_inflation)
        if maximum_size <= 0:
            raise ValueError("maximum_size must be positive")
        if maximum_relative_inflation < 0.0:
            raise ValueError("maximum_relative_inflation must be non-negative")
        if len(eps_values) == 0:
            return []

        order = np.argsort(eps_values, kind="stable")
        buckets: list[np.ndarray] = []
        start = 0
        while start < len(order):
            first_radius = float(eps_values[order[start]])
            relative_limit = first_radius * (1.0 + maximum_relative_inflation)
            stop = start + 1
            while (
                stop < len(order)
                and stop - start < maximum_size
                and float(eps_values[order[stop]]) <= relative_limit
            ):
                stop += 1
            buckets.append(order[start:stop])
            start = stop
        return buckets

    def _compute_bucketed_cnn_bounds(
        self,
        *,
        label_index: int,
        label: int,
        X: torch.Tensor,
        eps_label: np.ndarray,
        bucket_batch_size: int,
        maximum_relative_inflation: float,
        norm: int,
        dtype: torch.dtype,
        lirpa_method: str,
        classification_margin: float,
        adaptive_eps: bool,
        adaptive_eps_shrink_factor: float,
        adaptive_eps_max_shrinks: int,
        adaptive_eps_min: float,
        adaptive_eps_center_tol: float,
        show_progress: bool,
    ) -> tuple:
        """Certify CNN anchors in sound common-radius buckets.

        Each bucket is evaluated at its largest radius. The resulting affine
        bounds are therefore valid for every smaller original ball in that
        bucket. Anchors whose centers do not certify under this conservative
        relaxation fall back to the exact individual-radius path.
        """
        eps_initial = np.asarray(eps_label, dtype=np.float64).copy()
        eps_final = eps_initial.copy()
        sample_count = len(X)
        n_shrinks = np.zeros(sample_count, dtype=np.int64)
        n_binary_steps = np.zeros(sample_count, dtype=np.int64)
        center_slack = np.full(sample_count, np.nan, dtype=np.float64)
        center_certified = np.zeros(sample_count, dtype=bool)
        lA_by_sample = [None] * sample_count
        lbias_by_sample = [None] * sample_count
        uA_by_sample = [None] * sample_count
        ubias_by_sample = [None] * sample_count
        X_by_sample = [None] * sample_count

        buckets = self._partition_cnn_radius_buckets(
            eps_initial,
            maximum_size=bucket_batch_size,
            maximum_relative_inflation=maximum_relative_inflation,
        )
        fallback_indices: list[int] = []
        maximum_observed_inflation = 0.0

        iterator = tqdm(
            buckets,
            desc=f"Class {int(label)} radius buckets",
            unit="bucket",
            leave=False,
            disable=not show_progress,
        )
        for indices in iterator:
            index_list = indices.tolist()
            bucket_radii = eps_initial[indices]
            common_radius = float(np.max(bucket_radii))
            positive = bucket_radii > 0.0
            if np.any(positive):
                maximum_observed_inflation = max(
                    maximum_observed_inflation,
                    float(np.max(common_radius / bucket_radii[positive] - 1.0)),
                )

            X_batch = X[index_list].to(device=self.device, dtype=dtype)
            X_batch = self._reshape_cnn_batch(X_batch)
            with _lirpa_grad_context(lirpa_method):
                lA, lbias, uA, ubias = self._run_bounds(
                    label_index,
                    X_batch,
                    eps=common_radius,
                    norm=norm,
                    dtype=dtype,
                    lirpa_method=lirpa_method,
                )
            X_stored = X_batch.detach().cpu().numpy().reshape(len(indices), -1)

            for local_index, sample_index_value in enumerate(indices):
                sample_index = int(sample_index_value)
                slack = _center_certification_slack(
                    lA[local_index:local_index + 1],
                    lbias[local_index:local_index + 1],
                    X_stored[local_index],
                    classification_margin=classification_margin,
                )
                certified = slack >= -adaptive_eps_center_tol
                if not certified:
                    fallback_indices.append(sample_index)
                    continue
                lA_by_sample[sample_index] = lA[local_index:local_index + 1]
                lbias_by_sample[sample_index] = lbias[local_index:local_index + 1]
                uA_by_sample[sample_index] = uA[local_index:local_index + 1]
                ubias_by_sample[sample_index] = ubias[local_index:local_index + 1]
                X_by_sample[sample_index] = X_stored[local_index:local_index + 1]
                center_slack[sample_index] = slack
                center_certified[sample_index] = True

            del X_batch
            if self.device.type == "cuda" and self._empty_cuda_cache:
                torch.cuda.empty_cache()

        for sample_index in fallback_indices:
            eps_i = float(eps_initial[sample_index])
            shrink_count = 0
            while True:
                X_single = X[sample_index:sample_index + 1].to(
                    device=self.device,
                    dtype=dtype,
                )
                X_single = self._reshape_cnn_batch(X_single)
                with _lirpa_grad_context(lirpa_method):
                    lA_i, lbias_i, uA_i, ubias_i = self._run_bounds(
                        label_index,
                        X_single,
                        eps=eps_i,
                        norm=norm,
                        dtype=dtype,
                        lirpa_method=lirpa_method,
                    )
                X_single_stored = X_single.detach().cpu().numpy().reshape(1, -1)
                slack = _center_certification_slack(
                    lA_i,
                    lbias_i,
                    X_single_stored[0],
                    classification_margin=classification_margin,
                )
                certified = slack >= -adaptive_eps_center_tol
                del X_single
                if self.device.type == "cuda" and self._empty_cuda_cache:
                    torch.cuda.empty_cache()

                if certified or not adaptive_eps:
                    break
                should_retry = (
                    shrink_count < adaptive_eps_max_shrinks
                    and eps_i * adaptive_eps_shrink_factor >= adaptive_eps_min
                )
                if not should_retry:
                    break
                eps_i *= adaptive_eps_shrink_factor
                shrink_count += 1

            lA_by_sample[sample_index] = lA_i
            lbias_by_sample[sample_index] = lbias_i
            uA_by_sample[sample_index] = uA_i
            ubias_by_sample[sample_index] = ubias_i
            X_by_sample[sample_index] = X_single_stored
            eps_final[sample_index] = eps_i
            n_shrinks[sample_index] = shrink_count
            center_slack[sample_index] = slack
            center_certified[sample_index] = certified

        if any(value is None for value in lA_by_sample):
            raise RuntimeError("CNN radius batching did not produce every anchor bound")

        diagnostics = {
            "cnn_radius_bucket_count": int(len(buckets)),
            "cnn_radius_bucket_fallback_count": int(len(fallback_indices)),
            "cnn_radius_bucket_max_size": int(
                max((len(bucket) for bucket in buckets), default=0)
            ),
            "cnn_radius_bucket_mean_size": float(
                sample_count / max(len(buckets), 1)
            ),
            "cnn_radius_bucket_max_relative_inflation": float(
                maximum_observed_inflation
            ),
        }
        return (
            np.concatenate(lA_by_sample, axis=0),
            np.concatenate(lbias_by_sample, axis=0),
            np.concatenate(uA_by_sample, axis=0),
            np.concatenate(ubias_by_sample, axis=0),
            np.concatenate(X_by_sample, axis=0),
            eps_final,
            n_shrinks,
            n_binary_steps,
            center_slack,
            center_certified,
            diagnostics,
        )

    def estimate_memory_usage(
        self,
        n_samples_per_class: int,
        verbose: bool = True
    ) -> dict:
        """
        Estimate GPU memory usage and recommend batch size.

        Parameters
        ----------
        n_samples_per_class : int
            Number of samples to process per class.
        verbose : bool, optional
            Print recommendations (default: True).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'memory_per_sample_mb': Estimated GPU memory per sample (MB)
            - 'total_memory_mb': Estimated total GPU memory for all samples (MB)
            - 'recommended_batch_size': Recommended batch size for limited GPU
        """
        # Get sample shape
        sample_shape = self.dataset.tensors[0][0].shape
        if self.cnn:
            # CNN: (C, H, W)
            sample_size = np.prod(sample_shape)
        else:
            # Feedforward: (d,)
            sample_size = sample_shape[0]

        # Rough estimation:
        # - Input: sample_size * 4 bytes (float32)
        # - LiRPA bounds: ~10x input size (conservative estimate)
        # - Model activations: ~5x input size
        bytes_per_sample = sample_size * 4 * 15  # Total ~15x multiplier
        mb_per_sample = bytes_per_sample / (1024 ** 2)

        total_memory_mb = mb_per_sample * n_samples_per_class * self.n_classes

        # Get available GPU memory if using CUDA
        if self.device.type == 'cuda':
            gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)
            gpu_memory_available = gpu_memory_total * 0.8  # Use 80% to be safe
            recommended_batch = max(1, int(gpu_memory_available / (mb_per_sample * 20)))
        else:
            gpu_memory_total = None
            recommended_batch = n_samples_per_class  # No limit for CPU

        if verbose:
            print(f"Memory Estimation for {n_samples_per_class} samples/class:")
            print(f"  Sample shape: {sample_shape}")
            print(f"  Estimated memory per sample: {mb_per_sample:.2f} MB")
            print(f"  Total estimated memory: {total_memory_mb:.2f} MB")
            if self.device.type == 'cuda':
                print(f"  GPU total memory: {gpu_memory_total:.2f} MB")
                print(f"  Recommended batch size: {recommended_batch}")
                if total_memory_mb > gpu_memory_available:
                    print(f"  ⚠️  WARNING: Estimated usage exceeds available GPU memory!")
                    print(f"     Use batch_size={recommended_batch} in compute_all_bounds()")
            else:
                print(f"  Running on CPU (no memory limit)")

        return {
            'memory_per_sample_mb': mb_per_sample,
            'total_memory_mb': total_memory_mb,
            'recommended_batch_size': recommended_batch
        }

    def compute_all_bounds(
        self,
        eps: float = 0.1,
        norm: int = 2,
        max_samples_per_class: int = None,
        batch_size: int = None,
        dtype: torch.dtype = torch.float32,
        eps_array: np.ndarray = None,
        lirpa_method: str = "backward",
        classification_margin: float = 0.0,
        adaptive_eps: bool = False,
        adaptive_eps_shrink_factor: float = 0.5,
        adaptive_eps_max_shrinks: int = 8,
        adaptive_eps_min: float = 1.0e-6,
        adaptive_eps_center_tol: float = 1.0e-6,
        adaptive_eps_binary_search_steps: int = 0,
        precomputed_bounds: dict | None = None,
        class_completed_callback=None,
        build_parallelism: int = 1,
        cnn_radius_batch_size: int = 1,
        cnn_radius_batch_max_relative_inflation: float = 0.0,
        show_progress: bool = True,
    ) -> dict:
        """
        Compute LiRPA bounds for all classes.

        For each class, extracts samples belonging to that class from the
        dataset, runs LiRPA, and stores the resulting linear bounds.

        Parameters
        ----------
        eps : float, optional
            Perturbation radius used for all samples when ``eps_array`` is not
            provided (default: 0.1).
        norm : int, optional
            Lp norm for perturbation (default: 2).
        max_samples_per_class : int, optional
            Maximum number of samples to use per class. If None, use all.
        batch_size : int, optional
            Process samples in batches to save GPU memory. If None, process all at once.
            Recommended: 10-20 for MNIST on limited GPU memory.
            Fully connected inputs with varying ``eps_array`` values are
            processed in vectorized wavefront batches. Convolutional inputs
            currently fall back to one sample at a time because auto-LiRPA's
            convolution interval propagation requires a scalar epsilon.
        eps_array : np.ndarray, optional
            Per-sample epsilon values aligned with the full dataset, shape
            ``(N_total,)``.  When provided, each sample is certified with its
            own epsilon.  If all values in a class are equal the existing batch
            path is used. Otherwise fully connected inputs use variable-radius
            batches and convolutional inputs are processed individually.
        adaptive_eps : bool, optional
            If True, shrink each epsilon until the center satisfies the LiRPA
            lower-bound halfspaces or the retry budget is exhausted. For fully
            connected inputs, anchors needing the same retry wave are batched.
        adaptive_eps_binary_search_steps : int, optional
            If positive, refine the bracket between the first certified epsilon
            and the previous failed epsilon using this many bisection steps.
        build_parallelism : int, optional
            Number of independent class shards. Each shard owns a copied model
            and reusable LiRPA sessions. The default of one preserves serial
            execution.
        cnn_radius_batch_size : int, optional
            Maximum number of CNN anchors certified together using a common
            conservative radius. A value of one disables radius bucketing.
        cnn_radius_batch_max_relative_inflation : float, optional
            Maximum relative increase from an anchor radius to its bucket's
            common radius. Bounds computed on the larger ball remain sound on
            the original ball. Failed centers fall back to individual bounds.

        Returns
        -------
        dict
            Dictionary mapping label -> dict with keys:
            - 'lA': lower bound A matrices (N, k-1, d)
            - 'lbias': lower bound biases (N, k-1)
            - 'uA': upper bound A matrices (N, k-1, d)
            - 'ubias': upper bound biases (N, k-1)
            - 'X': original samples (N, d)
            - 'eps': final per-sample epsilon values (N,)
            - 'eps_initial': initial per-sample epsilon values (N,)
            - 'adaptive_eps_n_shrinks': number of epsilon shrinks per sample
            - 'adaptive_eps_n_binary_steps': number of binary refinements per sample
            - 'adaptive_eps_center_slack': final center slack per sample
            - 'adaptive_eps_center_certified': final center-certification flag
        """
        classification_margin = float(classification_margin)
        adaptive_eps = bool(adaptive_eps)
        adaptive_eps_shrink_factor = float(adaptive_eps_shrink_factor)
        if not (0.0 < adaptive_eps_shrink_factor < 1.0):
            raise ValueError("adaptive_eps_shrink_factor must be in (0, 1)")
        adaptive_eps_max_shrinks = int(adaptive_eps_max_shrinks)
        if adaptive_eps_max_shrinks < 0:
            raise ValueError("adaptive_eps_max_shrinks must be non-negative")
        adaptive_eps_min = float(adaptive_eps_min)
        if adaptive_eps_min < 0.0:
            raise ValueError("adaptive_eps_min must be non-negative")
        adaptive_eps_center_tol = float(adaptive_eps_center_tol)
        if adaptive_eps_center_tol < 0.0:
            raise ValueError("adaptive_eps_center_tol must be non-negative")
        adaptive_eps_binary_search_steps = int(adaptive_eps_binary_search_steps)
        if adaptive_eps_binary_search_steps < 0:
            raise ValueError("adaptive_eps_binary_search_steps must be non-negative")
        build_parallelism = int(build_parallelism)
        if build_parallelism <= 0:
            raise ValueError("build_parallelism must be positive")
        cnn_radius_batch_size = int(cnn_radius_batch_size)
        if cnn_radius_batch_size <= 0:
            raise ValueError("cnn_radius_batch_size must be positive")
        cnn_radius_batch_max_relative_inflation = float(
            cnn_radius_batch_max_relative_inflation
        )
        if cnn_radius_batch_max_relative_inflation < 0.0:
            raise ValueError(
                "cnn_radius_batch_max_relative_inflation must be non-negative"
            )

        all_bounds = dict(precomputed_bounds or {})
        labels_tensor = self.dataset.tensors[1]

        remaining_labels = [
            int(label) for label in self.class_labels if int(label) not in all_bounds
        ]
        if build_parallelism > 1 and len(remaining_labels) > 1:
            workers = min(build_parallelism, len(remaining_labels))
            shards = [remaining_labels[index::workers] for index in range(workers)]
            callback_lock = Lock()

            def shard_completed_callback(label: int, values: dict) -> None:
                if class_completed_callback is None:
                    return
                with callback_lock:
                    class_completed_callback(label, values)

            worker_inputs = []
            for shard in shards:
                worker = PreimageApproximation(
                    copy.deepcopy(self.model),
                    self.dataset,
                    self.device,
                    cnn=self.cnn,
                    model_input_shape=self.model_input_shape,
                    reuse_lirpa_graph=self.reuse_lirpa_graph,
                )
                worker.class_labels = list(shard)
                worker.label_to_index = dict(self.label_to_index)
                worker.n_classes = int(self.n_classes)
                # Emptying the global CUDA allocator cache from concurrent
                # workers serializes otherwise independent LiRPA calls.
                worker._empty_cuda_cache = False
                worker_inputs.append(worker)

            child_kwargs = {
                "eps": eps,
                "norm": norm,
                "max_samples_per_class": max_samples_per_class,
                "batch_size": batch_size,
                "dtype": dtype,
                "eps_array": eps_array,
                "lirpa_method": lirpa_method,
                "classification_margin": classification_margin,
                "adaptive_eps": adaptive_eps,
                "adaptive_eps_shrink_factor": adaptive_eps_shrink_factor,
                "adaptive_eps_max_shrinks": adaptive_eps_max_shrinks,
                "adaptive_eps_min": adaptive_eps_min,
                "adaptive_eps_center_tol": adaptive_eps_center_tol,
                "adaptive_eps_binary_search_steps": adaptive_eps_binary_search_steps,
                "precomputed_bounds": None,
                "class_completed_callback": shard_completed_callback,
                "build_parallelism": 1,
                "cnn_radius_batch_size": cnn_radius_batch_size,
                "cnn_radius_batch_max_relative_inflation": (
                    cnn_radius_batch_max_relative_inflation
                ),
                "show_progress": False,
            }

            def run_worker(worker: "PreimageApproximation") -> dict:
                if self.device.type != "cuda":
                    return worker.compute_all_bounds(**child_kwargs)
                stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(stream):
                    result = worker.compute_all_bounds(**child_kwargs)
                stream.synchronize()
                return result

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(run_worker, worker) for worker in worker_inputs]
                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(
                        iterator,
                        total=len(futures),
                        desc="Computing bound shards",
                        unit="shard",
                    )
                for future in iterator:
                    shard_bounds = future.result()
                    for label, values in shard_bounds.items():
                        all_bounds[int(label)] = values

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return {
                int(label): all_bounds[int(label)]
                for label in self.class_labels
                if int(label) in all_bounds
            }

        for label in tqdm(
            self.class_labels,
            desc="Computing bounds",
            disable=not show_progress,
        ):
            if label in all_bounds:
                continue
            label_mask = labels_tensor == label

            # Extract samples for this class
            X = self.dataset.tensors[0][label_mask]

            if max_samples_per_class is not None:
                X = X[:max_samples_per_class]
                label_mask_np = label_mask.numpy()
                # Rebuild a trimmed mask so eps_label aligns with the trimmed X
                indices = np.where(label_mask_np)[0][:max_samples_per_class]
            else:
                indices = None  # use label_mask directly

            # Determine per-sample eps for this class
            if eps_array is not None:
                if indices is not None:
                    eps_label = eps_array[indices]
                else:
                    eps_label = eps_array[label_mask.numpy()]
            else:
                eps_label = np.full(len(X), eps)
            eps_initial_label = np.asarray(eps_label, dtype=np.float64).copy()

            # Decide processing mode
            eps_is_constant = np.all(eps_label == eps_label[0])

            adaptive_n_shrinks = np.zeros(len(X), dtype=np.int64)
            adaptive_n_binary_steps = np.zeros(len(X), dtype=np.int64)
            adaptive_center_slack = np.full(len(X), np.nan, dtype=np.float64)
            adaptive_center_certified = np.zeros(len(X), dtype=bool)
            cnn_batch_diagnostics: dict[str, int | float] = {}

            if eps_is_constant and not adaptive_eps:
                # ----------------------------------------------------------------
                # Batch path (original behaviour, or constant-eps shortcut)
                # ----------------------------------------------------------------
                eps_scalar = float(eps_label[0])

                if batch_size is not None and len(X) > batch_size:
                    lA_list, lbias_list = [], []
                    uA_list, ubias_list = [], []
                    X_stored_list = []

                    n_batches = (len(X) + batch_size - 1) // batch_size

                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, len(X))
                        X_batch = X[start_idx:end_idx].to(dtype).to(self.device)

                        X_batch = self._reshape_cnn_batch(X_batch)

                        with _lirpa_grad_context(lirpa_method):
                            lA, lbias, uA, ubias = self._run_bounds(
                                self.label_to_index[int(label)], X_batch, eps=eps_scalar,
                                norm=norm, dtype=dtype, lirpa_method=lirpa_method
                            )

                        lA_list.append(lA)
                        lbias_list.append(lbias)
                        uA_list.append(uA)
                        ubias_list.append(ubias)

                        X_batch_stored = X_batch.cpu().numpy()
                        if self.cnn:
                            X_batch_stored = X_batch_stored.reshape(X_batch_stored.shape[0], -1)
                        X_stored_list.append(X_batch_stored)

                        del X_batch
                        if self.device.type == 'cuda' and self._empty_cuda_cache:
                            torch.cuda.empty_cache()

                    lA = np.concatenate(lA_list, axis=0)
                    lbias = np.concatenate(lbias_list, axis=0)
                    uA = np.concatenate(uA_list, axis=0)
                    ubias = np.concatenate(ubias_list, axis=0)
                    X_stored = np.concatenate(X_stored_list, axis=0)

                else:
                    X_dev = X.to(dtype).to(self.device)
                    X_dev = self._reshape_cnn_batch(X_dev)

                    with _lirpa_grad_context(lirpa_method):
                        lA, lbias, uA, ubias = self._run_bounds(
                            self.label_to_index[int(label)], X_dev, eps=eps_scalar,
                            norm=norm, dtype=dtype, lirpa_method=lirpa_method
                        )

                    X_stored = X_dev.cpu().numpy()
                    if self.cnn:
                        X_stored = X_stored.reshape(X_stored.shape[0], -1)

                for i in range(len(X_stored)):
                    adaptive_center_slack[i] = _center_certification_slack(
                        lA[i:i + 1],
                        lbias[i:i + 1],
                        X_stored[i],
                        classification_margin=classification_margin,
                    )
                    adaptive_center_certified[i] = (
                        adaptive_center_slack[i] >= -adaptive_eps_center_tol
                    )

            elif (
                self.cnn
                and cnn_radius_batch_size > 1
                and batch_size is not None
                and int(batch_size) > 1
                and adaptive_eps_binary_search_steps == 0
            ):
                # ----------------------------------------------------------------
                # Sound bucketed-radius CNN path. Every batch uses the largest
                # radius in a narrow bucket, so its bounds remain valid for all
                # original balls. Conservative failures use the exact path.
                # ----------------------------------------------------------------
                (
                    lA,
                    lbias,
                    uA,
                    ubias,
                    X_stored,
                    eps_label,
                    adaptive_n_shrinks,
                    adaptive_n_binary_steps,
                    adaptive_center_slack,
                    adaptive_center_certified,
                    cnn_batch_diagnostics,
                ) = self._compute_bucketed_cnn_bounds(
                    label_index=self.label_to_index[int(label)],
                    label=int(label),
                    X=X,
                    eps_label=eps_label,
                    bucket_batch_size=min(
                        int(batch_size),
                        cnn_radius_batch_size,
                    ),
                    maximum_relative_inflation=(
                        cnn_radius_batch_max_relative_inflation
                    ),
                    norm=norm,
                    dtype=dtype,
                    lirpa_method=lirpa_method,
                    classification_margin=classification_margin,
                    adaptive_eps=adaptive_eps,
                    adaptive_eps_shrink_factor=adaptive_eps_shrink_factor,
                    adaptive_eps_max_shrinks=adaptive_eps_max_shrinks,
                    adaptive_eps_min=adaptive_eps_min,
                    adaptive_eps_center_tol=adaptive_eps_center_tol,
                    show_progress=show_progress,
                )

            elif (
                not self.cnn
                and batch_size is not None
                and int(batch_size) > 1
                and adaptive_eps_binary_search_steps == 0
            ):
                # ----------------------------------------------------------------
                # Batched variable-radius path for fully connected networks.
                # auto-LiRPA supports a (batch, 1) epsilon tensor for affine
                # inputs. Failed anchors advance together through shrink waves.
                # Convolution bounds currently require a scalar epsilon and
                # therefore continue to use the per-sample path below.
                # ----------------------------------------------------------------
                (
                    lA,
                    lbias,
                    uA,
                    ubias,
                    X_stored,
                    eps_label,
                    adaptive_n_shrinks,
                    adaptive_n_binary_steps,
                    adaptive_center_slack,
                    adaptive_center_certified,
                ) = self._compute_batched_variable_eps_bounds(
                    label_index=self.label_to_index[int(label)],
                    X=X,
                    eps_label=eps_label,
                    batch_size=int(batch_size),
                    norm=norm,
                    dtype=dtype,
                    lirpa_method=lirpa_method,
                    classification_margin=classification_margin,
                    adaptive_eps=adaptive_eps,
                    adaptive_eps_shrink_factor=adaptive_eps_shrink_factor,
                    adaptive_eps_max_shrinks=adaptive_eps_max_shrinks,
                    adaptive_eps_min=adaptive_eps_min,
                    adaptive_eps_center_tol=adaptive_eps_center_tol,
                )

            else:
                # ----------------------------------------------------------------
                # Per-sample path: loop one sample at a time. Adaptive epsilon
                # also uses this path because each sample can end at a different
                # radius after center-certification retries.
                # ----------------------------------------------------------------
                lA_list, lbias_list = [], []
                uA_list, ubias_list = [], []
                X_stored_list = []
                eps_final_list = []

                for i in tqdm(
                    range(len(X)),
                    desc=f"Class {int(label)} anchors",
                    unit="anchor",
                    leave=False,
                    disable=not show_progress,
                ):
                    def _run_single_at_eps(eps_value: float):
                        X_single_local = X[i:i+1].to(dtype).to(self.device)

                        X_single_local = self._reshape_cnn_batch(X_single_local)

                        with _lirpa_grad_context(lirpa_method):
                            lA_local, lbias_local, uA_local, ubias_local = self._run_bounds(
                                self.label_to_index[int(label)], X_single_local,
                                eps=float(eps_value), norm=norm, dtype=dtype,
                                lirpa_method=lirpa_method
                            )

                        X_stored_local = X_single_local.cpu().numpy()
                        if self.cnn:
                            X_stored_local = X_stored_local.reshape(1, -1)

                        slack_local = _center_certification_slack(
                            lA_local,
                            lbias_local,
                            X_stored_local[0],
                            classification_margin=classification_margin,
                        )
                        certified_local = slack_local >= -adaptive_eps_center_tol

                        del X_single_local
                        if self.device.type == 'cuda' and self._empty_cuda_cache:
                            torch.cuda.empty_cache()

                        return (
                            lA_local,
                            lbias_local,
                            uA_local,
                            ubias_local,
                            X_stored_local,
                            slack_local,
                            certified_local,
                        )

                    eps_i = float(eps_label[i])
                    n_shrinks = 0
                    n_binary_steps = 0
                    previous_failed_eps = None

                    while True:
                        (
                            lA_i,
                            lbias_i,
                            uA_i,
                            ubias_i,
                            X_single_stored,
                            center_slack,
                            center_certified,
                        ) = _run_single_at_eps(eps_i)

                        if (
                            adaptive_eps
                            and center_certified
                            and previous_failed_eps is not None
                            and adaptive_eps_binary_search_steps > 0
                        ):
                            low_eps = float(eps_i)
                            high_eps = float(previous_failed_eps)
                            best = (
                                lA_i,
                                lbias_i,
                                uA_i,
                                ubias_i,
                                X_single_stored,
                                center_slack,
                                center_certified,
                                low_eps,
                            )
                            for _ in range(adaptive_eps_binary_search_steps):
                                mid_eps = 0.5 * (low_eps + high_eps)
                                (
                                    mid_lA,
                                    mid_lbias,
                                    mid_uA,
                                    mid_ubias,
                                    mid_X_stored,
                                    mid_slack,
                                    mid_certified,
                                ) = _run_single_at_eps(mid_eps)
                                n_binary_steps += 1
                                if mid_certified:
                                    low_eps = mid_eps
                                    best = (
                                        mid_lA,
                                        mid_lbias,
                                        mid_uA,
                                        mid_ubias,
                                        mid_X_stored,
                                        mid_slack,
                                        mid_certified,
                                        mid_eps,
                                    )
                                else:
                                    high_eps = mid_eps

                            (
                                lA_i,
                                lbias_i,
                                uA_i,
                                ubias_i,
                                X_single_stored,
                                center_slack,
                                center_certified,
                                eps_i,
                            ) = best

                        if center_certified or not adaptive_eps:
                            break

                        should_retry = (
                            n_shrinks < adaptive_eps_max_shrinks
                            and eps_i * adaptive_eps_shrink_factor >= adaptive_eps_min
                        )
                        if not should_retry:
                            break

                        previous_failed_eps = eps_i
                        eps_i *= adaptive_eps_shrink_factor
                        n_shrinks += 1

                    lA_list.append(lA_i)
                    lbias_list.append(lbias_i)
                    uA_list.append(uA_i)
                    ubias_list.append(ubias_i)
                    X_stored_list.append(X_single_stored)
                    eps_final_list.append(eps_i)
                    adaptive_n_shrinks[i] = n_shrinks
                    adaptive_n_binary_steps[i] = n_binary_steps
                    adaptive_center_slack[i] = center_slack
                    adaptive_center_certified[i] = center_certified

                lA = np.concatenate(lA_list, axis=0)
                lbias = np.concatenate(lbias_list, axis=0)
                uA = np.concatenate(uA_list, axis=0)
                ubias = np.concatenate(ubias_list, axis=0)
                X_stored = np.concatenate(X_stored_list, axis=0)
                eps_label = np.asarray(eps_final_list, dtype=np.float64)

            all_bounds[label] = {
                'lA': lA,
                'lbias': lbias,
                'uA': uA,
                'ubias': ubias,
                'X': X_stored,
                'eps': eps_label,
                'eps_initial': eps_initial_label,
                'adaptive_eps_n_shrinks': adaptive_n_shrinks,
                'adaptive_eps_n_binary_steps': adaptive_n_binary_steps,
                'adaptive_eps_center_slack': adaptive_center_slack,
                'adaptive_eps_center_certified': adaptive_center_certified,
                **cnn_batch_diagnostics,
            }
            if class_completed_callback is not None:
                class_completed_callback(int(label), all_bounds[label])

        return all_bounds
