"""LiRPA orchestration and preimage approximation."""

import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict
from tqdm import tqdm

from .wrapping import WrappedModel
from .bounds import get_lower_bound, get_upper_bound


def run_lirpa(
    model: nn.Module,
    label: int,
    X: torch.Tensor,
    n_classes: int,
    device: torch.device,
    eps: float = 0.1,
    norm: int = 2
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
    wrapped = WrappedModel(model, label, device, n_labels=n_classes).to(device)

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
        method='backward',
        return_A=True,
        needed_A_dict=needed_A,
    )

    # Extract A matrices for the input
    A = A_dict[bounded_model.output_name[0]][bounded_model.input_name[0]]

    # Get lower and upper bounds
    lA = A['lA'].detach().cpu().numpy()
    lbias = A['lbias'].detach().cpu().numpy()
    uA = A['uA'].detach().cpu().numpy()
    ubias = A['ubias'].detach().cpu().numpy()

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
        cnn: bool = False
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

        # Handle different dataset formats
        if hasattr(dataset, 'tensors'):
            # TensorDataset format
            self.dataset = dataset
            self.n_classes = len(dataset.tensors[1].unique())
        elif isinstance(dataset, list):
            # List of tuples format - convert to TensorDataset
            X_list, y_list = [], []
            for x, y in dataset:
                X_list.append(x)
                y_list.append(y)
            X_tensor = torch.stack(X_list)
            y_tensor = torch.stack(y_list) if isinstance(y_list[0], torch.Tensor) else torch.tensor(y_list)
            self.dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            self.n_classes = len(y_tensor.unique())
        else:
            raise ValueError(
                "Dataset must be a TensorDataset or a list of (X, y) tuples"
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
        batch_size: int = None
    ) -> dict:
        """
        Compute LiRPA bounds for all classes.

        For each class, extracts samples belonging to that class from the
        dataset, runs LiRPA, and stores the resulting linear bounds.

        Parameters
        ----------
        eps : float, optional
            Perturbation radius (default: 0.1).
        norm : int, optional
            Lp norm for perturbation (default: 2).
        max_samples_per_class : int, optional
            Maximum number of samples to use per class. If None, use all.
        batch_size : int, optional
            Process samples in batches to save GPU memory. If None, process all at once.
            Recommended: 10-20 for MNIST on limited GPU memory.

        Returns
        -------
        dict
            Dictionary mapping label -> dict with keys:
            - 'lA': lower bound A matrices (N, k-1, d)
            - 'lbias': lower bound biases (N, k-1)
            - 'uA': upper bound A matrices (N, k-1, d)
            - 'ubias': upper bound biases (N, k-1)
            - 'X': original samples (N, d)
        """
        all_bounds = {}

        for label in tqdm(range(self.n_classes), desc="Computing bounds"):
            # Extract samples for this class
            X = self.dataset.tensors[0][self.dataset.tensors[1] == label]

            if max_samples_per_class is not None:
                X = X[:max_samples_per_class]

            # Process in batches if batch_size is specified
            if batch_size is not None and len(X) > batch_size:
                lA_list, lbias_list = [], []
                uA_list, ubias_list = [], []
                X_stored_list = []

                n_batches = (len(X) + batch_size - 1) // batch_size

                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(X))
                    X_batch = X[start_idx:end_idx].float().to(self.device)

                    # Reshape for CNN if needed
                    if self.cnn:
                        X_batch = X_batch.view(-1, 1, 28, 28)

                    # Run LiRPA on batch
                    with torch.no_grad():
                        lA, lbias, uA, ubias = run_lirpa(
                            self.model,
                            label,
                            X_batch,
                            self.n_classes,
                            self.device,
                            eps=eps,
                            norm=norm
                        )

                    # Store batch results
                    lA_list.append(lA)
                    lbias_list.append(lbias)
                    uA_list.append(uA)
                    ubias_list.append(ubias)

                    # Store X - flatten for CNN
                    X_batch_stored = X_batch.cpu().numpy()
                    if self.cnn:
                        X_batch_stored = X_batch_stored.reshape(X_batch_stored.shape[0], -1)
                    X_stored_list.append(X_batch_stored)

                    # Free GPU memory
                    del X_batch
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                # Concatenate all batches
                lA = np.concatenate(lA_list, axis=0)
                lbias = np.concatenate(lbias_list, axis=0)
                uA = np.concatenate(uA_list, axis=0)
                ubias = np.concatenate(ubias_list, axis=0)
                X_stored = np.concatenate(X_stored_list, axis=0)

            else:
                # Process all at once (original behavior)
                X = X.float().to(self.device)

                # Reshape for CNN if needed
                if self.cnn:
                    X = X.view(-1, 1, 28, 28)

                # Run LiRPA
                lA, lbias, uA, ubias = run_lirpa(
                    self.model,
                    label,
                    X,
                    self.n_classes,
                    self.device,
                    eps=eps,
                    norm=norm
                )

                # Store bounds - flatten X for CNN to make it compatible with sampler
                X_stored = X.cpu().numpy()
                if self.cnn:
                    # Flatten (N, 1, 28, 28) -> (N, 784)
                    X_stored = X_stored.reshape(X_stored.shape[0], -1)

            all_bounds[label] = {
                'lA': lA,
                'lbias': lbias,
                'uA': uA,
                'ubias': ubias,
                'X': X_stored
            }

        return all_bounds
