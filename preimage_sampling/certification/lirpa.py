"""LiRPA orchestration and preimage approximation."""

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

    def compute_all_bounds(
        self,
        eps: float = 0.1,
        norm: int = 2,
        max_samples_per_class: int = None
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
