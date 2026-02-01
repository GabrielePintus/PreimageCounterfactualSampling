"""Model wrapping utilities for certification."""

import torch
import torch.nn as nn


class WrappedModel(nn.Module):
    """
    A wrapper for a neural network model that enforces linear constraints
    on the output logits via a fixed linear layer.

    The wrapper constructs a constraint matrix C and bias b such that
    the logit of a specific class (the chosen label) is compared against
    all other classes: g_i = z_label - z_i for all i != label.

    This enables one-vs-all certification: the target label is certified
    if all g_i >= 0.

    Attributes
    ----------
    model : nn.Module
        The base model whose outputs are constrained.
    label : int
        The target class label for which constraints are applied.
    device : torch.device
        The device on which computations are performed.
    n_labels : int
        The total number of labels/classes.
    constraint_layer : nn.Linear
        A fixed linear layer implementing the constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        label: int,
        device: torch.device,
        n_labels: int = 10
    ):
        """
        Initialize the wrapped model.

        Parameters
        ----------
        model : nn.Module
            The base neural network model.
        label : int
            The class label to constrain against others.
        device : torch.device
            The device (CPU/GPU) to place tensors on.
        n_labels : int, optional
            Total number of labels/classes (default: 10).
        """
        super().__init__()
        self.model = model
        self.label = label
        self.device = device
        self.n_labels = n_labels

        self.constraint_layer = nn.Linear(
            in_features=n_labels,
            out_features=n_labels - 1,
            bias=True
        )
        self._init_weights()

    @staticmethod
    def build_C_matrix(label: int, n_labels: int = 10) -> torch.Tensor:
        """
        Construct the constraint matrix C for one-vs-all comparison.

        For label y, the matrix C has rows:
            C[i] = [0, ..., 0, 1, 0, ..., -1, ..., 0]
                             ↑              ↑
                           pos y        pos i (i≠y)

        such that C @ z = z_y - z_i for each i != y.

        Parameters
        ----------
        label : int
            The chosen class label.
        n_labels : int, optional
            The number of labels/classes (default: 10).

        Returns
        -------
        torch.Tensor
            The constraint matrix of shape (n_labels-1, n_labels).
        """
        C = -torch.eye(n_labels - 1)
        C = torch.cat(
            (C[:, :label], torch.ones(n_labels - 1, 1), C[:, label:]),
            dim=1
        )
        return C

    @classmethod
    def build_C_b(
        cls,
        label: int,
        n_labels: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Construct the constraint matrix C and bias vector b.

        Parameters
        ----------
        label : int
            The chosen class label.
        n_labels : int, optional
            The number of labels/classes (default: 10).

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            The constraint matrix C and the bias vector b (zeros).
        """
        C = cls.build_C_matrix(label, n_labels)
        b = torch.zeros(n_labels - 1)
        return C, b

    def _init_weights(self) -> None:
        """Initialize the fixed weights of the constraint layer."""
        C, b = self.build_C_b(self.label, self.n_labels)
        self.constraint_layer.weight = nn.Parameter(C, requires_grad=False)
        self.constraint_layer.bias = nn.Parameter(b, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the wrapped model with constraints.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            The constrained output g = z_label - z_i for each i != label,
            shape (batch_size, n_labels-1).
        """
        z = self.model(x)  # Model logits: (batch_size, n_labels)
        g = self.constraint_layer(z)  # Apply constraints: (batch_size, n_labels-1)
        return g.view(x.size(0), -1)  # Flatten per batch
