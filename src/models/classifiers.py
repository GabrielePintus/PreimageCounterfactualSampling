"""Neural network classifier architectures."""

import numpy as np
import torch
import torch.nn as nn


def infer_tabular_classifier_dims_from_state_dict(state_dict) -> tuple[list[int], int]:
    """Infer all hidden widths and the output size from a tabular state dict.

    Both Lightning keys (``model.net.*``) and bare model keys (``net.*``) are
    accepted. Sorting by sequential-module index preserves compatibility with
    the historical two-hidden-layer layout.
    """
    import re

    linear_weights = []
    for key, value in state_dict.items():
        match = re.fullmatch(r"(?:model\.)?net\.(\d+)\.weight", str(key))
        if match is not None and getattr(value, "ndim", 0) == 2:
            linear_weights.append((int(match.group(1)), value))
    linear_weights.sort(key=lambda item: item[0])
    if len(linear_weights) < 2:
        raise KeyError(
            "Could not infer TabularClassifier dimensions: expected at least "
            "one hidden Linear weight and one output Linear weight."
        )
    return (
        [int(weight.shape[0]) for _, weight in linear_weights[:-1]],
        int(linear_weights[-1][1].shape[0]),
    )


class SimpleClassifier(nn.Module):
    """
    Simple feedforward neural network for classification.

    Default architecture is for 2D spiral dataset with 10 classes:
    - Input: 2 features
    - Hidden layer 1: 64 units + ReLU
    - Hidden layer 2: 32 units + ReLU
    - Output: 10 classes

    Parameters
    ----------
    input_dim : int, optional
        Input feature dimension (default: 2)
    hidden_dims : tuple[int, int], optional
        Hidden layer dimensions (default: (64, 32))
    num_classes : int, optional
        Number of output classes (default: 10)
    """

    def __init__(self, input_dim=2, hidden_dims=(64, 32), num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes)
        """
        return self.net(x)


class MNISTClassifier(nn.Module):
    """
    Convolutional neural network for MNIST digit classification.

    Architecture:
    - Conv2d(1, 16, kernel=5, stride=2, padding=2) + ReLU
    - Conv2d(16, 16, kernel=5, stride=2, padding=2) + ReLU
    - Conv2d(16, 32, kernel=3, stride=2, padding=1) + ReLU
    - Conv2d(32, 32, kernel=3, stride=2, padding=1) + ReLU
    - Flatten + ReLU
    - Linear(32*2*2, 10)

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes (default: 10)
    """

    def __init__(self, num_classes=10):
        super(MNISTClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x7x7

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x2x2
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32 * 2 * 2, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 28, 28)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes)
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        return self.classifier(h)


class LeNet5Classifier(nn.Module):
    """ReLU LeNet-5-style classifier adapted to 28x28 MNIST images.

    The two average-pooling stages reduce the convolutional representation to
    ``16 x 5 x 5`` before the three fully connected layers.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes (default: 10).
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        """Return logits for a batch shaped ``(N, 1, 28, 28)``."""
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)





class TabularClassifier(nn.Module):
    """
    Feedforward neural network for tabular data with one-hot encoded categoricals.

    Input features are a flat float vector where numerical columns are
    StandardScaler-normalized and categorical columns are one-hot encoded (OHE).
    The forward network is: ``(Linear → ReLU → Dropout) × depth → Linear``.
    Modules retain the historical ``Dropout, Linear, ReLU`` registration order
    so existing two-layer checkpoint keys remain loadable.
    The first Linear layer is lazy, so input dimensionality is inferred at
    runtime and can match either raw OHE features or PCA-projected features.

    Parameters
    ----------
    input_types : list[str]
        List of feature types per *original* column: "numerical" or "categorical".
    cardinalities : list[int]
        Number of OHE categories for each categorical feature, in order.
        Length must equal the number of "categorical" entries in input_types.
    hidden_dims : sequence[int], optional
        Hidden layer dimensions (default: (64, 32)).
    num_classes : int, optional
        Number of output classes (default: 2).
    dropout : float, optional
        Dropout probability applied after each hidden ReLU (default: 0.0 = disabled).
    """

    def __init__(self, input_types, cardinalities, hidden_dims=(64, 32), num_classes=2, dropout=0.0):
        super(TabularClassifier, self).__init__()
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims entries must be positive")
        self.input_types = input_types
        self.cardinalities = list(cardinalities)
        self.hidden_dims = hidden_dims

        # Precompute slice boundaries in the OHE feature vector.
        self._slices = []  # (start, end) per original column
        pos = 0
        cat_iter = iter(cardinalities)
        for t in input_types:
            if t == "numerical":
                self._slices.append((pos, pos + 1))
                pos += 1
            elif t == "categorical":
                card = next(cat_iter)
                self._slices.append((pos, pos + card))
                pos += card
            else:
                raise ValueError(f"Unknown input type: {t}")

        self._ohe_dim = pos
        self.embed_dim = pos
        self._input_dim_observed = None
        layers = []
        previous_dim = None
        for hidden_dim in hidden_dims:
            layers.append(nn.Dropout(dropout))
            if previous_dim is None:
                layers.append(nn.LazyLinear(hidden_dim))
            else:
                layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def _resolved_input_dim(self):
        first_linear = self.net[1]
        in_features = getattr(first_linear, "in_features", None)
        if isinstance(in_features, int) and in_features > 0:
            return in_features
        return self._input_dim_observed

    def _ensure_ohe_space(self) -> None:
        resolved_input_dim = self._resolved_input_dim()
        if resolved_input_dim is not None and resolved_input_dim != self._ohe_dim:
            raise RuntimeError(
                "decode/feature_dims are only valid in raw OHE space "
                f"(expected dim={self._ohe_dim}, got dim={resolved_input_dim})."
            )

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Snap a continuous OHE vector (from the QP solver) to valid discrete features.

        The QP solver operates in the continuous OHE space, so categorical blocks
        may have fractional values. This method snaps each block to the nearest
        valid one-hot vertex via argmax; numerical features are passed through.

        Parameters
        ----------
        z : torch.Tensor
            Shape (batch_size, n_ohe_features). Continuous output from QP solver.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, n_ohe_features). Numerical slots unchanged;
            categorical slots are 0/1 one-hot.
        """
        self._ensure_ohe_space()
        x_out = torch.zeros_like(z)
        for t, (start, end) in zip(self.input_types, self._slices):
            if t == "numerical":
                x_out[:, start] = z[:, start]
            else:
                idx = z[:, start:end].argmax(dim=1)
                x_out[:, start:end].scatter_(1, idx.unsqueeze(1), 1.0)
        return x_out

    @torch.no_grad()
    def decode_bn(self, z_bn: torch.Tensor, bn: torch.nn.BatchNorm1d) -> torch.Tensor:
        """Invert BatchNorm then argmax-snap categorical blocks.

        The QP solver (Fix 2) operates in BN-normalized space. This method
        maps z_bn back to the raw OHE space (by inverting BN), then snaps
        categorical blocks to valid one-hot vertices via decode().

        Parameters
        ----------
        z_bn : torch.Tensor
            Shape (batch_size, n_ohe_features). Point in BN-normalized space.
        bn : torch.nn.BatchNorm1d
            The BatchNorm layer with running stats from training.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, n_ohe_features). Valid discrete OHE vector.
        """
        self._ensure_ohe_space()
        weight = bn.weight.data   # (d,)
        bias   = bn.bias.data     # (d,)
        mean   = bn.running_mean  # (d,)
        var    = bn.running_var   # (d,)
        eps    = bn.eps           # scalar

        z_bn = z_bn.to(weight.device)
        # Invert BN: x = (z_bn - bias) / weight * sqrt(var + eps) + mean
        x_ohe = (z_bn - bias) / weight * torch.sqrt(var + eps) + mean
        return self.decode(x_ohe)

    def feature_dims(self, feature_names: list, all_cols: list) -> 'np.ndarray':
        """
        Return the OHE-space dimension indices for the given original feature names.

        Use this to build the ``fixed_dims`` argument for
        ``CertCFAtlas.find_counterfactual``.

        Parameters
        ----------
        feature_names : list[str]
            Column names that should be held constant.
        all_cols : list[str]
            Ordered list of all original column names (same order used at training time).

        Returns
        -------
        np.ndarray of int
            Indices into the n_ohe_features-dimensional space.

        Example
        -------
        fixed = model.feature_dims(['race', 'sex'], _ALL_COLS)
        result = atlas.find_counterfactual(z_query, target_class=1, fixed_dims=fixed)
        """
        self._ensure_ohe_space()
        dims = []
        for name in feature_names:
            i = all_cols.index(name)
            start, end = self._slices[i]
            dims.extend(range(start, end))
        return np.array(dims, dtype=int)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, n_ohe_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        if self._input_dim_observed is None:
            self._input_dim_observed = int(x.shape[-1])
        # Execute dropout after each hidden activation while keeping the module
        # registration indices used by historical checkpoints.
        for hidden_index in range(len(self.hidden_dims)):
            dropout = self.net[3 * hidden_index]
            linear = self.net[3 * hidden_index + 1]
            relu = self.net[3 * hidden_index + 2]
            x = dropout(relu(linear(x)))
        return self.net[3 * len(self.hidden_dims)](x)
