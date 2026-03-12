"""Neural network classifier architectures."""

import numpy as np
import torch
import torch.nn as nn


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





class TabularClassifier(nn.Module):
    """
    Feedforward neural network for tabular data with one-hot encoded categoricals.

    Input features are a flat float vector where numerical columns are
    StandardScaler-normalized and categorical columns are one-hot encoded (OHE).
    The network is: BatchNorm1d → Linear → ReLU → Linear → ReLU → Linear.
    LiRPA certifies the full ``net`` (including BN) directly on the OHE input.

    Parameters
    ----------
    input_types : list[str]
        List of feature types per *original* column: "numerical" or "categorical".
    cardinalities : list[int]
        Number of OHE categories for each categorical feature, in order.
        Length must equal the number of "categorical" entries in input_types.
    hidden_dims : tuple[int, ...], optional
        Hidden layer dimensions (default: (64, 32)).
    num_classes : int, optional
        Number of output classes (default: 2).
    dropout : float, optional
        Dropout probability applied after each hidden ReLU (default: 0.0 = disabled).
    """

    def __init__(self, input_types, cardinalities, hidden_dims=(64, 32), num_classes=2, dropout=0.0):
        super(TabularClassifier, self).__init__()
        self.input_types = input_types
        self.cardinalities = list(cardinalities)

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

        n_features = pos
        self.embed_dim = n_features
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes),
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
        ``CertifiedAtlas.find_counterfactual``.

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
        return self.net(x)
