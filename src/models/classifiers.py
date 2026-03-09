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
    Simple feedforward neural network for tabular data classification.

    Numerical features are passed directly; categorical features are embedded
    via nn.Embedding with per-feature cardinalities.

    Parameters
    ----------
    input_types : list[str]
        List of feature types, one per column: "numerical" or "categorical".
    cardinalities : list[int]
        Number of unique values for each categorical feature, in the order they
        appear in input_types. Must have length == number of "categorical" entries.
    embedding_dim : int, optional
        Embedding dimension used for all categorical features (default: 8).
    hidden_dims : tuple[int, ...], optional
        Hidden layer dimensions (default: (64, 32)).
    num_classes : int, optional
        Number of output classes (default: 2).
    """

    def __init__(self, input_types, cardinalities, embedding_dim=8, hidden_dims=(64, 32), num_classes=2):
        super(TabularClassifier, self).__init__()
        self.input_types = input_types
        self.embeddings = nn.ModuleList()
        cat_iter = iter(cardinalities)
        # Precompute slice boundaries in the embedded representation
        self._slices = []  # (start, end) per feature column
        pos = 0
        input_dim = 0
        for t in input_types:
            if t == "numerical":
                self.embeddings.append(None)
                self._slices.append((pos, pos + 1))
                pos += 1
                input_dim += 1
            elif t == "categorical":
                cardinality = next(cat_iter)
                self.embeddings.append(nn.Embedding(cardinality, embedding_dim))
                self._slices.append((pos, pos + embedding_dim))
                pos += embedding_dim
                input_dim += embedding_dim
            else:
                raise ValueError(f"Unknown input type: {t}")
        self.embed_dim = pos
        self.bn = nn.BatchNorm1d(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map raw features to the embedded representation.

        Numerical columns are passed through as-is; categorical columns are
        looked up in their embedding table. The result is a single float tensor
        suitable for LiRPA bound propagation through self.net.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, n_features). Categorical columns must contain
            integer indices in [0, cardinality).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, embed_dim).
        """
        parts = []
        for i, (emb, t) in enumerate(zip(self.embeddings, self.input_types)):
            if t == "numerical":
                parts.append(x[:, i:i+1])
            else:
                parts.append(emb(x[:, i].long()))
        return self.bn(torch.cat(parts, dim=1))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode an embedded representation back to original feature space.

        ``embed()`` applies batch normalisation after concatenating numerical
        and categorical embedding dims.  This method inverts that BN first
        (recovering pre-BN values), then:

        * Numerical features: the pre-BN value equals the original raw value.
        * Categorical features: the pre-BN value is compared against the raw
          embedding weights (before BN) to find the nearest valid category.

        Parameters
        ----------
        z : torch.Tensor
            Shape (batch_size, embed_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, n_features). Categorical columns contain the
            nearest valid integer category index (as float).
        """
        # Invert batch normalisation: z = γ*(x_pre - μ)/√(σ²+ε) + β
        # → x_pre = (z - β)/γ * √(σ²+ε) + μ
        std = torch.sqrt(self.bn.running_var + self.bn.eps)
        x_prebn = (z - self.bn.bias) / self.bn.weight * std + self.bn.running_mean

        x_out = torch.empty(z.shape[0], len(self.input_types), device=z.device, dtype=z.dtype)
        for i, (emb, t, (start, end)) in enumerate(zip(self.embeddings, self.input_types, self._slices)):
            if t == "numerical":
                x_out[:, i] = x_prebn[:, start]
            else:
                prebn_block = x_prebn[:, start:end]          # (batch, emb_dim)
                dists = torch.cdist(prebn_block, emb.weight) # (batch, cardinality)
                x_out[:, i] = dists.argmin(dim=1).to(z.dtype)
        return x_out

    def feature_dims(self, feature_names: list, all_cols: list) -> 'np.ndarray':
        """
        Return the embedding-space dimension indices for the given feature names.

        Use this to build the ``fixed_dims`` argument for
        ``CertifiedAtlas.find_counterfactual``.

        Parameters
        ----------
        feature_names : list[str]
            Column names that should be held constant.
        all_cols : list[str]
            Ordered list of all column names (same order used at training time).

        Returns
        -------
        np.ndarray of int
            Indices into the embed_dim-dimensional space.

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
            Input tensor of shape (batch_size, n_features). Categorical columns
            must contain integer indices in [0, cardinality).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        return self.net(self.embed(x))
