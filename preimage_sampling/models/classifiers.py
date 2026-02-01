"""Neural network classifier architectures."""

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
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(32 *2 *2, 10)
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
        return self.net(x)
