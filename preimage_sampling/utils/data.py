"""Data generation and model loading utilities."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def make_spiral(
    n_samples_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.1,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic spiral dataset for classification.

    Creates a multi-class spiral dataset where each class forms an arm
    of the spiral. Commonly used for visualizing neural network decision
    boundaries in 2D.

    Parameters
    ----------
    n_samples_per_class : int, optional
        Number of samples to generate per class (default: 100).
    n_classes : int, optional
        Number of spiral arms/classes (default: 3).
    noise : float, optional
        Standard deviation of Gaussian noise added to angular component (default: 0.1).
    seed : int, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples_per_class * n_classes, 2).
    y : np.ndarray
        Label vector of shape (n_samples_per_class * n_classes,).

    Examples
    --------
    >>> X, y = make_spiral(n_samples_per_class=200, n_classes=10, noise=0.1)
    >>> X.shape, y.shape
    ((2000, 2), (2000,))
    """
    if seed is not None:
        np.random.seed(seed)

    X = []
    y = []

    for class_id in range(n_classes):
        # Radial component: linearly increase from 0 to 1
        r = np.linspace(0.0, 1, n_samples_per_class)

        # Angular component: offset by class, with noise
        t = np.linspace(
            class_id * 4, (class_id + 1) * 4, n_samples_per_class
        ) + np.random.randn(n_samples_per_class) * noise

        # Convert to Cartesian coordinates
        x = r * np.sin(t)
        y_coord = r * np.cos(t)

        X.append(np.stack([x, y_coord], axis=1))
        y.append(np.full(n_samples_per_class, class_id))

    return np.vstack(X), np.concatenate(y)


def load_model(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str = 'cpu',
    strict: bool = True
) -> nn.Module:
    """
    Load model state dict from file.

    Parameters
    ----------
    model : nn.Module
        The model instance to load weights into.
    path : str or Path
        Path to the saved state dict file (.pth or .pt).
    device : torch.device or str, optional
        Device to load the model onto (default: 'cpu').
    strict : bool, optional
        Whether to strictly enforce that the keys in state_dict match (default: True).

    Returns
    -------
    nn.Module
        The model with loaded weights, set to eval mode.

    Examples
    --------
    >>> model = SimpleClassifier()
    >>> model = load_model(model, 'weights.pth', device='cuda')
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model


def save_model(
    model: nn.Module,
    path: str | Path,
    create_dir: bool = True
) -> None:
    """
    Save model state dict to file.

    Parameters
    ----------
    model : nn.Module
        The model to save.
    path : str or Path
        Destination path for the state dict file (.pth or .pt).
    create_dir : bool, optional
        Whether to create parent directories if they don't exist (default: True).

    Examples
    --------
    >>> model = SimpleClassifier()
    >>> save_model(model, 'checkpoints/model.pth')
    """
    path = Path(path)

    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)
