"""Utilities for extracting bounds from LiRPA results."""

import numpy as np


def get_lower_bound(A: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract lower bound matrices from auto_LiRPA result dictionary.

    Parameters
    ----------
    A : dict
        The A dictionary returned by auto_LiRPA's compute_bounds with return_A=True.
        Expected structure: A['/x'] contains 'lA' and 'lbias' tensors.

    Returns
    -------
    lA : np.ndarray
        Lower bound A matrix, shape depends on network architecture.
    lbias : np.ndarray
        Lower bound bias vector, flattened to 1D.
    """
    lA = A['/x']['lA'].detach().cpu().numpy().squeeze()
    lbias = A['/x']['lbias'].detach().cpu().numpy().reshape(-1)
    return lA, lbias


def get_upper_bound(A: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract upper bound matrices from auto_LiRPA result dictionary.

    Parameters
    ----------
    A : dict
        The A dictionary returned by auto_LiRPA's compute_bounds with return_A=True.
        Expected structure: A['/x'] contains 'uA' and 'ubias' tensors.

    Returns
    -------
    uA : np.ndarray
        Upper bound A matrix, shape depends on network architecture.
    ubias : np.ndarray
        Upper bound bias vector, flattened to 1D.
    """
    uA = A['/x']['uA'].detach().cpu().numpy().squeeze()
    ubias = A['/x']['ubias'].detach().cpu().numpy().reshape(-1)
    return uA, ubias
