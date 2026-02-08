#!/usr/bin/env python3
"""Module for calculating matrix definiteness."""
import numpy as np


def definiteness(matrix):
    """Calculate the definiteness of a matrix.

    Args:
        matrix: numpy.ndarray of shape (n, n)

    Returns:
        String describing definiteness, or None

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is valid (square and 2D)
    if matrix.size == 0:
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric (within tolerance)
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    # Check definiteness based on eigenvalues
    tolerance = 1e-10
    pos_count = np.sum(eigenvalues > tolerance)
    neg_count = np.sum(eigenvalues < -tolerance)
    zero_count = np.sum(np.abs(eigenvalues) <= tolerance)

    n = len(eigenvalues)

    if pos_count == n:
        return "Positive definite"
    elif neg_count == n:
        return "Negative definite"
    elif pos_count + zero_count == n and zero_count > 0:
        return "Positive semi-definite"
    elif neg_count + zero_count == n and zero_count > 0:
        return "Negative semi-definite"
    elif pos_count > 0 and neg_count > 0:
        return "Indefinite"
    else:
        return None
