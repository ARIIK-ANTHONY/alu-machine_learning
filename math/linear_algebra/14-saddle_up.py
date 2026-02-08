#!/usr/bin/env python3
"""Module for numpy matrix multiplication."""
import numpy as np


def np_matmul(mat1, mat2):
    """Multiply two numpy matrices.

    Args:
        mat1: First numpy array
        mat2: Second numpy array

    Returns:
        Matrix product
    """
    return np.matmul(mat1, mat2)
