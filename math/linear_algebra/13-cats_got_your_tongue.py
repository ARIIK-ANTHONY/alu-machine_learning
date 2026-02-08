#!/usr/bin/env python3
"""Module for numpy array concatenation."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate two numpy arrays along a specific axis.

    Args:
        mat1: First numpy array
        mat2: Second numpy array
        axis: Axis to concatenate along

    Returns:
        New concatenated numpy array
    """
    return np.concatenate((mat1, mat2), axis=axis)
