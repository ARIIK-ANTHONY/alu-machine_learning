#!/usr/bin/env python3
"""
This module contains a function that
calculates total intra-cluster variance for a dataset
"""

import numpy as np


def variance(X, C):
    """Calculate the total intra-cluster variance.

    X is a numpy.ndarray of shape (n, d) containing the data points.
    C is a numpy.ndarray of shape (k, d) containing the centroids.

    Returns the total intra-cluster variance, or None if invalid.
    """
    if type(X) is not np.ndarray or type(C) is not np.ndarray:
        return None
    if X.ndim != 2 or C.ndim != 2:
        return None
    if X.shape[0] == 0 or X.shape[1] == 0 or C.shape[0] == 0:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    if not np.issubdtype(X.dtype, np.number):
        return None
    if not np.issubdtype(C.dtype, np.number):
        return None

    distances = np.sum((X - C[:, np.newaxis]) ** 2, axis=-1)
    return np.sum(np.min(distances, axis=0))
