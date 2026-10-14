#!/usr/bin/env python3
"""
This module contains a function that
initializes variables for a Gaussian Mixture Model
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize priors, means, and covariance matrices for a GMM.

    Args:
        X: A NumPy array of shape (n, d) containing the data points.
        k: A positive integer representing the number of clusters.

    Returns:
        A tuple (pi, m, S), or (None, None, None) if inputs are invalid.
        pi has shape (k,), m has shape (k, d), and S has shape (k, d, d).
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if (X.shape[0] == 0 or X.shape[1] == 0
            or not np.issubdtype(X.dtype, np.number)):
        return None, None, None

    n, d = X.shape

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0 or k > n:
        return None, None, None

    pi = np.full(k, 1 / k)
    m, _ = kmeans(X, k)

    if m is None:
        return None, None, None

    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
