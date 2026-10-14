#!/usr/bin/env python3
"""
This module contains a function that
tests for the optimum number of clusters by variance
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Analyze K-means results for different numbers of clusters.

    Args:
        X: A 2D NumPy array containing the data points.
        kmin: Minimum number of clusters to test.
        kmax: Maximum number of clusters to test.
        iterations: Maximum number of K-means iterations.

    Returns:
        A list of (centroids, cluster assignments) and a list of
        variance reductions, or (None, None) for invalid inputs.
    """
    if (type(X) is not np.ndarray or X.ndim != 2
            or X.shape[0] == 0 or X.shape[1] == 0
            or not np.issubdtype(X.dtype, np.number)):
        return None, None

    if type(kmin) is not int or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmax) is not int or kmax <= kmin:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    first_var = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        results.append((C, clss))
        var = variance(X, C)

        if var is None:
            return None, None

        if first_var is None:
            first_var = var

        d_vars.append(first_var - var)

    return results, d_vars
