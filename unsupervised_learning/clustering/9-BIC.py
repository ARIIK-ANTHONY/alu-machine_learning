#!/usr/bin/env python3
"""
This module contains a function that finds the best number
of clusters for a GMM using the Bayesian Information Criterion
"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC.

    X: numpy.ndarray of shape (n, d) containing the dataset
    kmin: minimum number of clusters
    kmax: maximum number of clusters
    iterations: maximum number of EM iterations
    tol: tolerance for EM convergence
    verbose: determines whether EM progress is printed

    Returns:
        best_k, best_result, likelihoods, bics
        Returns four Nones if inputs are invalid.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or X.shape[0] == 0 or X.shape[1] == 0
            or not np.issubdtype(X.dtype, np.number)
            or not np.all(np.isfinite(X))):
        return None, None, None, None

    n, d = X.shape

    if (not isinstance(kmin, int) or isinstance(kmin, bool)
            or kmin < 1 or kmin >= n):
        return None, None, None, None

    if kmax is None:
        kmax = iterations
    elif (not isinstance(kmax, int) or isinstance(kmax, bool)
          or kmax <= kmin or kmax > n):
        return None, None, None, None

    if (not isinstance(iterations, int)
            or isinstance(iterations, bool) or iterations <= 0):
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    if kmax > n:
        kmax = n

    likelihoods = []
    bics = []
    best_bic = np.inf
    best_k = None
    best_result = None

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if (pi is None or m is None or S is None
                or g is None or log_likelihood is None):
            return None, None, None, None

        parameters = (
            (k - 1) + k * d + k * d * (d + 1) / 2
        )
        bic = parameters * np.log(n) - 2 * log_likelihood

        likelihoods.append(log_likelihood)
        bics.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return (best_k, best_result, np.array(likelihoods),
            np.array(bics))
