#!/usr/bin/env python3
"""
This module contains a function that calculates
expectation step in the EM algorithm for a GMM
"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    X: numpy.ndarray of shape (n, d) containing the dataset
    pi: numpy.ndarray of shape (k,) containing the priors
    m: numpy.ndarray of shape (k, d) containing the means
    S: numpy.ndarray of shape (k, d, d) containing covariance matrices

    Returns:
        g: numpy.ndarray of shape (k, n) containing posterior probabilities
        log_likelihood: log-likelihood of the model
        Returns (None, None) if inputs are invalid.
    """
    if (not isinstance(X, np.ndarray)
            or not isinstance(pi, np.ndarray)
            or not isinstance(m, np.ndarray)
            or not isinstance(S, np.ndarray)):
        return None, None

    if (X.ndim != 2 or pi.ndim != 1
            or m.ndim != 2 or S.ndim != 3):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (n == 0 or d == 0 or k == 0
            or m.shape[0] != k
            or m.shape[1] != d
            or S.shape[0] != k
            or S.shape[1] != d
            or S.shape[2] != d
            or not np.issubdtype(X.dtype, np.number)
            or not np.issubdtype(pi.dtype, np.number)
            or not np.issubdtype(m.dtype, np.number)
            or not np.issubdtype(S.dtype, np.number)
            or not np.all(np.isfinite(X))
            or not np.all(np.isfinite(pi))
            or not np.all(np.isfinite(m))
            or not np.all(np.isfinite(S))
            or np.any(pi < 0)
            or not np.isclose(np.sum(pi), 1)):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    likelihoods = np.sum(g, axis=0)

    if (np.any(likelihoods <= 0)
            or not np.all(np.isfinite(likelihoods))):
        return None, None

    log_likelihood = np.sum(np.log(likelihoods))
    g = g / likelihoods

    return g, log_likelihood
