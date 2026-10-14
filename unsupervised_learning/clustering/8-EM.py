#!/usr/bin/env python3
"""
This module contains a function that perfoms
expectation maximization for a GMM
"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k,
                             iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a Gaussian Mixture Model.

    X: numpy.ndarray of shape (n, d) containing the dataset
    k: positive integer containing the number of clusters
    iterations: positive integer containing the maximum number of iterations
    tol: non-negative float containing tolerance of the log likelihood
    verbose: boolean that determines if output should be printed

    Returns:
        pi, m, S, g, log_likelihood, or five Nones on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or X.shape[0] == 0 or X.shape[1] == 0
            or not np.issubdtype(X.dtype, np.number)):
        return None, None, None, None, None

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        return None, None, None, None, None

    if (not isinstance(iterations, int)
            or isinstance(iterations, bool) or iterations <= 0):
        return None, None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)

    if g is None or log_likelihood is None:
        return None, None, None, None, None

    if verbose:
        msg = "Log Likelihood after {} iterations: {}"
        print(msg.format(0, round(log_likelihood, 5)))

    iteration_count = 0
    previous_likelihood = log_likelihood

    for i in range(iterations):
        pi, m, S = maximization(X, g)

        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, log_likelihood = expectation(X, pi, m, S)

        if g is None or log_likelihood is None:
            return None, None, None, None, None

        iteration_count = i + 1

        if verbose and iteration_count % 10 == 0:
            print(msg.format(
                iteration_count, round(log_likelihood, 5)))

        if abs(log_likelihood - previous_likelihood) <= tol:
            break

        previous_likelihood = log_likelihood

    if verbose and iteration_count % 10 != 0:
        print(msg.format(iteration_count, round(log_likelihood, 5)))

    return pi, m, S, g, log_likelihood
