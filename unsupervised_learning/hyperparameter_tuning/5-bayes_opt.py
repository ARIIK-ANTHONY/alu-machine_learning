#!/usr/bin/env python3
""" Bayesian Optimization for hyperparameter tuning """

import numpy as np
GP = __import__('2-gp').GaussianProcess


def norm_pdf(x):
    """ Standard normal probability density function """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


def norm_cdf(x):
    """ Standard normal cumulative distribution function using error function """
    return 0.5 * (1 + np.vectorize(np.math.erf)(x / np.sqrt(2)))


class BayesianOptimization:
    """ Performs Bayesian optimization on a black-box function """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, ac_func='EI'):
        """ Initializes Bayesian Optimization

        Args:
            f: black-box function to optimize
            X_init: numpy.ndarray of shape (t, 1) initial input samples
            Y_init: numpy.ndarray of shape (t, 1) initial output samples
            bounds: tuple (min, max) of bounds of the search space
            ac_samples: number of samples for acquisition function
            l: length parameter for kernel
            sigma_f: standard deviation for kernel
            ac_func: acquisition function ('EI', 'PI', 'UCB')
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.ac_func = ac_func

    def acquisition(self):
        """ Calculates the acquisition function values

        Returns:
            numpy.ndarray: acquisition values for each sample point
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)
        sigma = np.maximum(sigma, 1e-9)

        if self.ac_func == 'EI':
            # Expected Improvement
            mu_sample_opt = np.min(self.gp.Y)
            with np.errstate(divide='ignore', invalid='ignore'):
                imp = mu_sample_opt - mu - 1e-9
                Z = imp / sigma
                ei = imp * norm_cdf(Z) + sigma * norm_pdf(Z)
                ei[sigma == 0.0] = 0.0
                ei[ei < 0] = 0
            return ei.flatten()

        elif self.ac_func == 'PI':
            # Probability of Improvement
            mu_sample_opt = np.min(self.gp.Y)
            with np.errstate(divide='ignore', invalid='ignore'):
                imp = mu_sample_opt - mu - 1e-9
                Z = imp / sigma
                pi = norm_cdf(Z)
            return pi.flatten()

        elif self.ac_func == 'UCB':
            # Upper Confidence Bound
            kappa = 2.0
            ucb = mu + kappa * sigma
            return ucb.flatten()

    def optimize(self, iterations=100):
        """ Optimizes the black-box function

        Args:
            iterations: maximum number of iterations to perform

        Returns:
            X_opt: numpy.ndarray of shape (1,) optimal point
            Y_opt: numpy.ndarray of shape (1,) optimal function value
        """
        for i in range(iterations):
            # Get next sample point by maximizing acquisition function
            ac_values = self.acquisition()
            X_next = self.X_s[np.argmax(ac_values)].reshape(1, -1)

            # Early stopping if point already sampled
            if np.any(np.all(self.gp.X == X_next, axis=1)):
                break

            # Evaluate black-box function at next point
            Y_next = self.f(X_next)

            # Update Gaussian Process with new sample
            self.gp.update(X_next, Y_next)

        # Find optimal point (minimum Y value)
        Y_opt_idx = np.argmin(self.gp.Y)
        X_opt = self.gp.X[Y_opt_idx].reshape(-1)
        Y_opt = self.gp.Y[Y_opt_idx].reshape(-1)

        return X_opt, Y_opt
