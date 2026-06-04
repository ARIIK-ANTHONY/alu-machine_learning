#!/usr/bin/env python3
""" Bayesian Optimization for hyperparameter tuning """

import numpy as np
GP = __import__('2-gp').GaussianProcess


def norm_pdf(x):
    """ Standard normal probability density function """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


def norm_cdf(x):
    """ Standard normal cumulative distribution function """
    return 0.5 * (1 + np.vectorize(np.math.erf)(x / np.sqrt(2)))


class BayesianOptimization:
    """ Performs Bayesian optimization on a black-box function """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """ Initializes Bayesian Optimization """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ Calculates the acquisition function values """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.flatten()
        sigma = np.maximum(sigma, 1e-9)
        mu = mu.flatten()

        # Expected Improvement
        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
        else:
            mu_sample_opt = np.max(self.gp.Y)

        with np.errstate(divide='ignore', invalid='ignore'):
            imp = mu_sample_opt - mu - self.xsi
            Z = imp / sigma
            ei = imp * norm_cdf(Z) + sigma * norm_pdf(Z)
            ei[sigma == 0.0] = 0.0
            ei[ei < 0] = 0

        X_next = self.X_s[np.argmax(ei)].reshape(1, -1)
        return X_next, ei

    def optimize(self, iterations=100):
        """ Optimizes the black-box function """
        for i in range(iterations):
            X_next, _ = self.acquisition()

            # Check if point already sampled
            already_sampled = False
            for point in self.gp.X:
                if np.abs(point[0] - X_next[0, 0]) < 1e-8:
                    already_sampled = True
                    break

            if already_sampled:
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        # Find optimal point
        if self.minimize:
            best_idx = np.argmin(self.gp.Y)
        else:
            best_idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[best_idx].flatten()
        Y_opt = self.gp.Y[best_idx].flatten()

        return X_opt, Y_opt
