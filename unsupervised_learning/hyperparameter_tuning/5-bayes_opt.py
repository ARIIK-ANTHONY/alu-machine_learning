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
                 sigma_f=1, ac_func='EI'):
        """ Initializes Bayesian Optimization """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.ac_func = ac_func
        self.bounds = bounds

    def acquisition(self):
        """ Calculates the acquisition function values """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)
        sigma = np.where(sigma < 1e-9, 1e-9, sigma)

        if self.ac_func == 'EI':
            # Expected Improvement
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu
            Z = imp / sigma
            ei = imp * norm_cdf(Z) + sigma * norm_pdf(Z)
            return ei.flatten()
        elif self.ac_func == 'PI':
            # Probability of Improvement
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu
            Z = imp / sigma
            pi = norm_cdf(Z)
            return pi.flatten()
        elif self.ac_func == 'UCB':
            # Upper Confidence Bound
            kappa = 2.0
            ucb = mu + kappa * sigma
            return ucb.flatten()
        return np.zeros(len(self.X_s))

    def optimize(self, iterations=100):
        """ Optimizes the black-box function """
        # Run optimization for specified iterations
        for _ in range(iterations):
            # Get next point to sample
            ac_vals = self.acquisition()
            idx = np.argmax(ac_vals)
            X_next = self.X_s[idx].reshape(1, -1)

            # Check for duplicate
            duplicate = False
            for point in self.gp.X:
                if np.allclose(point, X_next[0], atol=1e-8):
                    duplicate = True
                    break

            if duplicate:
                break

            # Evaluate function and update GP
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        # Find optimal values
        best_idx = np.argmin(self.gp.Y)
        X_opt = self.gp.X[best_idx].flatten()
        Y_opt = self.gp.Y[best_idx].flatten()

        return X_opt, Y_opt
