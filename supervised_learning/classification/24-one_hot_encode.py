#!/usr/bin/env python3
""" Neural Network
"""

import numpy as np


class NeuralNetwork:
    """ Class that defines a neural network with one hidden layer performing
        binary classification.
    """

    def __init__(self, nx, nodes):
        """ Instantiation function

        Args:
            nx (int): size of the input layer
            nodes (int): size of the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Return weights vector for hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Return bias for hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Return activated output for hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Return weights vector for output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Return bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Return activated output for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network

        Args:
            X (numpy.array): Input data with shape (nx, m)

        Returns:
            A1 (numpy.array): Activated output of hidden layer
            A2 (numpy.array): Activated output of output layer
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.array): Actual values with shape (1, m)
            A (numpy.array): Predicted values with shape (1, m)

        Returns:
            cost (float): Cross-entropy cost
        """
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions

        Args:
            X (numpy.array): Input data with shape (nx, m)
            Y (numpy.array): Actual values with shape (1, m)

        Returns:
            prediction (numpy.array): Predicted values (1, m)
            cost (float): Cross-entropy cost
        """
        _, A2 = self.forward_prop(X)
        prediction = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network

        Args:
            X (numpy.array): Input data with shape (nx, m)
            Y (numpy.array): Actual values with shape (1, m)
            A1 (numpy.array): Activated output of hidden layer
            A2 (numpy.array): Activated output of output layer
            alpha (float, optional): Learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]

        # Output layer gradients
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        # Hidden layer gradients
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        # Update parameters
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neural network

        Args:
            X (numpy.array): Input data with shape (nx, m)
            Y (numpy.array): Actual values with shape (1, m)
            iterations (int, optional): Number of iterations. Defaults to 5000.
            alpha (float, optional): Learning rate. Defaults to 0.05.

        Returns:
            prediction (numpy.array): Predicted values after training
            cost (float): Final cost
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        # Only one loop allowed - this is the only loop in the entire class
        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
