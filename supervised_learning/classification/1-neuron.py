#!/usr/bin/env python3

import numpy as np

class Neuron:
    """Neuron class for binary classification."""
    
    def __init__(self, nx):
        """Constructor
        
        Args:
            nx (int): number of input features
            
        Attributes (private):
            __W (numpy.ndarray): weights vector of shape (1, nx)
            __b (float): bias
            __A (float): activated output
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Initialize private attributes
        self.__W = np.random.randn(1, nx)  # Shape (1, nx)
        self.__b = 0
        self.__A = 0
    
    @property
    def W(self):
        """Getter for weights"""
        return self.__W
    
    @property
    def b(self):
        """Getter for bias"""
        return self.__b
    
    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

import numpy as np

Neuron = __import__('1-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)
