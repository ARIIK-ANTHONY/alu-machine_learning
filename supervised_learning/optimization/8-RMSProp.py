#!/usr/bin/env python3
""" RMSProp optimization for neural network training """

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates RMSProp optimization operation for a neural network

    RMSProp (Root Mean Square Propagation) is an adaptive learning rate
    optimization algorithm that maintains a moving average of squared
    gradients to normalize updates.

    Args:
        loss: the loss of the network
        alpha: the learning rate
        beta2: the RMSProp weight (decay rate)
        epsilon: small number to avoid division by zero

    Returns:
        the RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon)
    return optimizer.minimize(loss)
