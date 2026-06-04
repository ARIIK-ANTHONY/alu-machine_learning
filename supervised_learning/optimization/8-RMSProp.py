#!/usr/bin/env python3
""" Creates RMSProp optimization operation """

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates training operation using RMSProp optimization algorithm

    Args:
        loss: loss of the network
        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero

    Returns:
        RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon)
    return optimizer.minimize(loss)
