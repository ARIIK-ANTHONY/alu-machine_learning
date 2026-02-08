#!/usr/bin/env python3
"""Module for numpy element-wise operations."""
import numpy as np


def np_elementwise(mat1, mat2):
    """Perform element-wise operations on two numpy arrays.

    Args:
        mat1: First numpy array
        mat2: Second numpy array

    Returns:
        Tuple containing (sum, difference, product, quotient)
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return add, sub, mul, div
