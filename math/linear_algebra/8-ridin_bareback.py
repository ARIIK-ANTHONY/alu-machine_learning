#!/usr/bin/env python3
"""Module for matrix multiplication."""


def mat_mul(mat1, mat2):
    """Multiply two matrices.

    Args:
        mat1: First matrix
        mat2: Second matrix

    Returns:
        New matrix product, or None if multiplication is not possible
    """
    # Check if matrices can be multiplied
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize result matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
