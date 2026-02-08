#!/usr/bin/env python3
"""Module for matrix transpose operation."""


def matrix_transpose(matrix):
    """Transpose a 2D matrix.

    Args:
        matrix: The matrix to transpose

    Returns:
        New transposed matrix
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # Create empty transposed matrix
    transpose = [[0 for _ in range(rows)] for _ in range(cols)]

    # Fill the transposed matrix
    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = matrix[i][j]

    return transpose
