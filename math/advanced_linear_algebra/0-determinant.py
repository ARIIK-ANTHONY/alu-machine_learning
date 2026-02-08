#!/usr/bin/env python3
"""Module for calculating matrix determinant."""


def determinant(matrix):
    """Calculate the determinant of a matrix.

    Args:
        matrix: List of lists representing a matrix

    Returns:
        Determinant of the matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")

    # Handle empty matrix case
    if matrix == [[]]:
        return 1

    # Check if matrix is square
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a square matrix")
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        a, b = matrix[0][0], matrix[0][1]
        c, d = matrix[1][0], matrix[1][1]
        return a * d - b * c

    # Recursive case: n x n matrix
    det = 0
    for col in range(n):
        # Create submatrix by removing first row and current column
        submatrix = []
        for i in range(1, n):
            row = []
            for j in range(n):
                if j != col:
                    row.append(matrix[i][j])
            submatrix.append(row)

        # Calculate cofactor and add to determinant
        sign = (-1) ** col
        cofactor = sign * matrix[0][col] * determinant(submatrix)
        det += cofactor

    return det
