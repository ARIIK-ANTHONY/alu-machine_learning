#!/usr/bin/env python3
"""Module for calculating cofactor matrix."""


def determinant(matrix):
    """Calculate determinant for internal use."""
    # Handle empty matrix case
    if matrix == [[]]:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        a, b = matrix[0][0], matrix[0][1]
        c, d = matrix[1][0], matrix[1][1]
        return a * d - b * c

    det = 0
    n = len(matrix)
    for col in range(n):
        submatrix = []
        for i in range(1, n):
            row = []
            for j in range(n):
                if j != col:
                    row.append(matrix[i][j])
            submatrix.append(row)
        sign = (-1) ** col
        cofactor = sign * matrix[0][col] * determinant(submatrix)
        det += cofactor
    return det


def cofactor(matrix):
    """Calculate the cofactor matrix of a matrix.

    Args:
        matrix: List of lists representing a matrix

    Returns:
        Cofactor matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is non-empty and square
    if not matrix or len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # Calculate cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            # Create submatrix by removing row i and column j
            submatrix = []
            for row_idx in range(n):
                if row_idx != i:
                    subrow = []
                    for col_idx in range(n):
                        if col_idx != j:
                            subrow.append(matrix[row_idx][col_idx])
                    submatrix.append(subrow)

            # Special case: if submatrix is empty (1x1 case handled above)
            if not submatrix:
                submatrix = [[]]

            # Calculate cofactor: (-1)^(i+j) * determinant(submatrix)
            sign = 1 if (i + j) % 2 == 0 else -1
            cofactor_row.append(sign * determinant(submatrix))
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
