#!/usr/bin/env python3
"""Module for matrix concatenation."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two 2D matrices along a specific axis.

    Args:
        mat1: First matrix
        mat2: Second matrix
        axis: Axis to concatenate along (0 for rows, 1 for columns)

    Returns:
        New concatenated matrix, or None if concatenation is not possible
    """
    # Check dimensions
    if axis == 0:  # Concatenate rows (vertically)
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Create deep copies to avoid mutations
        result = [row[:] for row in mat1]
        result.extend([row[:] for row in mat2])
        return result
    elif axis == 1:  # Concatenate columns (horizontally)
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            # Create new row by concatenating columns
            new_row = mat1[i][:] + mat2[i][:]
            result.append(new_row)
        return result
    return None
