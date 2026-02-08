#!/usr/bin/env python3
"""Module for matrix shape calculation."""

def matrix_shape(matrix):
    """Calculate the shape of a matrix.
    
    Args:
        matrix: The matrix to analyze
    
    Returns:
        List of integers representing the shape
    """
    shape = []
    current = matrix
    
    while isinstance(current, list) and len(current) > 0:
        shape.append(len(current))
        current = current[0]
    
    return shape
