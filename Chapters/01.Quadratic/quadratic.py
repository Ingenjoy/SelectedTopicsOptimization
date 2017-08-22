"""
Created on Tuesday 22 August 2017
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Functions for quadratic optimization
"""

import numpy as np

def solve_1d_quadratic(p, q, r=0):
    """
    Finds the minimizer of an 1D quadratic system, raises an error if there is
    no minimizer (p<0)

    Inputs:
        - p, q, r: the coefficients of the 1D quadratic system

    Output:
        - xstar: the minimizer
    """
    assert p > 0
    return - q / p

def solve_nd_quadratic(P, q, r=0):
    """
    Finds the minimizer of an 1D quadratic system, raises an error if there is
    no minimizer (P is not positive-definite)

    Inputs:
        - Q, q, r: the terms of the nD quadratic system

    Output:
        - xstar: the minimizer, an (n x 1) vector
    """
    assert np.all(np.linalg.eigvalsh(P) > 0)
    return - np.linalg.solve(P, q)
