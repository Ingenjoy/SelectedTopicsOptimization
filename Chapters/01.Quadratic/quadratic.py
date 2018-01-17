"""
Created on Tuesday 22 August 2017
Last update: Thursday 9 November 2017

@author: Michiel Stock
michielfmstock@gmail.com

Functions for quadratic optimization
"""

import numpy as np

def solve_1d_quadratic(p, q, r=0):
    """
    Finds the minimizer of an 1-D quadratic system, raises an error if there is
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
    Finds the minimizer of an N-D quadratic system,
    raises an error if there is no minimizer
    (P is not positive-definite)

    Inputs:
        - P, q, r: the terms of the nD quadratic system

    Output:
        - xstar: the minimizer, an (n x 1) vector
    """
    assert np.all(np.linalg.eigvalsh(P) > 0)
    return - np.linalg.solve(P, q)

def quadratic_exact_line_search(P, q, Dx, x):
    """
    Find the exact step size that minimized a quadratic system in
    a given point x for a given search direction Dx

    Inputs:
        - P, q: the terms of the nD quadratic system
        - x: starting point
        - Dx: search direction

    Output:
        - t: optimal step size
    """
    DxTPx = np.sum(Dx.T @ P @ x)
    DxTq = np.sum(Dx * q)
    return - (DxTPx + DxTq) / DxTPx

def gradient_descent_quadratic(P, q, x0, epsilon=1e-3, trace=False):
    """
    Gradient descent for quadratic systems

    Inputs:
        - P, q: the terms of the nD quadratic system
        - x0: starting point
        - trace: (bool) store the path that is followed?

    Outputs:
        - xstar: the found minimum
        - x_steps: path in the domain that is followed (if trace=True)
        - f_steps: image of x_steps (if trace=True)
    """
    x = x0  # initial value
    if trace: x_steps = [x0.copy()]
    if trace: f_steps = [f(x0)]
    while True:
        Dx = - P @ q
        if np.linalg.norm(Dx) < epsilon:
            break
        t = quadratic_exact_line_search(P, q, Dx, x)
        x += t * Dx
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(f(x))
    if trace: return x, x_steps, f_steps
    else: return x

#TODO: implement momentum updates
def gradient_descent_quadratic_momentum(P, q, x0, epsilon=1e-3, trace=False):
    """
    Gradient descent for quadratic systems with momentum

    Inputs:
        - P, q: the terms of the nD quadratic system
        - x0: starting point
        - trace: (bool) store the path that is followed?

    Outputs:
        - xstar: the found minimum
        - x_steps: path in the domain that is followed (if trace=True)
        - f_steps: image of x_steps (if trace=True)
    """
    x = x0  # initial value
    if trace: x_steps = [x0.copy()]
    if trace: f_steps = [f(x0)]
    while True:
        Dx = - P @ q
        if np.linalg.norm(Dx) < epsilon:
            break
        t = quadratic_exact_line_search(P, q, Dx, x)
        x += t * Dx
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(f(x))
    if trace: return x, x_steps, f_steps
    else: return x
