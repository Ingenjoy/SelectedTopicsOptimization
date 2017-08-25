"""
Created on Tuesday 22 August 2017
Last update: Friday 25 August 2017

@author: Michiel Stock
michielfmstock@gmail.com

Functions for quadratic optimization
"""

import numpy as np

def evaluate_quadratic(x, P, q, r=0):
    """
    Evaluate a quadratic system

    Inputs:
        - point to evaluate
        - p, q, r: the coefficients of the n-D quadratic system

    Output:
        - fx: the function evalutation
    """
    return np.sum(0.5 * x.T @ P @ x + q.T @ x + r)  # sum to output a scalar

def solve_1d_quadratic(p, q, r=0):
    """
    Finds the minimizer of an 1D quadratic system, raises an error if there is
    no minimizer (p<0)

    Inputs:
        - p, q, r: the coefficients of the 1-D quadratic system

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
        - Q, q, r: the terms of the n-D quadratic system

    Output:
        - xstar: the minimizer, an (n x 1) vector
    """
    assert np.all(np.linalg.eigvalsh(P) > 0)
    return - np.linalg.solve(P, q)

def gradient_descent_quadratic(P, q, r, x0, nu=1e-3, trace=False):
    """
    Gradient descent algorithms for quadratic systems, uses exact line search

    Inputs:
        - Q, q, r: the terms of the n-D quadratic system
        - x0: starting point (n x 1 vector)
        - nu: parameter to determine if the algortihm is convered
        - trace: (bool) store the path that is followed?

    Outputs:
        - xstar: the found minimum
        - x_steps: path in the domain that is followed (if trace=True)
        - f_steps: image of x_steps (if trace=True)
    """
    x = x0  # initial value
    if trace: x_steps = [x0.copy()]
    if trace: f_steps = [evaluate_quadratic(x, P, q, r)]
    while True:
        grad = P @ x + q
        Dx = - grad  # choose direction
        if np.linalg.norm(Dx) <= nu:
            break  # converged
        t = solve_1d_quadratic(np.sum(Dx.T @ P @ Dx),
                    np.sum(Dx * grad))
        x += t * Dx
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(evaluate_quadratic(x, P, q, r))
    if trace: return x, x_steps, f_steps
    else: return x


def gradient_descent_quadratic_momentum(P, q, r, x0, beta=0.8, nu=1e-3, trace=False):
    """
    Gradient descent algorithms for quadratic systems, uses exact line search
    and momentum

    Inputs:
        - Q, q, r: the terms of the n-D quadratic system
        - momentum parameter
        - x0: starting point (n x 1 vector)
        - nu: parameter to determine if the algortihm is convered
        - trace: (bool) store the path that is followed?

    Outputs:
        - xstar: the found minimum
        - x_steps: path in the domain that is followed (if trace=True)
        - f_steps: image of x_steps (if trace=True)
    """
    x = x0  # initial value
    Dx = np.zeros_like(x0)
    if trace: x_steps = [x0.copy()]
    if trace: f_steps = [evaluate_quadratic(x, P, q, r)]
    while True:
        grad = P @ x + q
        Dx = beta * Dx - grad  # choose direction
        if np.linalg.norm(grad) <= nu:
            break  # converged
        t = solve_1d_quadratic(np.sum(Dx.T @ P @ Dx),
                    + np.sum(Dx * grad))
        print(t)
        x += t * Dx
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(evaluate_quadratic(x, P, q, r))
    if trace: return x, x_steps, f_steps
    else: return x


if __name__ == '__main__':
    main()
