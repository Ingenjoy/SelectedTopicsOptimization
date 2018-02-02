"""
Created on Tuesday 22 August 2017
Last update: Thursday 18 January 2018

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

def compute_quadratic(P, q, r, x):
    """
    Evaluates a quadratic function

    Inputs:
        - P, q, r: the terms of the nD quadratic system
        - x: point at which the quadratic function is evaluated

    returns
        - f(x)
    """
    return np.sum(x.T @ P @ x / 2 + x.T @ q + r)

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
    PDx = P @ Dx
    DxTPDx = np.sum(PDx * Dx)
    DxTq = np.sum(Dx * q)
    DxTPx = np.sum(PDx * x)
    t_optimal =  - (DxTPx + DxTq) / DxTPDx
    return t_optimal

def gradient_descent_quadratic(P, q, x0, epsilon=1e-4, trace=False):
    """
    Gradient descent for quadratic systems

    Inputs:
        - P, q: the terms of the nD quadratic system
        - x0: starting point
        - trace: (bool) count number of steps?

    Outputs:
        - xstar: the found minimum
        - n_steps: number of steps before algorithm terminates (if trace=True)
    """
    x = x0  # initial value
    n_steps = 0
    while True:
        Dx = - P @ x - q
        if np.linalg.norm(Dx) < epsilon:
            break
        t = quadratic_exact_line_search(P, q, Dx, x)
        x += t * Dx
        n_steps += 1
    if trace: return x, n_steps
    else: return x

def gradient_descent_quadratic_momentum(P, q, x0,
                                        beta=0.2, epsilon=1e-4,
                                        trace=False):
    """
    Gradient descent for quadratic systems with momentum

    Inputs:
        - P, q: the terms of the nD quadratic system
        - x0: starting point
        - beta: momentum parameter (default set to 0.2)
        - trace: (bool) count number of steps?

    Outputs:
        - xstar: the found minimum
        - n_steps: number of steps before algorithm terminates (if trace=True)
    """
    x = x0  # initial value
    n_steps = 0
    Dx = np.zeros_like(x)
    while True:
        gradient = P @ x + q
        Dx *= beta
        Dx -= gradient * (1 - beta)
        if np.linalg.norm(gradient) < epsilon:
            break
        t = quadratic_exact_line_search(P, q, Dx, x)
        x += t * Dx
        n_steps += 1
    if trace: return x, n_steps
    else: return x

if __name__ == '__main__':

    # assignment 1

    print('solution 1d stystem: xstar={}'.format(
            solve_1d_quadratic(8, 8, 2)
    ))

    # example system

    P = np.array([[4,1],[1,2]]) * 2
    q = np.array([[2], [1]])
    r = 12
    x0 = np.zeros((2, 1))
    #x0 = np.array([[1.0], [1.0]])

    def fun_quadr(x):
        return np.sum(x.T @ P @ x) / 2 + np.sum(q * x) + r

    # assignment 2

    xstar_exact = solve_nd_quadratic(P, q)
    print('analytic solution nd system: xstar_exact={}'.format(xstar_exact))

    # assignment 4
    xstar_gd, n_steps_gd= gradient_descent_quadratic(P, q,
                        x0.copy(), trace=True)
    print('solution using GD: xstar_exact={} ({} steps)'.format(
                                    xstar_exact, n_steps_gd))

    # assignment 5
    xstar_gdm, n_steps_gdm = gradient_descent_quadratic_momentum(P, q,
                        x0.copy(), beta=0.2, trace=True)
    print('solution using GDM: xstar_exact={} ({} steps)'.format(xstar_exact,
                            n_steps_gdm))


    # new problem

    P = np.array([[500, 2], [2, 1]])
    q = np.array([-40, 100]).reshape((-1,1))
    x0 = np.zeros((2, 1))

    print('new values:')
    xstar_gd, n_steps_gd= gradient_descent_quadratic(P, q,
                        x0.copy(), trace=True)
    print('solution using GD: xstar_exact={} ({} steps)'.format(
                                    xstar_exact, n_steps_gd))

    xstar_gdm, n_steps_gdm = gradient_descent_quadratic_momentum(P, q,
                        x0.copy(), beta=0.2, trace=True)
    print('solution using GDM: xstar_exact={} ({} steps)'.format(xstar_exact,
                            n_steps_gdm))
