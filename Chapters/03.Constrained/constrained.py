# -*- coding: utf-8 -*-
"""
Created on Tue 19 Jan 2016
Last update: Tue 07 Mar 2017

@author: Michiel Stock
michielfmstock@gmail.com

Solutions algorithms Chapter 01: Unconstrained optimization
implements:
    - backtracking ls
    - gradient descent
    - Newton's method
"""

import numpy as np

def solve_constrained_quadratic_problem(P, q, A, b):
    """
    Solve a linear constrained quadratic convex problem.
    Inputs:
        - P, q: quadratic and linear parameters of
                the linear function to be minimized
        - A, b: system of the linear constaints
    Outputs:
        - xstar: the exact minimizer
        - vstar: the optimal Lagrance multipliers
    """
    p, n = A.shape  # size of the problem
    solution = np.linalg.solve(np.bmat([[P, A.T],
                [A, np.zeros((p, p))]]), np.bmat([[-q], [b]]))
    xstar = solution[:n]
    vstar = solution[n:]
    return np.array(xstar), np.array(vstar)

def linear_constrained_newton(f, x0, grad_f, hess_f, A, b, stepsize=0.25, epsilon=1e-3, trace=False):
    '''
    Newton's method for minimizing functions with linear constraints.
    Inputs:
        - f: function to be minimized
        - x0: starting point (does not have to be feasible)
        - grad_f: gradient of the function to be minimized
        - hess_f: hessian matrix of the function to be minimized
        - A, b: linear constraints
        - stepsize: stepsize for each Newton step (fixed)
        - epsilon: parameter to determine if the algortihm is converged
        - trace: (bool) store the path that is followed?
    Outputs:
        - xstar: the found minimum
        - x_steps: path in the domain that is followed (if trace=True)
        - f_steps: image of x_steps (if trace=True)
    '''
    assert stepsize < 1 and stepsize > 0
    x = x0  # initial value
    p, n = A.shape
    if trace: x_steps = [x.copy()]
    if trace: f_steps = [f(x0)]
    while True:
        ddfx = hess_f(x)
        dfx = grad_f(x)
        r = b - A.dot(x)
        Dx, _ = solve_constrained_quadratic_problem(ddfx, dfx, A, r)
        newton_decrement = - np.sum(Dx * dfx) / 2
        if newton_decrement < epsilon:  # stopping criterion
            break  # converged
        x += stepsize * Dx
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(f(x))
    if trace: return x, x_steps, f_steps
    else: return x


if __name__ == '__main__':
    pass
