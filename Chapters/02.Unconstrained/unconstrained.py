# -*- coding: utf-8 -*-
"""
Created on Tue 19 Jan 2016
Last update: Wed 27 Dec 2017

@author: Michiel Stock
michielfmstock@gmail.com

Solutions algorithms Chapter 01: Unconstrained optimization
implements:
    - backtracking ls
    - gradient descent
    - Newton's method
"""

import numpy as np


def backtracking_line_search(f, x0, Dx, grad_f, alpha=0.05, beta=0.6):
    '''
    Uses backtracking for finding the minimum over a line.
    Inputs:
        - f: function to be searched over a line
        - x0: initial point
        - Dx: direction to search
        - grad_f: gradient of f
        - alpha
        - beta
    Output:
        - t: suggested stepsize
    '''
    t = 1
    while f(x0 + t * Dx) > f(x0) + alpha * t * np.sum(grad_f(x0) * Dx):
        t *= beta
    return t

def gradient_descent(f, x0, grad_f, alpha=0.05, beta=0.6, nu=1e-3, trace=False):
    '''
    General gradient descent algorithm.

    Inputs:
        - f: function to be minimized
        - x0: starting point
        - grad_f: gradient of the function to be minimized
        - alpha: parameter for btls
        - beta: parameter for btls
        - nu: parameter to determine if the algortihm is convered
        - trace: (bool) store the path that is followed?

    Outputs:
        - xstar: the found minimum
        - x_steps: path in the domain that is followed (if trace=True)
        - f_steps: image of x_steps (if trace=True)
    '''
    x = x0  # initial value
    if trace: x_steps = [x0.copy()]
    if trace: f_steps = [f(x0)]
    while True:
        Dx = - grad_f(x)  # choose direction
        if np.linalg.norm(grad_f(x)) <= nu:
            break  # converged
        t = backtracking_line_search(f, x, Dx, grad_f, alpha, beta)
        x += t * Dx
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(f(x))
    if trace: return x, x_steps, f_steps
    else: return x

def coordinate_descent(f, x0, grad_f, alpha=0.2, beta=0.7, nu=1e-3, trace=False):
    '''
    General coordinate descent algorithm.

    Inputs:
        - f: function to be minimized
        - x0: starting point
        - grad_f: gradient of the function to be minimized
        - alpha: parameter for btls
        - beta: parameter for btls
        - nu: parameter to determine if the algortihm is convered
        - trace: (bool) store the path that is followed?

    Outputs:
        - xstar: the found minimum
        - x_steps: path in the domain that is followed (if trace=True)
        - f_steps: image of x_steps (if trace=True)
    '''
    x = x0  # initial value
    n, _ = x.shape
    if trace: x_steps = [x0.copy()]
    if trace: f_steps = [f(x0)]
    while True:
        df = grad_f(x)  # choose direction
        i = np.abs(df).argmax()
        Dx = - df[i, 0] * np.eye(1,n,i).T
        if np.linalg.norm(grad_f(x)) <= nu:
            break  # converged
        t = backtracking_line_search(f, x, Dx, grad_f, alpha, beta)
        x += t * Dx
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(f(x))
    if trace: return x, x_steps, f_steps
    else: return x

def newtons_method(f, x0, grad_f, hess_f, alpha=0.3, beta=0.8, epsilon=1e-3,
                   trace=False):
    '''
    Newton's method for minimizing functions.

    Inputs:
        - f: function to be minimized
        - x0: starting point
        - grad_f: gradient of the function to be minimized
        - hess_f: hessian matrix of the function to be minimized
        - alpha: parameter for btls
        - beta: parameter for btls
        - nu: parameter to determine if the algortihm is convered
        - trace: (bool) store the path that is followed?

    Outputs:
        - xstar: the found minimum
        - x_steps: path in the domain that is followed (if trace=True)
        - f_steps: image of x_steps (if trace=True)
    '''
    x = x0  # initial value
    if trace: x_steps = [x.copy()]
    if trace: f_steps = [f(x0)]
    while True:
        Dx = - np.linalg.solve(hess_f(x), grad_f(x))
        if - grad_f(x).T.dot(Dx) / 2 <= epsilon:   # stopping criterion
            break  # converged
        t = backtracking_line_search(f, x, Dx, grad_f, alpha, beta)
        x += Dx * t
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(f(x))
    if trace: return x, x_steps, f_steps
    else: return x


if __name__ == '__main__':
    from sys import path
    import matplotlib.pyplot as plt
    from teachingtools import *
    path.append('../../Scripts')
    from plotting import plot_contour, add_path, blue, red, green, yellow

    # assignment 1
    # ------------

    function = lambda x : x**2 - 2*x - 5
    gradient_function = lambda x : 2*x -2
    Dx = 5

    tbest = backtracking_line_search(function, 0, 5, gradient_function)

    x = np.linspace(-1, 2)
    fig, ax = plt.subplots()
    ax.plot(x, function(x), blue, label='$x^3-2*x-5$')
    ax.scatter(0, function(0), color=green, label='$x_0$')
    ax.vlines(0+tbest*Dx, -7, -1, red)
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig('Figures/backtracking_ls.png')

    # assignment 2
    # ------------

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.set_title('Quadratic')
    ax2.set_title('Non-quadratic')
    plot_contour(quadratic, (-11, 11), (-5, 5), ax1)
    plot_contour(nonquadratic, (-2, 2), (-1, 1), ax2)

    xstar_q, x_steps_q, f_steps_q = gradient_descent(quadratic,\
                                                    np.array([[10.0], [1.0]]),
                                        grad_quadratic, nu=1e-5, trace=True)
    add_path(ax1, x_steps_q, red, label='GD')

    print('Number of steps quadratic function (gradient descent): {}'.format(\
                                                    len(x_steps_q) - 1))

    xstar_nq, x_steps_nq, f_steps_nq = gradient_descent(nonquadratic,
                                            np.array([[-0.5], [0.9]]),
                        grad_nonquadratic, nu=1e-5, trace=True)
    add_path(ax2, x_steps_nq, red, label='GD')

    print('Number of steps non-quadratic function (gradient descent): {}'.format(len(
                    f_steps_nq) - 1))

    fig.tight_layout()
    fig.savefig('Figures/gradient_descent.png')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.set_title('Quadratic')
    ax2.set_title('Non-quadratic')
    ax1.plot(np.abs(f_steps_q), color=blue)
    ax1.semilogy()
    ax1.set_title('Convergence ')
    ax2.plot(np.abs(f_steps_nq[:-1] - f_steps_nq[-1]), color=blue)  # error compared to last step
    ax2.semilogy()

    for ax in (ax1, ax2):
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Absolute error')

    fig.tight_layout()
    fig.savefig('Figures/convergence_gd.png')

    # assignment 3
    # ------------

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.set_title('Quadratic')
    ax2.set_title('Non-quadratic')
    plot_contour(quadratic, (-11, 11), (-5, 5), ax1)
    plot_contour(nonquadratic, (-2, 2), (-1, 1), ax2)

    xstar_q, x_steps_q, f_steps_q = coordinate_descent(quadratic, np.array([[10.1], [1.0]]),
                                                     grad_quadratic, nu=1e-5, trace=True)
    add_path(ax1, x_steps_q, red)

    print('Number of steps quadratic function (steepest descent): {}'.format(len(x_steps_q) - 1))

    xstar_nq, x_steps_nq, f_steps_nq = coordinate_descent(nonquadratic, np.array([[-0.5], [0.9]]),
                                                        grad_nonquadratic, nu=1e-5, trace=True)
    add_path(ax2, x_steps_nq, red)

    print('Number of steps non-quadratic function (steepest descent): {}'.format(len(f_steps_nq) - 1))
    fig.tight_layout()
    fig.savefig('Figures/steepest_descent.png')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.set_title('Quadratic')
    ax2.set_title('Non-quadratic')
    ax1.plot(np.abs(f_steps_q), color=blue)
    ax1.semilogy()
    ax2.plot(np.abs(f_steps_nq[:-1] - f_steps_nq[-1]), color=blue)  # error compared to last step
    ax2.semilogy()

    for ax in (ax1, ax2):
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Absolute error')

    fig.tight_layout()
    fig.savefig('Figures/convergence_sd.png')

    # assignment 4
    # ------------

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.set_title('Quadratic')
    ax2.set_title('Non-quadratic')
    plot_contour(quadratic, (-11, 11), (-5, 5), ax1)
    plot_contour(nonquadratic, (-2, 2), (-1, 1), ax2)

    xstar_q, x_steps_q, f_steps_q = newtons_method(quadratic, np.array([[10.0],
                            [1.0]]), grad_quadratic, hessian_quadratic,
                            epsilon=1e-8, trace=True)
    add_path(ax1, x_steps_q, red, label='Newton')

    print("Number of steps quadratic function (Newton's method): {}".format(len(x_steps_q) - 1))

    xstar_nq, x_steps_nq, f_steps_nq = newtons_method(nonquadratic,
                                np.array([[-0.5], [0.9]]), grad_nonquadratic,
                                hessian_nonquadratic, epsilon=1e-5, trace=True)
    add_path(ax2, x_steps_nq, red, label='Newton')

    print("Number of steps non-quadratic function (Newton's method): {}".format(len(
                                x_steps_nq) - 1))
    fig.tight_layout()
    fig.savefig('Figures/newtons_method.png')

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.set_title('Quadratic')
    ax2.set_title('Non-quadratic')
    ax1.plot(f_steps_q, color=blue)
    ax1.semilogy()
    ax2.plot(f_steps_nq, color=blue)

    for ax in (ax1, ax2):
        ax.set_xlabel('iteration')
        ax.set_ylabel('function value')

    fig.tight_layout()
    fig.savefig('Figures/convergence_nm.png')
