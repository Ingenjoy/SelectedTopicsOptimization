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
    x = np.array(x).reshape((-1, 1))
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

def gradient_descent_quadratic(P, q, r, x0, nu=1e-5, trace=False):
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


def gradient_descent_quadratic_momentum(P, q, r, x0, alpha=0.1, beta=0.8, nu=1e-5,
                                trace=False):
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
        Dx = beta * Dx - alpha * grad  # choose direction
        if np.linalg.norm(grad) <= nu:
            break  # converged
        t = solve_1d_quadratic(np.sum(Dx.T @ P @ Dx),
                    + np.sum(Dx * grad))
        x += t * Dx
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(evaluate_quadratic(x, P, q, r))
    if trace: return x, x_steps, f_steps
    else: return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sys import path
    path.append('../../Scripts')
    from plotting import plot_contour, add_path, blue, red

    # make quadratic problem
    P = np.diag([1, 10])
    q = np.zeros((2, 1))
    r = 0
    x0 = np.array([[10.0],[1.0]])

    xstar = solve_nd_quadratic(P, q)  # origin

    xstar_gd, x_steps_gd, f_steps_gd = gradient_descent_quadratic(P, q, r,
                        x0.copy(), trace=True)
    xstar_gdm, x_steps_gdm, f_steps_gdm = gradient_descent_quadratic_momentum(P, q,
                        r, x0.copy(), alpha=0.05, beta=0.5, trace=True)

    # convergence gradient descent
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
    ax0.plot(f_steps_gd, color=blue, label='GD', lw=2)
    ax0.semilogy()
    ax0.legend(loc=0)
    ax0.set_xlabel(r'$k$')
    ax0.set_ylabel(r'$f(\mathbf{x}^{(k)})-f(\mathbf{x}^\star)$')

    plot_contour(lambda x : evaluate_quadratic(x, P, q, r),
                        (-11, 11), (-5, 5), ax1, plot_f=True)
    add_path(ax1, x_steps=x_steps_gd, col=blue, label='GD')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.legend(loc=0)
    fig.savefig('Figures/convergence_gd.png')

    # convergence gradient descent and momentum
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
    ax0.plot(f_steps_gd, color=blue, label='GD', lw=2)
    ax0.plot(f_steps_gdm, color=red, label='GDM', lw=2)
    ax0.semilogy()
    ax0.legend(loc=0)
    ax0.set_xlabel(r'$k$')
    ax0.set_ylabel(r'$f(\mathbf{x}^{(k)})-f(\mathbf{x}^\star)$')

    plot_contour(lambda x : evaluate_quadratic(x, P, q, r),
                        (-11, 11), (-5, 5), ax1, plot_f=True)
    add_path(ax1, x_steps=x_steps_gd, col=blue, label='GD')
    add_path(ax1, x_steps=x_steps_gdm, col=red, label='GDM')
    ax1.legend(loc=0)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    fig.savefig('Figures/convergence_gd_momentum.png')
