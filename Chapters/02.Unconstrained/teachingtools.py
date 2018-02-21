# -*- coding: utf-8 -*-
"""
Created on Tue 19 Jan 2016
Last update: Tue 14 Feb 2016

@author: Michiel Stock
michielfmstock@gmail.com

Some functions for Chapter 01: Unconstrained optimization
"""

import numpy as np
import sympy as sp
#from unconstrained import gradient_descent, newtons_method, coordinate_descent
import matplotlib.pyplot as plt
from sys import path
#path.append('../../helpers/')
#from plotting import plot_contour, add_path, blue, red, green, yellow, orange

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

# VISUALIZATION
# -------------


def plot_contour(f, xlim, ylim, ax, plot_f=True):
    '''
    Plots the contour of a 2D function to be minimized
    '''
    xvals = np.linspace(*xlim)
    yvals = np.linspace(*ylim)
    X, Y = np.meshgrid(xvals, yvals)
    Z = np.reshape(list(map(f, zip(X.ravel().tolist(), Y.ravel().tolist()))),
                   X.shape)
    ax.contour(X, Y, Z)
    if plot_f: ax.contourf(X, Y, Z, cmap='bone')


def add_path(ax, x_steps, col='b', label=''):
    '''
    Adds a path of an opitmization algorithm to a figure
    '''
    ax.plot([x[0] for x in x_steps], [x[1] for x in x_steps], c=col, label=label)

def show_condition(gamma, theta):
    quad_gamma = lambda x : quadratic(x, gamma, theta)
    x0 = np.array([[10.], [1.]])
    d_quad_gamma = lambda x : grad_quadratic(x, gamma, theta)
    dd_quad_gamma = lambda x : hessian_quadratic(x, gamma, theta)
    xstar_gd, x_steps_gd, f_steps_gd = gradient_descent(quad_gamma,
                                                    x0.copy(),
                                                     d_quad_gamma,
                                                     nu=1e-6, trace=True)
    xstar_cd, x_steps_cd, f_steps_cd = coordinate_descent(quad_gamma,
                                                    x0.copy(),
                                                     d_quad_gamma,
                                                     nu=1e-6, trace=True)
    xstar_nm, x_steps_nm, f_steps_nm = newtons_method(quad_gamma, x0.copy(),
                                                   d_quad_gamma, dd_quad_gamma, epsilon=1e-6, trace=True)
    fig, ax1 = plt.subplots(ncols=1, figsize=(10, 10))
    plot_contour(quad_gamma, [-10, 10], [-11, 11], ax1)
    add_path(ax1, x_steps_gd, 'b', label='GD')
    add_path(ax1, x_steps_cd, 'r', label='CD')
    add_path(ax1, x_steps_nm, 'g', label='NM')
    ax1.legend(loc=3)
    print('Gradient descent iterations: {}'.format(len(x_steps_gd) - 1 ))
    print('Coordinate descent iterations: {}'.format(len(x_steps_cd) - 1 ))
    print('Newton\'s iterations: {}'.format(len(x_steps_nm) - 1))

def make_general_multidim_problem(n, m):
    """
    Generates the function, gradient and Hessian of a problem
    of the form

    f(x) = x^T C x - sum_i log(b_i - a_i^T x)

    where n is the dimension of the problem and m the rank

    the parameters a and b are generated randomly
    """
    a = np.random.randn(n, m) * np.random.binomial(1, 0.25, (n, m))
    b = np.random.rand(m, 1) * 10
    C = np.diag(np.random.rand(n)) * 10

    fun = lambda x : np.sum(np.array(x).T.dot(C).dot( x) - np.log(np.maximum(b - a.T.dot( x), 0) + 1e-40).sum())
    grad_fun = lambda x : 2 * C.dot(x) + np.sum(a / (b - a.T.dot(x)).T, axis=1, keepdims=True)
    hessian_fun = lambda x : 2 * C + (a / ((b - a.T.dot(x))**2).T).dot(a.T)
    return fun, grad_fun, hessian_fun

# EXAMPLES
# --------

# example of inexact line search

def show_inexact_ls(alpha=0.4, beta=0.9, Dx=10):
    """
    Demonstration of the hyperparameters of inexact line search
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    f = lambda x : x**2 - 10*x - 5
    df = lambda x : 2*x - 10
    t_begin = -0.4
    t_end = 1.5
    tvals = np.linspace(t_begin, t_end)
    ax.plot(tvals, f(tvals * Dx), color=blue,
                label=r'$f(\mathbf{x}+t\Delta \mathbf{x})$')
    ax.plot(tvals, f(0) + tvals * df(0) * Dx, color=red,
                label=r'$f(\mathbf{x})+t\nabla f(\mathbf{x})^\top \Delta \mathbf{x}$')
    ax.plot(tvals, f(0) + tvals * df(0) * alpha * Dx, color=red, ls='--',
                label=r'$f(\mathbf{x})+t\alpha\nabla f(\mathbf{x})^\top \Delta \mathbf{x}$')
    t = 1
    n_steps = 0
    while f(t * Dx) > f(0) + alpha * t * df(0) * Dx:
        ax.plot([t, t], [f(t * Dx), f(0) + alpha * t * df(0) * Dx], color=green,
                                    lw=2)
        f_old = f(t * Dx)
        t *= beta
        ax.plot([t, t / beta], [f(0) + alpha * t * df(0) * Dx, f_old],
                                    color=green, lw=2)
        n_steps += 1
    print('Converged after {} steps'.format(n_steps))
    ax.set_xlabel('$t$')
    ax.grid()
    ax.legend(loc=3)
    return fig


# defining the quadric function, gradient and hessian

from numpy import cos, sin

def get_transformation_matrix(gamma, theta):
    rot = np.array([[cos(theta), sin(theta)],
            [sin(theta), cos(theta)]])
    scaling = np.diag([1, gamma])
    return rot.T.dot(scaling).dot(rot)

def quadratic(x, gamma=10, theta=0.0):
    x = np.array(x).reshape((-1, 1))
    C = get_transformation_matrix(gamma, theta)
    return np.sum(0.5 * x.T.dot(C).dot(x))
    #return 0.5 * (x[0]**2 + gamma * x[1]**2)

def grad_quadratic(x, gamma=10, theta=0.0):
    x = np.array(x).reshape((-1, 1))
    C = get_transformation_matrix(gamma, theta)
    return C.dot(x)
    #return np.array([x[0], gamma * x[1]])

def hessian_quadratic(x, gamma=10, theta=0.0):
    C = get_transformation_matrix(gamma, theta)
    return C
    #return np.array([[1, 0], [0, gamma]])


# defining the non-quadric function, gradient and hessian


x1_, x2_ = sp.symbols('x1, x2')

nonquad_expr = sp.log(sp.exp(x1_ + 3 * x2_ - 0.1) + sp.exp(x1_ - 3 * x2_ - 0.1) +
                    sp.exp(-x1_ - 0.1))
#nonquad_expr /= 10
nonquadratic_f = sp.lambdify((x1_, x2_), nonquad_expr, np)
nonquadratic = lambda x : nonquadratic_f(x[0],x[1])
grad_nonquadratic_f = sp.lambdify((x1_, x2_), [nonquad_expr.diff(x1_),
                                  nonquad_expr.diff(x2_)], np)
grad_nonquadratic = lambda x : np.array(grad_nonquadratic_f(x[0], x[1]))

nqdx1dx1 = sp.lambdify((x1_, x2_), nonquad_expr.diff(x1_).diff(x1_), np)
nqdx1dx2 = sp.lambdify((x1_, x2_), nonquad_expr.diff(x1_).diff(x2_), np)
nqdx2dx2 = sp.lambdify((x1_, x2_), nonquad_expr.diff(x2_).diff(x2_), np)

def hessian_nonquadratic(x):
    return np.array([[nqdx1dx1(x[0,:], x[1,:]), nqdx1dx2(x[0,:], x[1,:])],
                     [nqdx1dx2(x[0], x[1]), nqdx2dx2(x[0], x[1])]]).reshape(2,2)

# example of of directions for steepest descent

def plot_vector(x, dx, ax, col, label=None):
    """
    Add a vector to a plot
    """
    ax.arrow(x[0], x[1], dx[0], dx[1], fc=col, ec=col,
            head_width=0.1)

def get_unit_circle(order=2, P=None, scale=0.5):
    """
    Get a matrix with 500 xy coordinates for the unit circle around x
    """
    X = np.zeros((500, 2))
    X[:,0] = np.cos(np.linspace(0, 2*np.pi, 500))
    X[:,1] = np.sin(np.linspace(0, 2*np.pi, 500))
    X /= np.linalg.norm(X, axis=1, ord=order).reshape((-1, 1))
    if P is not None:
        X /= (((X.dot(P)) * X).sum(1).reshape((-1, 1)))**0.5
    return X * scale

def get_steepest_descent(X, dx):
    return X[np.argmax(X.dot(dx))].reshape((-1, 1))

def show_steepest_descent_gradients(x, ax):
    plot_contour(nonquadratic, (-2, 2), (-1, 1), ax, plot_f=False)
    ax.scatter(x[0], x[1], c=black, label=r'$x$')

    # gradient + L_2
    neg_grad = -grad_nonquadratic(x)
    neg_grad /= np.linalg.norm(neg_grad)  # normalize
    neg_grad *= 0.4
    plot_vector(x, neg_grad , ax, red, '$\nabla f(x)')
    X2 = get_unit_circle(2, scale=0.4) + x.reshape((1, 2))
    ax.plot(X2[:, 0], X2[:, 1], c=red, label='$L_2$ norm')

    # L1 norm
    X1 = get_unit_circle(1, scale=0.4)
    dx_l1 = get_steepest_descent(X1, neg_grad)
    plot_vector(x, dx_l1.ravel(), ax, green)
    X1 += x.reshape((1, 2))
    ax.plot(X1[:, 0], X1[:, 1], c=green, label='$L_1$ norm')

    # scaling along axes
    X_P = get_unit_circle(2, P=np.diag([1, 4]))
    dx_P = get_steepest_descent(X_P, neg_grad)
    plot_vector(x, dx_P.ravel(), ax, black)
    X_P += x.reshape((1, 2))
    ax.plot(X_P[:, 0], X_P[:, 1], c=black, label='$P$ norm')

    # scaling using Hessian
    X_H = get_unit_circle(2, P=hessian_nonquadratic(x.reshape((-1, 1))),
                scale=0.1)
    dx_H = get_steepest_descent(X_H, neg_grad)
    plot_vector(x, dx_H.ravel(), ax, orange)
    X_H += x.reshape((1, -1))
    ax.plot(X_H[:, 0], X_H[:, 1], c=orange, label='Hessian norm')

    ax.legend(loc=3)

X1_cent = np.random.multivariate_normal([0, 0],  [[2, 1], [1, 3]], 20)
X2_cent = np.random.multivariate_normal([0, 0],  [[2, 1], [1, 3]], 20)

direction = np.array([1, -1.2])

sigmoid = lambda t : np.exp(t) / (1 + np.exp(t))

def logistic_toy(separation=0, log_lambda=1):
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
    X1 = X1_cent - separation * direction/2
    X2 = X2_cent + separation * direction/2
    ax0.scatter(X1[:,0], X1[:,1], c=orange, s=50, label='Class 1')
    ax0.scatter(X2[:,0], X2[:,1], marker='^', c=blue, s=50, label='Class 2')
    ax0.legend(loc=0)
    ax0.set_xlabel('x1')
    ax0.set_ylabel('x2')
    ax0.set_title('Scatterplot of observations')

    cross_entropy = lambda l, p : - (l * np.log(p))[p>0]
    loss_toy = lambda w : np.sum(np.log(sigmoid(X1.dot(w)))) +\
        np.sum(np.log(1 - sigmoid(X2.dot(w)))) + 10**log_lambda * np.sum(np.array(w)**2)
    plot_contour(loss_toy, (-6, 6), (-6, 6), ax1)
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')
    ax1.set_title('Contours of likelihood function')

if __name__ == '__main__':
    # make figure
    fig = show_inexact_ls(alpha=0.4, beta=0.9, Dx=10)
    fig.savefig('Figures/btls.png')
