# -*- coding: utf-8 -*-
"""
Created on Wed 20 Jan 2016
Last update: Fri 03 Mar 2017

@author: Michiel Stock
michielfmstock@gmail.com

Some functions for Chapter 03: Constrained optimization
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

# VISUALIZATION
# -------------


def plot_contour(f, xlim, ylim, ax, A=None, b=None):
    '''
    Plots the contour of a 2D function to be minimized
    adds a linear constraint of the form
        Ax = b
    '''
    xvals = np.linspace(*xlim)
    yvals = np.linspace(*ylim)
    X, Y = np.meshgrid(xvals, yvals)
    Z = np.reshape(list(map(f, zip(X.ravel().tolist(), Y.ravel().tolist()))),
                   X.shape)
    ax.contour(X, Y, Z)
    ax.contourf(X, Y, Z, cmap='bone')
    if A is not None and b is not None:
        ymap = (b - A[0] * xvals) / A[1]
        ax.plot(xvals, ymap, green, label='linear constraint')
        ax.set_ylim(ylim)


def add_path(ax, x_steps, col=blue):
    '''
    Adds a path of an opitmization algorithm to a figure
    '''
    ax.plot([x[0] for x in x_steps], [x[1] for x in x_steps], c=col)

# EXAMPLES
# --------


# defining the quadric function, gradient and hessian

def quadratic(x, gamma=4):
    return 0.5 * (x[0]**2 + gamma * x[1]**2)

def grad_quadratic(x, gamma=4):
    return np.array([x[0], gamma * x[1]])

def hessian_quadratic(x, gamma=4):
    return np.array([[1, 0], [0, gamma]])


# defining the non-quadric function, gradient and hessian


x1_, x2_ = sp.symbols('x1, x2')

nonquad_expr = sp.log(sp.exp(x1_ + 3*x2_ - 0.1) + sp.exp(x1_-3*x2_-0.1) +
                    sp.exp(-x1_ - 0.1))
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

def nonquadratic_ineq_const(x):
    if (x[0] - 1)**2 + (x[1] - 0.25)**2 - 4 < 0:
        return np.log(nonquadratic(x))
    else:
        return 2

def plot_log_barrier(t):
    """
    Plots the logaritmic barrier for given t
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([-3, 0, 0], [0, 0, 10], 'r--', label=r'$I_-(u)$')
    u = np.linspace(-3, 0, num=100000, endpoint=False)
    y = - (1 / t) * np.log(- u)
    ax.plot(u, y, label=r'$- (1/t)\log(-u)$')
    ax.set_xlabel('u')
    ax.set_ylim([-5, 10])
    ax.set_xlim([-3, 1])
    ax.legend(loc=2)

if __name__ == '__main__':
    pass
