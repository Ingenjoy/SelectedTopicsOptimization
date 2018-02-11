"""
Created on Saturday 26 August 2017
Last update: Friday 26 January 2018

@author: Michiel Stock
michielfmstock@gmail.com

Data and solution of the signal recovery problem
"""

import sys
import numpy as np
from random import randint
#import argparse
import matplotlib.pyplot as plt
blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'
from math import pi
import matplotlib.pyplot as plt

periodic_fun = lambda x : 3 * np.sin(x * 2 * pi / n) +\
            2 * np.cos(x * 4 * pi / n) +\
            np.sin(x * 4 *pi / n) + 0.8 * np.cos(x * 12 * pi / n)
"""

I removed the argument parser. It does not mix with loading this as a module.

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-n', '--n_points', type=int, default=1000,
                help='number of points (default: 1000)' )
arg_parser.add_argument('-m', '--n_measurements', type=int, default=100,
                help='number of measurements (default: 100)')
arg_parser.add_argument('-noise', type=float, default=1,
                help='std of noise (default: 1)')
arg_parser.add_argument('-C', type=float, default=1,
                help='regularization (default: 1)')
arg_parser.add_argument('-gamma', type=float, default=100,
                help='length scale of kernel (default: 100)')
args = arg_parser.parse_args()


n = args.n_points
m = args.n_measurements
noise = args.noise
C = args.C
gamma = args.gamma
reach = min(n//2, gamma * 3)
"""

f = periodic_fun
n = 1000
m = 100
noise = 1
C = 1
gamma = 100
reach = min(n//2, gamma * 3)

def generate_noisy_measurements(m, n, sigma=noise):
    """
    Generate noisy measurements according to some function f

    Inputs:
        - m : number of observations
        - n : dim of x
        - sigma : normally distributed noise (default = 1)

    Output:
        - y : vector of noisy measurements
        - I : vector of indices
    """
    I = np.random.randint(0, n-1, size=m)
    y = f(I) + np.random.randn(m) * sigma
    return y.reshape((-1, 1)), I

def make_connection_matrix(n, gamma=gamma):
    """
    Generates the kernel matrix and the inverse kernel matrix

    Uses gamma as a characteristic length scale of the radial basis kernel,
    assumes periodic boundaries.

    Inputs:
        - n : number of points
        - gamma : length scale

    Output:
        - K
        - Kinv
    """
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = 1
        for j in range(i):
            dsq = min((i-j)**2, (n-(i-j))**2)  # assume periodict boundaries
            kij = np.exp(-dsq / gamma**2)
            M[i,j] = kij
            M[j,i] = kij
    M += 1e-2 * np.eye(n)
    return M, np.linalg.inv(M)

def make_bookkeeping(I, n):
    """
    Constructs the bookkeeping matrix

    Inputs:
        - I : the indices of the measurements
        - n : dimensionality of signal vector

    Output:
        - L : m x n bookkeeping matrix
    """
    L = np.zeros((m, n), dtype=int)
    for i, j in enumerate(I):
        L[i,j] = 1
    return L


if __name__ == '__main__':
    np.random.seed(10)


    y, I = generate_noisy_measurements(m, n)
    R = make_bookkeeping(I, n)
    _, Kinv = make_connection_matrix(n)

    ivals = np.arange(n)

    # standard form
    P = R.T @ R + C * Kinv
    q = -R.T @ y

    x = np.linalg.solve(P, -q)

    fig, ax = plt.subplots()

    ax.plot(ivals, periodic_fun(ivals), color=blue, label='true signal', linewidth=2)
    ax.scatter(I, y, color=orange, label='observations ($y_i$)')
    ax.plot(ivals, x, color=green, label='recovered signal ($x_i$)', linewidth=2)
    ax.set_xlabel('$j$')
    ax.set_ylabel('value')

    ax.legend(loc=0)
    fig.savefig('Figures/signal.png')

    # Solve using gradient descent and momentum
    from quadratic import gradient_descent_quadratic_momentum
    betas = np.arange(0, 1, step=0.1)
    Cs = np.logspace(-2, 2, 5)

    # save x_stars
    xstars_by_C = {}
    steps = np.zeros((len(betas), len(Cs)), dtype=int)

    from progress.bar import Bar

    bar = Bar('Computing...', max=len(Cs)*len(betas))

    for j, C in enumerate(Cs):
        P = R.T @ R + C * Kinv
        for i, beta in enumerate(betas):
            bar.next()
            xstar, n_steps = gradient_descent_quadratic_momentum(P, q,
                        np.zeros((1000, 1)), beta=beta, trace=True, epsilon=1e-3)
            xstars_by_C[C] = xstar
            steps[i,j] = n_steps
    bar.finish()

    # show table
    import pandas as pd

    print(pd.DataFrame(steps, index=betas, columns=Cs))

    fig, axes = plt.subplots(ncols=2, figsize=(12,8))

    axes[0].imshow(np.log10(steps), interpolation='nearest', cmap='Greens')
    axes[0].set_xticks(range(len(Cs)))
    axes[0].set_yticks(range(len(betas)))
    axes[0].set_xticklabels(Cs)
    axes[0].set_yticklabels(betas)
    axes[0].set_xlabel(r'$C$')
    axes[0].set_ylabel(r'$\beta$')
    axes[0].set_title('Number of steps\n(log scale, darker=more steps)')


    axes[1].scatter(I, y, color=orange, label='observations ($y_i$)')
    axes[1].plot(ivals, periodic_fun(ivals), color=green, label='true signal',
                    alpha=0.7, linewidth=2)
    for i, C in enumerate(Cs):
        color = plt.get_cmap('hot')(i / len(Cs))
        axes[1].plot(xstars_by_C[C], color=color, linewidth=2,
                    label='recovered, C={}'.format(C))
    axes[1].legend(loc=0)
    axes[1].set_title('Effect of $C$ on signal recovery')

    fig.tight_layout()
    fig.savefig('Figures/signal_study.png')
