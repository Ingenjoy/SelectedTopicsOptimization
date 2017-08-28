"""
Created on Saturday 26 August 2017
Last update: Sunday 27 August 2017

@author: Michiel Stock
michielfmstock@gmail.com

Data and solution of the signal recovery problem
"""

import numpy as np
from random import randint

periodic_fun = lambda x : 3 * np.sin(x * 2 * pi / n) +\
            2 * np.cos(x * 4 * pi / n) +\
            np.sin(x * 4 *pi / n) + 0.8 * np.cos(x * 12 * pi / n)

n = 1000
m = 100
noise = 1
C = noise
gamma = 100
reach = min(n//2, gamma * 3)

def generate_noisy_measurements(m, n, f=periodic_fun, sigma=noise):
    """
    Generate noisy measurements according to some function f

    Inputs:
        - m : number of observations
        - n : dim of x
        - f : function
        - sigma : normally distributed noise (default = 1)

    Output:
        - y : vector of noisy measurements
        - I : vector of indices
    """
    I = np.random.randint(0, n-1, size=m)
    y = f(I) + np.random.randn(m) * sigma
    return y.reshape((-1, 1)), I

def make_connection_matrix(n, gamma=gamma, reach=reach):
    """
    Generates the inverse kernel matrix.
    Only has a certain reach

    Inputs:
        - n : number of points
        - gamma : length scale
        - reach : only consider this many neighbors (left and right)

    Output:
        - Kinv
    """
    M = np.zeros((n, n))
    for i in range(n):
        M[i, i] = 1
        for j in range(i):
            dsq = min((i-j)**2, (n-i+j)**2)  # assume periodict boundaries
            kij = np.exp(-dsq / gamma**2)
            M[i,j] = kij
            M[j,i] = kij
    M[:] = np.linalg.inv(M)
    return M

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
    from sys import path
    path.append('../../Scripts')
    from plotting import plt, blue, orange, green, red, yellow
    import seaborn as sns
    sns.set_style('white')
    from math import pi
    import matplotlib.pyplot as plt

    np.random.seed(10)

    y, I = generate_noisy_measurements(m, n)
    L = make_bookkeeping(I, n)
    Kinv = make_connection_matrix(n)

    ivals = np.arange(n)

    x = np.linalg.solve(L.T @ L + C * Kinv, L.T @ y)

    fig, ax = plt.subplots()

    ax.plot(ivals, periodic_fun(ivals), color=blue, label='true signal')
    ax.scatter(I, y, color=orange, label='observations ($y_i$)')
    ax.plot(ivals, x, color='green', label='recovered signal ($f_i$)')
    ax.set_xlabel('$i$')
    ax.set_ylabel('value')

    ax.legend(loc=0)
    fig.show()
