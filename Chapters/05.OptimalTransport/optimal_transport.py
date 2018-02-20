"""
Created on Tuesday 20 February 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Solution for the lecture optimal transport
"""

from itertools import permutations
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'


def kantorovich_brute_force(C):
    """
    Solves the Kantorovich assignment problem using
    brute force.

    Inputs:
        - C: cost matrix (square, size n x n)

    Outputs:
        - best_perm: optimal assigments (list of n indices matching the rows
                to the columns)
        - best_cost: optimal cost corresponding to the best permutation

    DO NOT USE FOR PROBLEMS OF A SIZE LARGER THAN 12!!!
    """
    n, m = C.shape
    assert n==m  # C should be square
    best_perm = None
    best_cost = np.inf
    for perm in permutations(range(n)):
        cost = 0
        for i, p_i in enumerate(perm):
            cost += C[i, p_i]
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return best_perm, best_cost


def compute_optimal_transport(M, r, c, lam, epsilon=1e-8,
                verbose=False):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter
        - verbose : report number of steps while running

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    iteration = 0
    while True:
        iteration += 1
        u = P.sum(1)  # marginals of rows
        error = np.max(np.abs(u - r))
        if verbose: print('Iteration {}: error={}'.format(
                            iteration, error
                        ))
        if error < epsilon:
            break
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * M)
