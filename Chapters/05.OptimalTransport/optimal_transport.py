"""
Created on Tuesday 20 February 2018
Last update: Saturday 03 March 2018

@author: Michiel Stock
michielfmstock@gmail.com

Solution for the lecture optimal transport
"""

from itertools import permutations
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from random import shuffle

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

def shuffle_rows(X):
    """
    Randomly shuffles rows of a numpy matrix
    """
    n, _ = X.shape
    indices = list(range(n))
    shuffle(indices)
    X[:] = X[indices]

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

def barrycenters_entropic(B, C, weights, lam=1, L=100):
    """
    Finds a barrycenter of Sinkhorn distances for a set of
    proabiltiy vectors (of same dimension) with a cost matrix
    and a set of weigths.

    Inputs:
        - B: matrix containing the probability vectors (n x s), positive
            and normalized
        - C: cost matrix (n x n), same everywhere
        - weigths: s weights for the relative importance
        - lam: strength of the entropic regularization
        - L: number of Sinkhorn updates

    Output:
        - a: barycenter (probability vector)
    """
    n, s = B.shape
    K = np.exp(- lam * C)
    U = np.ones_like(B) / n
    V = np.zeros_like(B)
    a = np.zeros((n, 1))
    for step in range(L):
        # update V
        V[:] = B / (K.T  @ U)
        # update a
        a[:] = np.exp(np.sum(np.log(K @ V) * weights.reshape((1, s)), axis=1, keepdims=True))
        U[:] = a / (K @ V)
    return a

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    B = np.zeros((10, 2))
    B[:5,0] = 0.2
    B[5:,1] = 0.2

    C = pairwise_distances(np.arange(10).reshape((-1,1)),
                metric='sqeuclidean')

    A = np.zeros((10, 11))
    for i in range(11):
        A[:,i] = barrycenters_entropic(B, C, np.array((1-i/10, 0+i/10)))[:,0]

    plt.imshow(A, interpolation='nearest')
    plt.show()

    # images
    B = np.zeros((400, 4))

    # square
    square = np.zeros((20, 20))
    square[5:-5,:][:,5:-5] = 1
    B[:,0] = square.flatten()

    # circle
    circle = np.array([[(i-9.5)**2+(j-9.5)**2<8**2for i in range(20)]
        for j in range(20)], dtype=float)
    B[:,1] = circle.flatten()

    # diamond
    diamond = np.array([[np.abs(i-9.5)+np.abs(j-9.5) < 8for i in range(20)]
        for j in range(20)], dtype=float)
    B[:,2] = diamond.flatten()

    # cross
    cross = np.zeros((20, 20))
    cross[5:-5,:] = 1
    cross[:, 5:-5] = 1
    B[:,3] = cross.flatten()

    B /= B.sum(0)


    C = pairwise_distances([[i, j] for i in range(20)
    for j in range(20)], metric="sqeuclidean")

    A = np.zeros((400, 25))

    image_nr = 0
    for di in np.linspace(0, 1, 5):
        for dj in np.linspace(0, 1, 5):
            weights = np.array([
                    (1-di) * (1-dj),
                    di * (1-dj),
                    (1-di) * dj,
                    di * dj
            ])
            A[:,image_nr] = barrycenters_entropic(B, C, weights, lam=1)[:,0]
            image_nr += 1

    fig, axes = plt.subplots(nrows=5, ncols=5)

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            ax.imshow(A[:,i+5*j].reshape((20,20))>1e-4, cmap='Greys',
                            interpolation='nearest')
            ax.set_yticks([])
            ax.set_xticks([])
    plt.savefig('Figures/barrycenters.png')
