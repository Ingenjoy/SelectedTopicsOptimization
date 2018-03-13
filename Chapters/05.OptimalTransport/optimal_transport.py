"""
Created on Tuesday 20 February 2018
Last update: Sunday 11 March 2018

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

def monge_brute_force(C):
    """
    Solves the Monge assignment problem using
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


def compute_optimal_transport(C, a, b, lam, epsilon=1e-8,
                verbose=False, return_iterations=False):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - C : cost matrix (n x m)
        - a : vector of marginals (n, )
        - b : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter
        - verbose : report number of steps while running
        - return_iterations : report number of iterations till convergence,
                default False

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
        - n_iterations : number of iterations, if `return_iterations` is set to
                        True
    """
    n, m = C.shape
    P = np.exp(- lam * C)
    iteration = 0
    while True:
        iteration += 1
        u = P.sum(1)  # marginals of rows
        max_deviation = np.max(np.abs(u - a))
        if verbose: print('Iteration {}: max deviation={}'.format(
                            iteration, max_deviation
                        ))
        if max_deviation < epsilon:
            break
        # scale rows
        P *= (a / u).reshape((-1, 1))
        # scale columns
        P *= (b / P.sum(0)).reshape((1, -1))
    if return_iterations:
        return P, np.sum(P * C), iteration
    else:
        return P, np.sum(P * C)

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
        - a: barrycenter (probability vector)
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
    plt.savefig('Figures/simpleinterpol.png')

    # images
    dim = 50
    B = np.zeros((dim**2, 4))

    # square
    square = np.zeros((dim, dim))
    square[dim//4:-dim//4,:][:,dim//4:-dim//4] = 1
    B[:,0] = square.flatten()

    # circle
    circle = np.array([[(i-dim/2)**2+(j-dim/2)**2 < dim**2 / 4**2 for i in range(dim)]
        for j in range(dim)], dtype=float)
    circle -= np.array([[(i-dim/2)**2+(j-dim/2)**2 < dim**2 / 6**2 for i in range(dim)]
        for j in range(dim)], dtype=float)
    B[:,1] = circle.flatten()

    # diamond
    diamond = np.array([[np.abs(i-dim/2)+np.abs(j-dim/2) < dim/4 for i in range(dim)]
        for j in range(dim)], dtype=float)
    B[:,2] = diamond.flatten()

    """
    # cross
    cross = np.zeros((dim, dim))
    cross[dim//3:-dim//3,:] = 1
    cross[:, dim//3:-dim//3] = 1
    B[:,3] = cross.flatten()
    """

    # two blobs

    two_blobs = np.zeros((dim, dim))
    two_blobs[:] += np.array([[(i-dim/4)**2+(j-dim/4)**2 < (dim/10)**2 for i in range(dim)]
        for j in range(dim)], dtype=float)
    two_blobs[:] += np.array([[(i-dim/4*3)**2+(j-dim/4*3)**2 < (dim/10)**2 for i in range(dim)]
        for j in range(dim)], dtype=float)
    B[:,3] = two_blobs.flatten()

    B /= B.sum(0)


    C = pairwise_distances([[i/dim, j/dim] for i in range(dim)
    for j in range(dim)], metric="sqeuclidean")

    A = np.zeros((dim**2, 25))

    image_nr = 0
    for di in np.linspace(0, 1, 5):
        for dj in np.linspace(0, 1, 5):
            weights = np.array([
                    (1-di) * (1-dj),
                    di * (1-dj),
                    (1-di) * dj,
                    di * dj
            ])
            A[:,image_nr] = barrycenters_entropic(B, C, weights,
                            lam=500, L=50)[:,0]
            image_nr += 1

    fig, axes = plt.subplots(nrows=5, ncols=5)

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            ax.imshow(A[:,i+5*j].reshape((dim,dim)) > 1e-5, cmap='Greys',
                            interpolation='nearest')
            ax.set_yticks([])
            ax.set_xticks([])
    plt.savefig('Figures/barrycenters.png')
