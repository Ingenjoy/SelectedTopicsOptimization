# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:13:18 2016
Last update on Sun Jun 12 2016

@author: michielstock

Create alignment independent features for proteins
"""

import numpy as np
#import numba

#@numba.jit
def ising_energy(x, Jij, H=None):
    """
    Computes the engery of an Ising model

    Inputs:
        - x : binary n x m matrix of spin states (value +1 or -1)
        - Jnm : magnetic coupling between neighbouring spins (typically +1 or -1)
        - H : magnetic field or potential on the spins

    Outputs:
        - engery of the system
    """
    n, m = x.shape
    energy = 0.0
    for i in range(n):
        for j in range(m):
            if i > 0:
                energy += Jij * x[i, j] * x[i-1, j]
            if i < n-1:
                energy += Jij * x[i, j] * x[i+1, j]
            if j > 0:
                energy += Jij * x[i, j] * x[i, j-1]
            if j < m-1:
                energy += Jij * x[i, j] * x[i, j+1]
    energy /= 2
    if H is not None:
        energy += np.sum(x * H)
    return - energy

def compute_delta_energy(x, i, j, Jij, H=None):
    n, m = x.shape
    delta_energy = 0.0 if H is None else 2 * H[i, j]
    if i > 0:
        delta_energy += Jij * x[i, j] * x[i-1, j]
    if i < n-1:
        delta_energy += Jij * x[i, j] * x[i+1, j]
    if j > 0:
        delta_energy += Jij * x[i, j] * x[i, j-1]
    if j < m-1:
        delta_energy += Jij * x[i, j] * x[i, j+1]
    return delta_energy

def ising_best_change(x, Jij, H=None):
    """
    Computes the change in energy by flipping each bit

    Inputs:
        - x : binary n x m matrix of spin states (value +1 or -1)
        - Jnm : magnetic coupling between neighbouring spins (typically +1 or -1)
        - H : magnetic field or potential on the spins

    Yields:
        - (change in energy, (i, j))
    """
    n, m = x.shape
    best_delta = 1e100
    for i in range(n):
        for j in range(m):
            delta_energy = compute_delta_energy(x, i, j, Jij, H)
            if delta_energy < best_delta:
                best_pos = i, j
                best_delta = delta_energy
    return best_delta, best_pos


def random_ising(size, p=0.5):
    x = np.random.binomial(1, p, size)
    x[x==0] = -1
    H = np.random.randn(*x.shape)
    return x

if __name__=='__main__':
    size = 100, 100
    n, m = size
    Jij = 1
    H = np.random.randn(*size)
    #H = None
    x = random_ising(size)

    e1 = ising_energy(x, Jij, H)

    d = compute_delta_energy(x, 10, 15, Jij, H)

    x[10, 15] = - x[10, 15]

    e2 = ising_energy(x, Jij, H)
    print('Should be the same:')
    print(e2 - e1, d)
    print()

    best_delta, (best_pos) = ising_best_change(x, Jij, H)

    x_new = x.copy()
    x_new[best_pos] = - x_new[best_pos]

    print('Best difference:')
    print(best_delta, ising_energy(x_new, Jij, H) - ising_energy(x, Jij, H))
