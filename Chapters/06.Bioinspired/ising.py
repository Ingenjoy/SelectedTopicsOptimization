# -*- coding: utf-8 -*-
"""
Created on Tue 2 May 2017
Last update on Tue 9 May 2017

@author: michielstock

Implementation of Ising models
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
        - energy of the system
    """
    n, m = x.shape
    energy = 0.0
    for i in range(n):
        # use cyclic boundaries
        i_minus_1 = i - 1 if i > 0 else n - 1
        i_plus_1 = i + 1 if i < n - 1 else 0
        for j in range(m):
            # use cyclic boundaries
            j_minus_1 = j - 1 if j > 0 else m - 1
            j_plus_1 = j + 1 if j < m - 1 else 0
            energy += Jij * x[i, j] * (x[i_minus_1, j] + x[i_plus_1, j] +\
                        x[i, j_minus_1] + x[i, j_plus_1] )
    energy /= 2
    if H is not None:
        energy += np.sum(x * H)
    return - energy

def compute_delta_energy(x, i, j, Jij, H=None):
    n, m = x.shape
    # use cyclic boundaries
    i_minus_1 = i - 1 if i > 0 else n - 1
    i_plus_1 = i + 1 if i < n - 1 else 0
    j_minus_1 = j - 1 if j > 0 else m - 1
    j_plus_1 = j + 1 if j < m - 1 else 0
    delta_energy = 0.0 if H is None else 2 * H[i, j] * x[i, j]
    delta_energy += 2 * Jij * x[i, j] * (x[i_minus_1, j] + x[i_plus_1, j] +\
                        x[i, j_minus_1] + x[i, j_plus_1] )
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

def hill_climbing_ising(x0, Jij, H=None):
    x = x0.copy()
    energy = ising_energy(x, Jij)
    energies = [energy]
    local = False  # in local minimum
    while not local:
        local = True
        best_delta, best_pos = ising_best_change(x, Jij, H)
        if best_delta < 0:  # improvement
            energy += best_delta
            energies.append(energy)
            x[best_pos] = - x[best_pos]
            local = False
    return x, energies

from random import randint

def simulated_annealing_ising(x0, Jij, H, Tmax, Tmin, r, kT):
    x = x0.copy()
    n, m = x.shape
    energy = ising_energy(x, Jij)
    energies = [energy]
    T = Tmax
    while T > Tmin:
        for rep in range(kT):
            # get random point on the grid
            i, j = randint(0, n-1), randint(0, m-1)
            delta_energy = compute_delta_energy(x, i, j, Jij, H=None)
            if np.exp(-delta_energy / T) > np.random.rand():
                x[i, j] = -x[i, j]
                energy += delta_energy
                energies.append(energy)
        T *= r
    return x, energies

if __name__=='__main__':
    size = 100, 100
    n, m = size
    Jij = -1
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
