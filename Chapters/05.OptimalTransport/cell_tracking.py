"""
Created on Tuesday 20 February 2018
Last update: Tuesday 27 February 2018

@author: Michiel Stock
michielfmstock@gmail.com

Cell tracking exercise
"""

import numpy as np
import matplotlib.pyplot as plt
from optimal_transport import red, green, yellow, orange, blue, black
from optimal_transport import shuffle_rows

# plot cell
def plot_cells(X1, X2):
    fig, ax = plt.subplots()
    ax.scatter(X1[:,0], X1[:,1], s=50, color=orange,
                        label=r'location cells at $t_1$', zorder=2)
    ax.scatter(X2[:,0], X2[:,1], s=50, color=green,
                        label=r'location cells at $t_2$', zorder=2)
    ax.set_ylabel(r'$y$')
    ax.set_xlabel(r'$x$')
    ax.legend(loc=2)
    return fig, ax

np.random.seed(2)

# generate cells
n_cells = 10
sigma = 8  # initial spread
delta_sigma = 6  # random movement
drift = np.array([[-1.5, 3]])  # systematic drift

X1 = np.random.randn(n_cells, 2) * sigma
X2 = X1 + np.random.randn(n_cells, 2) * delta_sigma  + drift

shuffle_rows(X2)

if __name__ == '__main__':

    from optimal_transport import *


    fig, ax = plot_cells(X1, X2)
    fig.savefig('Figures/cells_locations.png')

    fig, ax = plot_cells(X1, X2)
    C = pairwise_distances(X1, X2, metric='sqeuclidean')
    best_perm, best_cost = monge_brute_force(C)

    for i, i_m in enumerate(best_perm):
        ax.plot([X1[i,0], X2[i_m,0]], [X1[i,1], X2[i_m,1]],
                color=red, alpha=1, zorder=1, lw=2)
    fig.savefig('Figures/cells_locations_matched.png')
