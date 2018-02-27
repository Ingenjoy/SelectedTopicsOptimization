"""
Created on Tuesday 20 February 2018
Last update: Sunday 25 February 2018

@author: Michiel Stock
michielfmstock@gmail.com

Cell tracking exercise
"""

import numpy as np
from optimal_transport import shuffle_rows

np.random.seed(42)

# generate cells
n_cells = 10
sigma = 10  # initial spread
delta_sigma = 3  # random movement
drift = np.array([[-1.5, 3]])  # systematic drift

X1 = np.random.randn(n_cells, 2) * sigma
X2 = X1 + np.random.randn(n_cells, 2) * delta_sigma  + drift

shuffle_rows(X2)  # make probem a bit harder

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from optimal_transport import *


    # plot cell
    def plot_cells():
        fig, ax = plt.subplots()
        ax.scatter(X1[:,0], X1[:,1], s=50, color=orange,
                            label=r'location cells at $t_1$')
        ax.scatter(X2[:,0], X2[:,1], s=50, color=green,
                            label=r'location cells at $t_2$')
        ax.set_ylabel(r'$y$')
        ax.set_xlabel(r'$x$')
        ax.legend(loc=0)
        return fig, ax

    fig, ax = plot_cells()
    fig.savefig('Figures/cells_locations.png')

    fig, ax = plot_cells()
    C = pairwise_distances(X1, X2, metric='seuclidean')
    best_perm, best_cost = kantorovich_brute_force(C)

    for i, i_m in enumerate(best_perm):
        ax.plot([X1[:,0], X2[:,0]], [X1[:,1], X2[:,1]],
                color=red, alpha=0.5)
    fig.savefig('Figures/cells_locations_matched.png')
