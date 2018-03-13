"""
Created on Tuesday 6 March 2018
Last update: Sunday 11 March 2018

@author: Michiel Stock
michielfmstock@gmail.com

Simple example of matching points

makes a data set the figures
"""

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from optimal_transport import red, green, yellow, orange, blue, black

np.random.seed(42)


X1, X2, y1, y2 = train_test_split(*make_blobs(n_samples=300, cluster_std=2.1),
        test_size=0.6)

for cl in range(3):
    mi = sum(y2==cl)  # number of cells of this class
    # add noise
    X2[y2==cl,:] += np.random.randn(mi, 2) * 2 #+ np.array([[2, -3]])
    # add drift
    X2[y2==cl,:] += np.random.randn(1, 2) * 6 - 1

def plot_cells(ax):
    """
    Plots the cells for the cell differentiation exercise on an axis
    """
    for cl, col in zip(range(3), [blue, orange, green]):
        X = X1[y1==cl,:]
        ax.scatter(X[:,0], X[:,1], color=col,
                    label=r'cluster {} ($t_1$)'.format(cl+1), zorder=3)

    ax.scatter(X2[:,0], X2[:,1], color='gray', alpha=0.8,
                    label=r'$t_2$', zorder=2)
    ax.set_xlabel(r'Expression gene 1')
    ax.set_ylabel(r'Expression gene 2')
    #ax.set_xlabel(r'$x_1$')
    #ax.set_ylabel(r'$x_2$')
    ax.legend(loc=0)

if __name__ == '__main__':
    from optimal_transport import *

    fig, ax = plt.subplots()

    def plot_cells(ax):
        for cl, col in zip(range(3), [blue, orange, green]):
            X = X1[y1==cl,:]
            ax.scatter(X[:,0], X[:,1], color=col,
                        label=r'cluster {} ($t_1$)'.format(cl+1), zorder=3)

        ax.scatter(X2[:,0], X2[:,1], color='gray', alpha=0.8,
                        label=r'$t_2$', zorder=2)
        ax.set_xlabel(r'Expression gene 1')
        ax.set_ylabel(r'Expression gene 2')
        #ax.set_xlabel(r'$x_1$')
        #ax.set_ylabel(r'$x_2$')
        ax.legend(loc=0)

    plot_cells(ax)
    fig.savefig('Figures/cells_diff.png')

    C = pairwise_distances(X1, X2, metric="euclidean")

    n, p = X1.shape
    m, p = X2.shape

    P, _ = compute_optimal_transport(C, np.ones(n)/n,
                    np.ones(m)/m, lam=10)

    fig, ax = plt.subplots()
    plot_cells(ax)

    # add path
    for i in range(n):
        for j in range(m):
            alpha = P[i,j] * m
            if alpha > 1e-2:
                ax.plot([X1[i,0], X2[j,0]], [X1[i,1], X2[j,1]], color=red,
            alpha=alpha, zorder=1)

    # find drift
    for cl in range(3):
        m_cl_1 = X1[y1==cl].mean(0)
        ax.scatter(*m_cl_1.tolist(), color=red)

        m_cl_2 = (P[y1==cl,:].sum(1, keepdims=True)**-1 * P[y1==cl,:] @ X2).mean(0)
        ax.scatter(*m_cl_2.tolist(), color=yellow)
        ax.arrow(*m_cl_1.tolist(), *(m_cl_2-m_cl_1).tolist(), color=yellow,
                lw=4, head_width=1, head_length=1, zorder=4)

    fig.savefig('Figures/cells_diff_matched.png')
