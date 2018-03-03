"""
Created on Sunday 25 February 2018
Last update: Tuesday 27 February 2018

@author: Michiel Stock
michielfmstock@gmail.com

Cell differentation exercise optimal transport
"""

import numpy as np
from optimal_transport import shuffle_rows
import matplotlib.pyplot as plt

np.random.seed(4)

class DriftingNormal():
    """Simple 2-D MVN with randomly drifting mean and covariance"""
    def __init__(self, dim=2, var_mu=0):
        super(DriftingNormal, self).__init__()
        self.dim = dim
        if var_mu:  # draw mu from a distribition
            self.mu = np.random.randn(dim) * var_mu
        else:
            self.mu = np.zeros(dim)
        self.cov = np.eye(dim)

    def set_mu(self, mu):
        self.mu[:] = mu

    def drift(self, mu_sigma=1, cov_sigma=1, alpha_sigma=0.05):
        self.mu += np.random.randn(*self.mu.shape) * mu_sigma
        # random covariance matrix
        T = np.random.randn(*self.cov.shape) * cov_sigma
        TTT = T @ T.T
        scaling = np.linalg.det(TTT)**(1/self.dim)
        TTT /= scaling
        self.cov *= 1 - alpha_sigma
        self.cov += alpha_sigma * TTT

    def sample(self, n_obs=1):
        return np.random.multivariate_normal(self.mu, cov=self.cov, size=n_obs)

n_centers = 5
n_steps = 10
mu_sigma = 1.5
n_obs = 20
growth_rate = 1.1
increase_mu = 1.05
var_mu = 6

drifting_normals = [DriftingNormal(var_mu=var_mu) for _ in range(n_centers)]

cells_measured_by_time = []

xlim = [0, 0]
ylim = [0, 0]

for time_step in range(n_steps):
    cells = np.concatenate([dn.sample(n_obs) for dn in drifting_normals],
                                        axis=0)
    shuffle_rows(cells)
    cells_measured_by_time.append(cells)
    n_obs = int(n_obs * growth_rate)
    for dn in drifting_normals:
        dn.drift(mu_sigma=mu_sigma)
    mu_sigma *= increase_mu
    # update limits
    xlim[0] = min(np.min(cells[:,0]), xlim[0])
    xlim[1] = max(np.max(cells[:,0]), xlim[1])
    ylim[0] = min(np.min(cells[:,1]), ylim[0])
    ylim[1] = max(np.max(cells[:,1]), ylim[1])

if __name__ == '__main__':

    uniform_weights = lambda n : np.ones(n) / n

    from matplotlib.animation import FuncAnimation
    from optimal_transport import *

    lam = 0.01

    cmap = plt.get_cmap('gnuplot')

    # make giff of cells
    fig, ax = plt.subplots()
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_ylabel(r'$x_2$')
    ax.set_xlabel(r'$x_1$')

    scatter = ax.scatter(cells_measured_by_time[0][:,0],
                            cells_measured_by_time[0][:,1])

    def animate(i):
        scatter.set_offsets(cells_measured_by_time[i])
        scatter.set_color(cmap(i / 10))
        ax.set_title('Time step {}'.format(i+1))

    anim = FuncAnimation(
        fig, animate, interval=500, frames=n_steps)

    anim.save('Figures/cell_diff.gif', writer='imagemagick')

    #fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 16))
    #axes = axes.flatten()

    #for t, ax in enumerate(axes):
    for t in range(9):
        fig, ax = plt.subplots()
        Xt = cells_measured_by_time[t]
        mt = len(Xt)
        Xtp1 = cells_measured_by_time[t+1]
        mtp1 = len(Xtp1)
        M = pairwise_distances(Xt, Xtp1, metric="sqeuclidean")
        P, _ = compute_optimal_transport(M, uniform_weights(mt),
                        uniform_weights(mtp1), lam=lam, epsilon=1e-5)
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_ylabel(r'$x_2$')
        ax.set_xlabel(r'$x_1$')
        # scatter plot
        for X, color in zip([Xt, Xtp1], [orange, green]):
            ax.scatter(X[:,0], X[:,1],
                                color=color, alpha=0.8)
        # mapping
        for i in range(mt):
            for j in range(i, mtp1):
                if P[i,j] > 1e-5:
                    ax.plot([Xt[i,0], Xtp1[j,0]], [Xt[i,1], Xtp1[j,1]],
                    color=red, alpha=P[i,j] * mt)

        ax.set_title('Mapping between time step {} (orange)\n and {} (green)'.format(t+1, t+2))

        fig.tight_layout()
        fig.savefig('Figures/cell_diff_maps_{}.png'.format(t+1))
