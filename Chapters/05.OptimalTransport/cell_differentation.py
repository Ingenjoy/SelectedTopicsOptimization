"""
Created on Sunday 25 February 2018
Last update: Monday 26 February 2018

@author: Michiel Stock
michielfmstock@gmail.com

Cell differentation exercise optimal transport
"""

import numpy as np
from optimal_transport import shuffle_rows
import matplotlib.pyplot as plt

np.random.seed(42)

class DriftingNormal():
    """Simple 2-D MVN with randomly drifting mean and covariance"""
    def __init__(self, dim=2):
        super(DriftingNormal, self).__init__()
        self.dim = dim
        self.mu = np.zeros(dim)
        self.cov = np.eye(dim)

    def drift(self, mu_sigma=1, cov_sigma=1):
        self.mu += np.random.randn(*self.mu.shape) * mu_sigma
        # random rotation matrix
        T = np.random.randn(*self.cov.shape) * cov_sigma
        #self.cov[:] = T @ self.cov @ T.T + np.eye(self.dim)
        #TODO: decrease random rotation
        #self.cov /= np.linalg.det(self.cov)**0.5  # normalize volume
        #self.cov *= (np.random.rand() - 0.5) * 0.25 + 0.5

    def sample(self, n_obs):
        return np.random.multivariate_normal(self.mu, cov=self.cov, size=n_obs)

n_centers = 10
n_steps = 10
mu_sigma = 1.2
n_obs = 50
growth_rate = 1.2
increase_mu = 1.05

drifting_normals = [DriftingNormal() for _ in range(n_centers)]

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

    lam = 1e1

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

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 16))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        Xi = cells_measured_by_time[i]
        mi = len(Xi)
        Xip1 = cells_measured_by_time[i+1]
        mip1 = len(Xip1)
        M = pairwise_distances(Xi, Xip1, metric="seuclidean")
        P, _ = compute_optimal_transport(M, uniform_weights(mi),
                        uniform_weights(mip1), lam=lam)
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_ylabel(r'$x_2$')
        ax.set_xlabel(r'$x_1$')
        # scatter plot
        for j, color in zip([i, i+1], [orange, green]):
            ax.scatter(cells_measured_by_time[j][:,0],
                                cells_measured_by_time[j][:,1],
                                color=color, alpha=0.8)
        # mapping
        for i in range(mi):
            for j in range(mip1):
                if P[i,j] > 1e-8:
                    ax.plot([Xi[i,0], Xip1[j,0]], [Xi[i,1], Xip1[j,1]],
                    color=red, alpha=P[i,j] * mi)

        ax.set_title('Mapping between time step {} (orange)\n and {} (green)'.format(i+1, i+2))

    fig.tight_layout()
    fig.savefig('Figures/cell_diff_maps.png')
