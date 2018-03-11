"""
Created on Tuesday 6 March 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Examples of probability distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from optimal_transport import red, green, yellow, orange, blue, black

figsize=(8, 4)

# discrete distribution

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=figsize)

x_hist = np.random.randn(100) / 3 + 4
ax0.hist(x_hist, color=green)
ax0.set_title('Histogram')
ax0.set_xlabel(r'$x$')
ax0.set_ylabel('Probability density')

x = np.arange(10)
y = np.exp(-x/2)
y /= y.sum()
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\mathcal{P}(X=x)$')
ax1.scatter(x, y, color=orange)
for xi, yi in zip(x, y):
    ax1.plot([xi, xi], [0, yi], color=blue, linestyle='--')
ax1.set_ylim([0, max(y)+0.1])
ax1.set_xlim([-0.5, 10])
ax1.set_title('Discrete probability\ndistribution')

fig.tight_layout()
fig.savefig('Figures/discrete_probs1.png')

# scatter

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=figsize)

ax0.scatter(np.random.rand(40), np.random.rand(40), color=red)
ax0.scatter(np.random.rand(40)+1, np.random.rand(40)+1, color=red)
ax0.set_xlim([-0.2,2.2])
ax0.set_ylim([-0.2,2.2])
ax0.set_title('Unweighted points')

ax1.scatter(np.random.rand(40), np.random.rand(40), s=10, color=yellow)
ax1.scatter(np.random.rand(40)+1, np.random.rand(40)+1, s=50, color=yellow)
ax1.set_xlim([-0.2,2.2])
ax1.set_ylim([-0.2,2.2])
ax1.set_title('Weighted points')

for ax in (ax0, ax1):
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

fig.tight_layout()
fig.savefig('Figures/discrete_probs2.png')
