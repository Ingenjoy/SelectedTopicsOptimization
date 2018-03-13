"""
Created on Sunday 11 March 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Example figure of optimal transport.
"""

from optimal_transport import *
import matplotlib.pyplot as plt

# normal distribution
x = np.linspace(-4, 4, num=100)
a = np.exp(- x**2)
a /= a.sum()  # normalize

b = np.exp(- (x-1.2)**2/0.3) + 0.5 * np.exp(-(x+0.7)**2 / 0.5)
b /= b.sum()

C = pairwise_distances(x.reshape((-1,1)), metric="sqeuclidean")
P, _ = compute_optimal_transport(C, a, b, lam=50)

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[1,0].plot(a, x, lw=3, color=orange)
axes[1,0].set_xlim([a.max()+0.01, 0])
axes[0,1].plot(x, b, lw=3, color=yellow)
axes[1,1].imshow(P, cmap='Greens')

for ax in axes.flatten():
    ax.set_yticks([])
    ax.set_xticks([])

fig.savefig('Figures/ot_example.png')
