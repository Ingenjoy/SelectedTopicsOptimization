"""
Created on Sunday 11 February 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Illustrate the logaritmic barrier
"""

from teachingtools import *

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([-3, 0, 0], [0, 0, 10], green, ls='--', lw=2, label=r'$I_-(u)$')
u = np.linspace(-3, 0, num=100000, endpoint=False)

t_steps = [0.1, 0.5, 1, 5, 10]

for i, t in enumerate(t_steps):
    color = plt.get_cmap('hot')(i/len(t_steps))
    y = - (1 / t) * np.log(- u)
    ax.plot(u, y, color=color, lw=2, alpha=0.8,
            label=r'$- (1/t)\log(-u)$ ($t={}$)'.format(t))

ax.set_xlabel(r'$u$')
ax.set_ylim([-10, 10])
ax.set_xlim([-3, 1])
ax.legend(loc=2)
fig.savefig('Figures/log_bar.png')
