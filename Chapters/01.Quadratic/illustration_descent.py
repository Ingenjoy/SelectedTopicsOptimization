"""
Created on Tuesday 26 December 2017
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Illustration of the descent step
"""

import sys
sys.path.append('helpers/')
from colors import red, blue, green, black
from plotting import plot_contour
import matplotlib.pyplot as plt
import numpy as np

format = 'png'

P = np.array([[4, 1], [1, 2]])
x = np.array([[1], [1]])

f = lambda x : np.sum(np.array(x) @ P @ np.array(x))

dfx = 0.5 * P @ x

# make contours
fig, ax = plt.subplots()
plot_contour(f, [0, 2], [0, 2], ax, False)

ax.arrow(x[0,0], x[1,0], dfx[0,0]*0.2, dfx[1,0]*0.2, color=red)
ax.text(x[0,0]+dfx[0,0]*0.2+0.05, x[1,0]+dfx[1,0]*0.2+0.05,
                    r'$\nabla f(\mathbf{x}^{(k)})$', color=red)

ax.text(x[0,0]+0.1, x[1,0]+0.1, r'$\mathbf{x}^{(k)}$',
                             color=black)

def compute_x2_line(x1):  # function to compute the line in x, orth to dfx
    intercept = np.sum(dfx * x)
    return (intercept - x1 * dfx[0,0]) / dfx[1,0]

ax.plot([0, 2], compute_x2_line(np.array([0,2])), color=green, ls='--', linewidth=2)
ax.text(1, 1.5, r'ascent ($(\Delta \mathbf{x}^{(k)})^\top \nabla f(\mathbf{x}) > 0$)',
                                    color=green)
ax.text(0.1, 1., r'descent ($(\Delta \mathbf{x}^{(k)})^\top \nabla f(\mathbf{x}) < 0$)',
                                    color=green)

ax.arrow(x[0,0], x[1,0], -dfx[0,0]*0.2, -dfx[1,0]*0.2, color=blue)
ax.text(x[0,0]-dfx[0,0]*0.2-0.1, x[1,0]-dfx[1,0]*0.2-0.1,
                    r'$\Delta \mathbf{x}^{(k)}$', color=blue)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_ylabel('$x_2$')
ax.set_xlabel('$x_1$')
ax.set_aspect('equal', 'box')
ax.set_ylim([0,2])
ax.set_xlim([0,2])
fig.tight_layout()

fig.savefig('Chapters/01.Quadratic/Figures/descent_step.{}'.format(format))

plt.close('all')
