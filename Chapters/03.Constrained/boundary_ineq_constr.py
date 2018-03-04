"""
Created on Tuesday 27 February 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Shows the inner product of the gradient of the nonquadr function
with the gradient of the inequality constraint
"""

import numpy as np
from teachingtools import nonquadratic, grad_quadratic, green, orange
import matplotlib.pyplot as plt

x0 = np.array([[1], [0.25]])  # centre of circle
thetas = np.linspace(0, 2 * np.pi, num=100)  # angle

fvals = []
fg_grad_prods = []
for theta in thetas:
    grad_v = np.array([[np.cos(theta)], [np.sin(theta)]])
    x = grad_v + x0
    grad_f = grad_quadratic(x)
    fvals.append(nonquadratic(x))
    fg_grad_prods.append(np.sum(grad_v*grad_f))

fig, axes = plt.subplots(nrows=2, sharex=True)

axes[0].plot(thetas, fvals, color=green, lw=2)
axes[0].set_ylabel(r'$f(\mathbf{x}(\theta))$')
axes[1].plot(thetas, fg_grad_prods, color=orange, lw=2)
axes[1].set_ylabel(r'$\nabla f(\mathbf{x}(\theta))^\top \nabla g(\mathbf{x}(\theta))$')
axes[1].set_xlabel(r'$\theta$')

for ax in axes:
    ax.grid()

fig.savefig('Figures/boundary_ineq_fig.png')
