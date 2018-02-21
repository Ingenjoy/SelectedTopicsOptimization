"""
Created on Monday 5 February 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Figure steepest descent
"""

from teachingtools import plt, show_steepest_descent_gradients, np

# plot contours
fig, ax = plt.subplots()
ax.set_aspect('equal')
x = np.array([0.7, -0.5])
#x = np.array([-1.5, 0.5])  #  other point
show_steepest_descent_gradients(x=x, ax=ax)
fig.tight_layout()
fig.savefig('Figures/sd_gradients.png')
