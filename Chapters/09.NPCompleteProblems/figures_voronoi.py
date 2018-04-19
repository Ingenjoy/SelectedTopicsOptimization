"""
Created on Thursday 19 April 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Makes a Voronoi diagram
"""

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

n_points = 35

points = np.random.rand(n_points, 2)
vor = Voronoi(points)
fig, ax = plt.subplots(figsize=(11, 8))
voronoi_plot_2d(vor, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig('Figures/voronoi.png')
