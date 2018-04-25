"""
Created on Wednesday 7 March 2018
Last update: Wednesday 25 April 2018

@author: Michiel Stock
michielfmstock@gmail.com

Make a city for the project of discrete optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import json

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

np.random.seed(2)
xmax = 250
ymax = 150
neighbors = 5

n_uniform = 1000
n_norm = 750

# make coordinates of the vertices
coordinates = []
# uniform spread
coordinates += (np.random.rand(n_uniform,2) * np.array([[xmax, ymax]])).tolist()

# MVN in the centre
coor_norm = []
mu = np.array([xmax, ymax]) / 2
cov = np.diag([1000, 500])
while len(coordinates) < n_uniform + n_norm:
    coor = np.random.multivariate_normal(mu, cov)
    if coor[0] > 0 and coor[0] < xmax:
        if coor[1] > 0 and coor[1] < ymax:
            coordinates.append(coor.tolist())

# make connections
balltree = BallTree(coordinates)
d, ind = balltree.query(coordinates, neighbors+1)
d = d[:,1:]  # first neighbor is itself
ind = ind[:,1:]  # first neighbor is itself

edges = set([])
for i, (dists, neighbors) in enumerate(zip(d, ind)):
    edges.update([(d, i, j) for d, j in zip(dists.tolist(), neighbors.tolist())])

# duplicate edges
edges.update([(d, j, i) for d, i, j in edges])

# make parks
park_A = set([i for i, (x, y) in enumerate(coordinates) if x < 50 and y > 75])
park_B = set([i for i, (x, y) in enumerate(coordinates)
                    if x > 190 and x < 210])
park_C = set(balltree.query_radius([[xmax/2, ymax/2]], 10)[0].tolist())

parks = park_A | park_B | park_C

fig, ax = plt.subplots(figsize=(20, 15))

for id, (x, y) in enumerate(coordinates):
    if id in parks:
        ax.scatter(x, y, color=green, s=20, zorder=2)
    else:
        ax.scatter(x, y, color=orange, s=20, zorder=2)

# add parks to plot
park_A_plot = plt.Rectangle((0, ymax - 75), 50, 75, alpha=0.3,
                                    color=green)
park_B_plot = plt.Rectangle((190, 0), 210 - 190, ymax, alpha=0.3,
                                    color=green)
park_C_plot = plt.Circle((xmax/2, ymax/2), 10, color=green, alpha=0.3)

ax.add_artist(park_A_plot)
ax.add_artist(park_B_plot)
ax.add_artist(park_C_plot)

# plot edges
for d, i, j in edges:
    xi, yi = coordinates[i]
    xj, yj = coordinates[j]
    ax.plot([xi, xj], [yi, yj], color='grey', alpha=0.7, lw=2, zorder=1)

# plot park letters
ax.text(25, 100, 'A', fontsize=42, color=red)
ax.text(200, 75, 'B', fontsize=42, color=red)
ax.text(xmax / 2, ymax / 2, 'C', fontsize=42, color=red)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Map of the city')
fig.patch.set_visible(False)
fig.savefig('Figures/city_map.png')

data = {
    'coordinates' : coordinates,
    'edges' : list(edges),
    'vertices' : list(range(n_uniform + n_norm)),
    'parks' : {
        'A' : list(park_A),
        'B' : list(park_B),
        'C' : list(park_C)
    }
}

with open('Data/city.json', 'w') as fh:
    json.dump(data, fh)
