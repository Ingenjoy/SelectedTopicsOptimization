"""
Created on Tuesday 12 March 2019
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

n_sources = 1000
n_sinks = 100
n = n_sources + n_sinks

# make coordinates of the vertices
coordinates = []
# uniform spread
coordinates += (np.random.rand(n_sources,2) * np.array([[xmax, ymax]])).tolist()

# add sinks
# will add them in a concentric circle

while len(coordinates) < n_sources + n_sinks:
    x = np.random.rand() * xmax
    y = np.random.rand() * ymax
    r = ((x - xmax/2)**2 + (y - ymax/2)**2)**0.5
    if r > 50 and r < 100:
        coordinates.append([x, y])

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

sources = set(range(n_sources))
sinks = set(range(n_sources, n))

fig, ax = plt.subplots(figsize=(20, 15))

for id, (x, y) in enumerate(coordinates):
    if id in sinks:
        ax.scatter(x, y, color=red, s=100, zorder=2)
    else:
        ax.scatter(x, y, color=green, s=20, zorder=2)

ax.scatter([], [], color=red, s=100, label="sinks")
ax.scatter([], [], color=green, s=20, label="sources")
# plot edges
for d, i, j in edges:
    xi, yi = coordinates[i]
    xj, yj = coordinates[j]
    ax.plot([xi, xj], [yi, yj], color='grey', alpha=0.7, lw=2, zorder=1)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Map of the city')
ax.legend(loc=0)
fig.patch.set_visible(False)
fig.savefig('Figures/city_map.png')

data = {
    'coordinates' : coordinates,
    'edges' : list(edges),
    'vertices' : list(range(n)),
    'sinks' : list(sinks),
    'sources' : list(sources)
}

with open('Data/city.json', 'w') as fh:
    json.dump(data, fh)
