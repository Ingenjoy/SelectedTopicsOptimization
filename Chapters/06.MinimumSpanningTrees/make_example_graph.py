"""
Created on Wednesday 7 March 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Make example graph for shortest path and MST
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

n_points = 50
r_big = 2
r_small = 0.5
neighbors = 4

coordinates = []
while len(coordinates) < n_points:
    x = (np.random.rand(2) - 0.5) * r_big * 4
    r = np.sum(x**2)**0.5
    if r < r_big and r > r_small:
        coordinates.append(x.tolist())

for x1, x2 in coordinates:
    plt.scatter(x1, x2, color=green, s=30, zorder=2)

balltree = BallTree(coordinates)
d, ind = balltree.query(coordinates, neighbors+1)
d = d[:,1:]  # first neighbor is itself
ind = ind[:,1:]  # first neighbor is itself

edge_list = set([])
for i, (dists, neighbors) in enumerate(zip(d, ind)):
    edge_list.update([(d, i, j) for d, j in zip(dists.tolist(), neighbors.tolist())])

for d, i, j in edge_list:
    xi1, xi2 = coordinates[i]
    xj1, xj2 = coordinates[j]
    plt.plot([xi1, xj1], [xi2, xj2], color='grey', alpha=0.7, lw=2, zorder=1)

plt.xticks([])
plt.yticks([])
plt.savefig('Figures/example_graph.png')

data = {
    'coordinates' : coordinates,
    'edges' : list(edge_list),
}

with open('Data/example_graph.json', 'w') as fh:
    json.dump(data, fh)
