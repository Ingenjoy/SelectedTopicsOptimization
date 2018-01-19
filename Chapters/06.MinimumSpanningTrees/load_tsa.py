"""
Created on Tue Mar 18 2017
Last update on -

@author: Michiel Stock

Load Travelling Salesman Problems
"""

import numpy as np

def make_graph_from_D(D):
    """
    Makes a graph from a distance matrix
    """
    n, _ = D.shape
    graph = {i : set([(D[i,j], j) for j in range(n) if j != i])
                                    for i in range(n)}
    return graph

# get coordinates
coordinates29 = np.genfromtxt('Data/cities_29.txt')[:,1:]
coordinates225 = np.genfromtxt('Data/cities_225.txt')[:,1:]

# get distances
distances29 = np.genfromtxt('Data/distances_29.txt')
distances225 = np.genfromtxt('Data/distances_225.txt')

# make graphs
graph29 = make_graph_from_D(distances29)
graph225 = make_graph_from_D(distances225)
