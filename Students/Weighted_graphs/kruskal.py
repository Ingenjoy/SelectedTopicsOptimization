# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:31:17 2016
Last update on Mon Apr 11 2016

@author: michielstock

Kruskal's algorithm for finding the maximum spanning tree
"""

from union_set_forest import USF

def kruskal(vertices, edges):
    """
    Kruskal's algorithm for finding a minimum spanning tree
    Input :
        - vertices : a set of the vertices of the Graph
        - edges : a list of weighted edges (e.g. (0.7, 'A', 'B')) for an
                    edge from node A to node B with weigth 0.7
    Output:
        a minumum spanning tree represented as a list of edges
    """
    union_set_forest = USF(vertices)
    edges = list(edges)  # might be saved in set format...
    edges.sort()
    forest = set([])
    for cost, u, v in edges:
        if union_set_forest.find(u) != union_set_forest.find(v):
            forest.add((u, v))
            union_set_forest.union(u, v)
    del union_set_forest
    return forest

if __name__ == '__main__':
    words = ['maan', 'laan', 'baan', 'mama', 'saai', 'zaai', 'naai', 'baai',
             'loon', 'boon', 'hoon', 'poon', 'leem', 'neem', 'peen', 'tton',
             'haar', 'haar', 'hoor', 'boor', 'hoer', 'boer', 'loer', 'poer']

    hamming = lambda w1, w2 : sum([ch1 != ch2 for ch1, ch2 in zip(w1, w2)])

    edges = [(hamming(w1, w2), w1, w2) for w1 in words
                for w2 in words if w1 is not w2]

    tree = kruskal(words, edges)
    print(tree)

    import networkx
    g = networkx.Graph()

    g.add_edges_from(tree)
    labels = {n:n for n in g.nodes()}
    networkx.draw(g, networkx.spring_layout(g))

    # draw maze

    import numpy as np

    size = 50

    M = np.random.randn(size, size)
    vertices = [(i, j) for i in range(size) for j in range(size)]
    edges = [(abs(M[i1, j1] - M[i2, j2]), (i1, j1), (i2, j2)) for i1,
             j1 in vertices for i2, j2 in vertices if abs(i1-i2) +
                abs(j1-j2) == 1  if (i1, j1) != (i2, j2)]

    maze_links = kruskal(vertices, edges)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_bgcolor('black')

    for (i1, j1), (i2, j2) in maze_links:
        ax.plot([i1, i2], [j1, j2], c='white', lw=5)
    fig.savefig('maze.pdf')
