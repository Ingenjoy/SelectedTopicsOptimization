# -*- coding: utf-8 -*-
"""
Created on Wed 6 Apr 2016
Last update on Thu 7 Apr 2016

@author: michielstock

Some routines to work with the parcours and mazes
for shortest path algorithms
"""

import numpy as np
from random import choice, randint
from shortestpaths import red

def links_to_graph(links):
    """
    Changes links to undirected graph
    """
    graph = {}  # change in dictionary graph
    for u, v in links:  # make undirected graph
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append((1, v))
        graph[v].append((1, u))
    return graph

def plot_parcour(links, ax, line_width=5):
    """
    Plots a maze or parcour on a given ax
    """
    max_x = 0
    max_y = 0
    for (i1, j1), (i2, j2) in links:
        ax.plot([i1, i2], [j1, j2], c='white', lw=line_width)
        if max(i1, i2) > max_x:
            max_x = max(i1, i2) + 0.5
        if max(j1, j2) > max_y:
            max_y = max(j1, j2) + 0.5
    ax.set_xlim([-0.5, max_x])
    ax.set_ylim([-0.5, max_y])
    ax.set_axis_bgcolor('black')

def add_path(path, ax, color=red):
    """
    Add a path to an ax
    """
    for i in range(len(path)-1):
        i1, j1 = path[i]
        i2, j2 = path[i+1]
        ax.plot([i1, i2], [j1, j2], c=color, lw=2)

def load_links(name):
    file_handler = open(name, 'r')
    links = set([])
    for line in file_handler:
        i1, j1, i2, j2 = map(int, line.rstrip().split(','))
        links.add(((i1, j1), (i2, j2)))
    return links
