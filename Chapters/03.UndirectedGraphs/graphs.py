# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 2017
Last update on Wed Mar 29 2017

@author: michielstock

Some graph algorithms
"""

import json
from random import choice
from scipy.spatial import Voronoi

# SAVING GRAPHS
# -------------

def save_graph(file_name, adj_list):
    """
    Saves graph in a JSON file

    Inputs:
        - file_name: file name
        - adj_list: the adjacency list (dictionary)
    """
    json.dump({str(k) : list(map(str, v)) for k, v in adj_list.items()}, open(file_name, 'w'))

def load_graph(file_name):
    """
    Loads graph from a JSON file

    Input:
        - file_name: file name

    Output:
        - adj_list: the adjacency list (dictionary)
    """
    adj_list = json.load(open(file_name, 'r'))
    return {str(k) : set(map(str, v)) for k, v in adj_list.items()}

def edges_to_adj_list(edges):
    """
    Transforms a set of edges in an adjacency list (represented as a dictiornary)

    For UNDIRECTED graphs, i.e. if v2 in adj_list[v1], then v1 in adj_list[v2]

    INPUT:
        - edges : a set or list of edges

    OUTPUT:
        - adj_list: a dictionary with the vertices as keys, each with
                a set of adjacent vertices.
    """
    adj_list = {}  # store in dictionary
    for v1, v2 in edges:
        if v1 in adj_list:  # edge already in it
            adj_list[v1].add(v2)
        else:
            adj_list[v1] = set([v2])
        if v2 in adj_list:  # edge already in it
            adj_list[v2].add(v1)
        else:
            adj_list[v2] = set([v1])
    return adj_list

def adj_list_to_edges(adj_list):
    """
    Transforms an adjacency list (represented as a dictiornary) in a set of edges

    For UNDIRECTED graphs, i.e. if v2 in adj_list[v1], then v1 in adj_list[v2]

    INPUT:
        - adj_list: a dictionary with the vertices as keys, each with
                a set of adjacent vertices

    OUTPUT:

        - edges : a set or list of edges
    """
    edges = set([])
    for v, neighbors in adj_list.items():
        for n in neighbors:
            edges.add((n, v))
    return edges

# CONNECTIVITY
# ------------

def give_connected_component(adj_list, vertex):
    """
    Returns the connected component

    Inputs:
        - adj_list: adjacency list of a graph
        - vertex: a given vertex

    Ouput:
        - a set of all vertices reachable from the given vertex
    """
    vertices_visited = set([])
    to_visit = set([vertex])
    connected = False
    while len(to_visit) > 0:
        current_vertex = to_visit.pop()
        vertices_visited.add(current_vertex)
        neighbors = adj_list[current_vertex]
        to_visit |= (neighbors - vertices_visited)  # new vertices
    return vertices_visited

def is_connected(adj_list):
    """
    Check if a graph is connected

    Input:
        - adj_list: adjacency list of a graph

    Ouptut:
        - boolean denoting if the graph is connected
    """
    vertex = list(adj_list.keys()).pop()  # an edge
    connected_component = give_connected_component(adj_list, vertex)
    return len(connected_component) == len(adj_list)

def has_path(adj_list, vertex1, vertex2):
    """
    Check if there is a path between two vertices

    Inputs:
        - adj_list: adjacency list of a graph
        - vertex1: first vertex
        - vertex2: second vertex

    Outputs:
        - boolean indicating if there is a path between vertex1 and vertex2
    """
    vertices_visited = set([])
    to_visit = set([vertex1])
    connected = False
    while len(to_visit) > 0:
        current_vertex = to_visit.pop()
        if current_vertex == vertex2:
            return True
        vertices_visited.add(current_vertex)
        neighbors = adj_list[current_vertex]
        to_visit |= (neighbors - vertices_visited)  # new vertices
    return False

def is_bridge(adj_list, edge):
    """
    Checks if an edge is a bridge

    Inputs:
        - adj_list: adjacency list of a graph
        - edge: tuple of two adjacent vertices

    Outputs:
        - boolean indicating if the given edge is a bridge
    """
    v1, v2 = edge
    component_with_v1 = give_connected_component(adj_list, v1)
    # remove edge
    adj_list[v1].remove(v2)
    adj_list[v2].remove(v1)
    component_without_v1 = give_connected_component(adj_list, v1)
    # restore edge
    adj_list[v1].add(v2)
    adj_list[v2].add(v1)
    bridge = component_with_v1 != component_without_v1
    return bridge

# RANDOM GRAPHS
# -------------

def generate_random_graph(n_vertices, n_edges):
    vertices = list(range(n_vertices))
    edges = set([])
    current = choice(vertices)
    while len(edges) < n_edges:
        next = choice(vertices)
        if current != next and ((current, next) not in edges) and ((next, current) not in edges) :
            edges.add((current, next))
            current = next
    return edges

def get_planar_graph(X):
    vor = Voronoi(X)
    return edges_to_adj_list(vor.ridge_points)
