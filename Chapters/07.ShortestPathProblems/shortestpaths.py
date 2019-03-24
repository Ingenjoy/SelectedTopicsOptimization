"""
Created on Tuesday 20 March 2018
Last update: Thursday 21 March 2019

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of Dijksta and A* to find the shortest path.
"""

from heapq import heappop, heappush
from numpy import inf
from numpy import sum as npsum

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

manhattan_distance = lambda x, y : sum([abs(xi - yi) for xi, yi in zip(x, y)])
euclidean_distance = lambda x, y : sum([(xi - yi)**2 for xi, yi in zip(x, y)])**0.5

def dijkstra(graph, source, sink=None):
    """
    Implementation of Dijkstra's shortest path algorithm.

    Inputs:
        - graph : dict representing the weighted graph
        - source : the source node
        - sink : the sink node (optional)

    Ouput:
            - distance : dict with the distances of the nodes to the source
            - previous : dict with for each node the previous node in the
                        shortest path from the source (if one is given)
    """
    # keep tentative distance source to vertex
    # initialize with infinity, except for the source
    distance = {v : inf for v in graph.keys()}
    distance[source] = 0
    # keep previous node in path for backtracking
    previous = {}
    # heap for vertices to check
    # priority based distance from source
    vertices_to_check = [(0, source)]

    while vertices_to_check:
        # pop vertex to explore
        dist, u = heappop(vertices_to_check)
        if u == sink:  # sink reached!
            break
        for dist_u_v, v in graph[u]:
            new_dist = dist + dist_u_v
            if new_dist < distance[v]:
                distance[v] = new_dist
                previous[v] = u
                heappush(vertices_to_check, (new_dist, v))
    if sink is None:
        return distance, previous
    else:
        return reconstruct_path(previous, source, sink), distance[sink]


def reconstruct_path(previous, source, sink):
    """
    Reconstruct the path from the output of the Dijkstra algorithm.

    Inputs:
            - previous : a dict with the previous node in the path
            - source : the source node
            - sink : the sink node
    Ouput:
            - the shortest path from source to sink
    """
    if sink not in previous:
        return []
    v = sink
    path = [v]
    while v is not source:
        v = previous[v]
        path = [v] + path
    return path

def a_star(graph, source, sink, heuristic):
    """
    Implementation of the A* shortest path algorithm.

    Inputs:
        - graph : dict representing the weighted graph
        - source : the source node
        - sink : the sink node (optional)
        - heuristic : a function with the heuristic for the shortest path
                    between two nodes

    Ouputs:
            - distance : dict with the distances of the nodes to the source
            - previous : dict with for each node the previous node in the
                        shortest path from the source
    """
    # keep tentative distance source to vertex
    # initialize with infinity, except for the source
    distance = {v : inf for v in graph.keys()}
    distance[source] = 0
    # keep previous node in path for backtracking
    previous = {}
    # vertices_to_check is a heap using the estimated distance
    # of a given node to a source as the priority
    vertices_to_check = [(heuristic(source, sink), source)]
    previous = {}

    while vertices_to_check:
        heuristic_dist, current = heappop(vertices_to_check)
        if current == sink:
            return reconstruct_path(previous, source, sink), distance[sink]
        for dist_current_neighbor, neighbor in graph[current]:
            new_dist_from_source = distance[current] + dist_current_neighbor
            if new_dist_from_source < distance[neighbor]:
                distance[neighbor] = new_dist_from_source
                min_dist_neighbor_source = distance[neighbor] +\
                        heuristic(neighbor, sink)
                heappush(vertices_to_check, (min_dist_neighbor_source, neighbor))
                previous[neighbor] = current

def bellman_ford(graph, source):
    """
    Implementation of Bellman-Ford algorithm. Will print if cycles are
    detected!

    Mainly for didactic purposes.

    Inputs:
        - graph : dict representing the weighted graph
        - source : the source node

    Ouputs:
            - distance : dict with the distances of the nodes to the source
    """
    # tentative distance
    distance = {v : inf for v in graph.keys()}
    distance[source] = 0
    # relaxation
    for _ in range(len(graph) - 1):
        # cycle over edges
        for v, neighbors in graph.items():
            for w, n in neighbors:
                # relaxation step
                distance[n] = min(distance[n], distance[v] + w)
    # detect cycles
    for v, neighbors in graph.items():
        for w, n in neighbors:
            if distance[v] + w < distance[n]:
                print("Cycle found via ", n)
    return distance

def edges_to_adj_list(edges):
    """
    Turns a list of edges in an adjecency list (implemented as a list).
    Edges don't have to be doubled, will automatically be symmetric

    Input:
        - edges : a list of weighted edges (e.g. (0.7, 'A', 'B') for an
                    edge from node A to node B with weigth 0.7)

    Output:
        - adj_list : a dict of a set of weighted edges
    """
    adj_list = {}
    for w, i, j in edges:
        for v in (i, j):
            if v not in adj_list:
                adj_list[v] = set([])
        adj_list[i].add((w, j))
        adj_list[j].add((w, i))
    return adj_list

if __name__ == '__main__':

    # TEST DIJKSTRA
    graph = {'A' : [(0.5, 'B'), (1.5, 'H')],
        'B' : [(1, 'C')],
        'C' : [(1, 'D'), (1, 'E')],
        'D' : [(2, 'F')],
        'E' : [(1, 'G')],
        'F' : [],
        'G' : [(1, 'F')],
        'H' : [((1, 'E'))]}

    distance, came_from = dijkstra(graph, 'A')
    print(reconstruct_path(came_from, 'A', 'E'))

    # TEST A*

    import numpy as np
    from random import choice

    nodes = [(np.random.randn(), np.random.randn()) for i in range(1000)]
    graph = {node : set([]) for node in nodes}
    for node1 in nodes:
        for i in range(50):
            node2 = choice(nodes)
            if node1 != node2:
                dist = euclidean_distance(node1, node2)
                graph[node1].add((dist, node2))

    print(a_star(graph, nodes[1], nodes[10], euclidean_distance))


    graph_neg_cycle = {
        'A' : [(1, 'C'), (3, 'E')],
        'B' : [(-1, 'D')],
        'C' : [(-2, 'D'), (2, 'B')],
        'D' : [(0.5, 'A'), (-1, 'E'), (2, 'F')],
        'E' : [],
        'F' : [(1, 'E'), (3, 'B')]
    }
