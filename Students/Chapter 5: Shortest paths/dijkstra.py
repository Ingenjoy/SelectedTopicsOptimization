# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:16:48 2016
Last update on Fri Mar 11 2016

@author: michielstock

Dijkstra's shortest path
"""

from heapq import heappop, heappush
from numpy import inf


def dijkstra(graph, source, sink=None):
    """
    Implementation of Dijkstra's shortest path algorithm
    Inputs:
        - graph : dict representing the weighted graph
        - source : the source node
        - sink : the sink node (optional)
    Ouput:
            - distance : dict with the distances of the nodes to the source
            - came_from : dict with for each node the came_from node in the shortest
                    path from the source
    """
    distance = {V : inf for V in graph.keys()}
    came_from = {}
    distance[source] = 0
    vertexes_to_check = [(0, source)]

    while len(vertexes_to_check) > 0:
        dist, U = heappop(vertexes_to_check)
        if U == sink:
            break
        for dist_U_V, V in graph[U]:
            new_dist = dist + dist_U_V
            if new_dist < distance[V]:
                distance[V] = new_dist
                came_from[V] = U
                heappush(vertexes_to_check, (new_dist, V))
    if sink is None:
        return distance, came_from
    else:
        return reconstruct_path(came_from, source, sink), distance[sink]


def reconstruct_path(came_from, source, sink):
    """
    Reconstruct the path from the output of the Dijkstra algorithm
    Inputs:
            - came_from : a dict with the came_from node in the path
            - source : the source node
            - sink : the sink node
    Ouput:
            - the shortest path from source to sink
    """
    if not came_from.has_key(sink):
        return []
    V = sink
    path = [V]
    while V is not source:
        V = came_from[V]
        path = [V] + path
    return path


if __name__ == '__main__':


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