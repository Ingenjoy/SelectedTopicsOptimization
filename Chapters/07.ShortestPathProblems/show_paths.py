"""
Created on Sunday 25 March 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Illustration of Dijkstra and A* search
"""


from shortestpaths import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sys import path
path.append('../06.MinimumSpanningTrees')
from tickettoride import edges_by_distance, vertex_coordinates, edges

# distance graph ticket to ride
graph_ttr_dist = edges_to_adj_list(edges_by_distance)
# cost graph ticket to ride
graph_ttr_cost = edges_to_adj_list(edges)

# euclidean distance ttr
def euclidean_dist_ttr(c1, c2):
    """
    Computes Euclidean distance between two cities
    """
    x1, y1 = vertex_coordinates[c1]
    x2, y2 = vertex_coordinates[c2]
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def make_shortest_path_animation(coordinates, edges, order, fig, ax,
                                pointsize=100):
    # make animation of the shortest path algorithms
    for w, i, j in edges:
        xi1, xi2 = coordinates[i]
        xj1, xj2 = coordinates[j]
        ax.plot([xi1, xj1], [xi2, xj2], color='grey', alpha=0.7, lw=2, zorder=1)

    # plot points
    for city, (x1, x2) in coordinates.items():
        ax.scatter(x1, x2, color=green, s=pointsize, zorder=2)
        ax.text(x1, x2+0.5, city, fontsize=10, zorder=5, color=blue,
                        horizontalalignment='center')

    # color source (first) and sink (last)
    x1, x2 = coordinates[order[0]]
    ax.scatter(x1, x2, color=blue, s=pointsize, zorder=3)
    x1, x2 = coordinates[order[-1]]
    ax.scatter(x1, x2, color=yellow, s=pointsize, zorder=3)

    # make animation
    anim = FuncAnimation(fig, lambda t : ax.scatter(*coordinates[order[t-1]],
                                color=orange, s=pointsize, zorder=4) if t else None,
                                frames=range(len(order)+1), interval=500)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    return anim

def a_star_illustration(graph, source, sink, heuristic=lambda x, y:0):
    """
    Implementation of the A* shortest path algorithm.

    Inputs:
        - graph : dict representing the weighted graph
        - source : the source node
        - sink : the sink node (optional)
        - heuristic : a function with the heuristic for the shortest path
                    between two nodes

    Ouputs:
            - order : order of which the vertices are checked
    """
    distance = {v : inf for v in graph.keys()}
    # vertices_to_check is a heap containing the estimated distance
    # of a given node to a source
    vertices_to_check = [(heuristic(source, sink), source)]
    previous = {}

    distance[source] = 0
    order = []

    while vertices_to_check:
        heuristic_dist, current = heappop(vertices_to_check)
        order.append(current)
        if current == sink:
            return order
        for dist_current_neighbor, neighbor in graph[current]:
            new_dist_from_source = distance[current] + dist_current_neighbor
            if new_dist_from_source < distance[neighbor]:
                distance[neighbor] = new_dist_from_source
                min_dist_neighbor_source = distance[neighbor] +\
                        heuristic(neighbor, sink)
                heappush(vertices_to_check, (min_dist_neighbor_source, neighbor))
                previous[neighbor] = current


source = 'Helena'
sink = 'Pittsburgh'

order = a_star_illustration(graph_ttr_dist, source, sink, heuristic=lambda x, y:0)
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Dijkstra to find the shortest path from {} to {}'.format(source, sink))
anim = make_shortest_path_animation(vertex_coordinates, edges, order, fig, ax)
anim.save('Figures/dijkstra_search.gif', dpi=80, writer='imagemagick')

order = a_star_illustration(graph_ttr_dist, source, sink,
                heuristic=euclidean_dist_ttr)
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('A* to find the shortest path from {} to {}'.format(source, sink))
anim = make_shortest_path_animation(vertex_coordinates, edges, order, fig, ax)
anim.save('Figures/A_star_search.gif', dpi=80, writer='imagemagick')
