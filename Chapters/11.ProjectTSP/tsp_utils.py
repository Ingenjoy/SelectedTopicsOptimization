"""
Created on Wednesday 2 May 2018
Last update: Tuesday 8 May 2018

@author: Michiel Stock
michielfmstock@gmail.com

Utils used for the TSP assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import pairwise_distances

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

coordinates = np.load('Data/coordinates.npy')
distances = pairwise_distances(coordinates)

n, _ = distances.shape
cities = set(range(n))
tour = list(cities)

def plot_cities(ax, coordinates, color=blue):
    """
    Plots the cities on a given axis.

    Inputs:
        - ax : the ax to plot on
        - coordinates : the coordinates of the cities
        - color : color of the cities (default blue)
    """
    ax.scatter(coordinates[:,0], coordinates[:,1], color=color, s=5, zorder=2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_tour(ax, tour, coordinates, distances, color=red, title=True):
    """
    Draws a tour on a given axis.

    Inputs:
        - ax : the ax to plot on
        - tour : a tour a list of indices
        - coordinates : the coordinates of the cities
        - distances : the matrix with distances
        - color : color of the tour (default red)
        - title : boolean, plot the title with cost?
    """
    for i, j in zip(tour[1:], tour[:-1]):
        xi, yi = coordinates[i,:]
        xj, yj = coordinates[j,:]
        ax.plot([xi, xj], [yi, yj], color=color, zorder=1)
    i, j = tour[-1], tour[0]
    xi, yi = coordinates[i,:]
    xj, yj = coordinates[j,:]
    ax.plot([xi, xj], [yi, yj], color=color, zorder=1)
    if title:
        ax.set_title('Tour of cost {:.2f}'.format(compute_tour_cost(tour,
                                                                distances)))

def compute_tour_cost(tour, distances, check=False):
    """
    Computes the total cost of a tour of the TSP. Optionally provides a
    sanity check to see of the tour is a correct solution.

    Input:
        - tour : list of n integers from 1 to n describing the order of the tour
                    (invariant under cyclic permunations)
        - distances : n x n distance matrix
        - check : boolean, check if a valid tour, default False

    Ouput:
        - tour_cost : cost of the tour
    """
    if check:
        assert len(tour)==n, 'Tour does not contain all cities'
        assert set(tour)==cities, 'Tour contains duplicate cites'
    tour_cost = np.sum(distances[tour[:-1], tour[1:]])
    tour_cost += distances[tour[-1], tour[0]]
    return tour_cost

def save_tour(fname, tour):
    """
    Save a tour to a JSON file.

    Inputs:
        - fname : filename
        - tour : tour (as a list)
    """
    with open(fname, 'w') as fp:
        json.dump(tour, fp)


def load_tour(fname):
    """
    Reads a tour from a JSON file.

    Input:
        - fname : filename

    Output:
        - tour : loaded tour
    """
    with open(fname, 'r') as fp:
        return json.load(fp)


if __name__ == '__main__':

    fig, ax = plt.subplots()
    plot_cities(ax, coordinates, color=blue)
    plot_tour(ax, tour, coordinates, distances, color=red)
    fig.tight_layout()
    fig.savefig('Figures/tsp_example.png')
