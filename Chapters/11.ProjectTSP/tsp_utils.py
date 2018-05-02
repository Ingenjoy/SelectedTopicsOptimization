"""
Created on Wednesday 2 May 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Utils used for the TSP assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

coordinates = np.load('Data/coordinates.npy')
distances = np.load('Data/distances.npy')

n, _ = distances.shape
cities = set(range(n))
tour = list(range(n))

def plot_cities(ax, color=blue):
    """
    Plots the cities on a given axis.
    """
    ax.scatter(coordinates[:,0], coordinates[:,1], color=color, s=5, zorder=2)

def plot_tour(ax, tour, color=red, title=True):
    """
    Adds a tour on a given axis.
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
        ax.set_title('Tour of length {}'.format(compute_tour_length(tour)))

def compute_tour_length(tour):
    """
    Computes the total length of a tour of the TSP.

    Input:
        - tour : list of integers from 1 to n describing the order of the tour
                    (invariant under cyclic permunations)

    Ouput:
        - tour_length : length of the tour
    """
    assert len(tour)==n, 'Tour does not contain all cities'
    assert set(tour)==cities
    tour_length = np.sum(distances[tour[:-1], tour[1:]])
    tour_length += distances[tour[-1], tour[0]]
    return tour_length

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
        - tour
    """
    with open(fname, 'r') as fp:
        return json.load(fp)



if __name__ == '__main__':

    fig, ax = plt.subplots()
    plot_cities(ax, color=blue)
    plot_tour(ax, tour, color=red)
    fig.savefig('Figures/tsp_example.png')
