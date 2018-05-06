"""
Created on Wednesday 2 May 2018
Last update: Friday 4 May 2018

@author: Michiel Stock
michielfmstock@gmail.com

Some algorithms for working with TSA data.

Implementation of
    - nearest neighbor
    - k-opt
    - simulated annealing
"""

import random as rd
import seaborn as sns
sns.set_style('white')
import numpy as np
from sys import path
path.append('../')
from tsp_utils import *
from itertools import combinations

def nearest_neighbor(start, distances):
    """
    Uses nearest neigbor to find a short tour.
    """
    tour = [start]
    visited = set(tour)
    n, _ = distances.shape
    tot_cost = 0
    while len(tour) < n:
        c, next = min([(distances[tour[-1],i], i) for i in range(n) if i not in visited])
        tot_cost += c
        visited.add(next)
        tour.append(next)
    tot_cost += distances[tour[-1], tour[0]]
    return tour, tot_cost

def make_k_opt(tour, distances, verbose=False):
    n = len(tour)
    local = False
    current_cost = compute_tour_cost(tour, distances)
    while not local:
        local = True
        for i in range(0, n-2):
            for j in range(i+2, n):
                new_tour = tour[:i] + list(reversed(tour[i:j])) + tour[j:]
                new_cost = compute_tour_cost(new_tour, distances)
                if new_cost < current_cost:
                    if verbose:
                        print('cost : {:.2f}'.format(new_cost))
                    tour = new_tour
                    current_cost = new_cost
                    local = False
    return tour

def greedy(distances): #DOES NOT WORK
    n, _ = distances.shape
    # make weighted edges
    edges = [(distances[i,j], i, j) for i in range(n-1) for j in range(i+1, n)]
    edges.sort()

    cities_in_edges = {i : 0 for i in range(n)}
    total_cost = 0
    edges_tour = []
    for w, i, j in edges:
        if len(edges_tour)==n: break
        if cities_in_edges[i] < 2 and cities_in_edges[j] < 2:
            total_cost += w
            cities_in_edges[i] += 1
            cities_in_edges[j] += 1
            edges_tour.append((i, j))
    path = {i : set([])}
    tour_as_list = [0]
    while tour:
        if tour_as_list[-1] in tour:
            tour_as_list.append(tour.pop(tour_as_list[-1]))
        else:
            tour_as_list.reverse()
    return tour_as_list

def simulated_annealing_tsa(distances, Tmax, Tmin, r, kT, start=None,
                            verbose=False):
    n, _ = distances.shape
    if start:
        tour = start
    else:
        tour = list(range(n))
        rd.shuffle(tour)
    T = Tmax + 0.0
    # save costs and tours
    current_cost = compute_tour_cost(tour, distances)
    costs = [current_cost]
    tours = [tour]
    while T >= Tmin:
        for rep in range(kT):
            # make new tour
            i, j = sorted(rd.sample(tour, 2))
            new_tour = tour[:i] + list(reversed(tour[i:j])) + tour[j:]
            #new_tour = [city for city in tour]
            #new_tour[i] = tour[j]
            #new_tour[j] = tour[i]
            new_cost = compute_tour_cost(new_tour, distances)
            i, j = rd.sample(tour, 2)
            if current_cost > new_cost or\
                        rd.random() < np.exp((current_cost - new_cost) / T):
                tour = new_tour
                current_cost = new_cost
        if verbose: print('T = {}, total cost = {:.2f}'.format(T, current_cost))
        costs.append(current_cost)
        tours.append(tour)
        T *= r
    return tours[-1], costs[-1], tours, costs



if __name__ == '__main__':
    n, _ = distances.shape

    """
    Generate random tour and
    """
    print('RANDOM TOURS')
    print('============')

    tour_rand = list(range(n))
    rd.shuffle(tour_rand)

    tour_rand_2_opt = make_k_opt(tour_rand, distances, True)

    print('-'*50)

    print('NEAREST NEIGHBOR')
    print('================')
    # find best tour using nn, iterate over all starting points
    nn_solutions = [nearest_neighbor(start, distances) for start in range(n)]
    tour_nn, tot_cost_nn = min(nn_solutions, key=lambda x:x[1])

    tour_nn_2_opt = make_k_opt(tour_nn, distances, True)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    for ax, color, tour in zip(np.ravel(axes), [green, yellow, orange, red],
                            [tour_rand, tour_rand_2_opt, tour_nn, tour_nn_2_opt]):
        plot_cities(ax, coordinates, color=blue)
        plot_tour(ax, tour, coordinates, distances, color=color)
    fig.tight_layout()
    fig.savefig('solution/random_nn_sol.pdf')

    print('-'*50)

    print('SIMULATED ANNEALING')
    print('===================')

    tour_sa, tot_cost_sa, tours_sa, costs_sa = simulated_annealing_tsa(distances, 50000,
                                    0.01, 0.98, 10000, verbose=True)

    from matplotlib.animation import FuncAnimation

    def update_tsa_fig(t, coordinates, tours, costs, axes):
        ax1, ax2 = axes
        ax1.clear()
        plot_cities(ax1, coordinates, color=green)
        plot_tour(ax1, tours[t], coordinates, distances, color=orange)
        ax1.set_yticks([])
        ax1.set_xticks([])
        ax2.clear()
        ax2.plot(costs, zorder=1, color=blue)
        ax2.scatter(t, costs[t], color=red, zorder=2, alpha=.9)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Total cost')

    def make_tsp_animation(coordinates, tours, costs, fig, axes):
        # plot edges graph
        n_steps = len(tours)
        # make animation
        anim = FuncAnimation(fig, lambda t : update_tsa_fig(t, coordinates, tours, costs, axes),
                        frames=range(n_steps), interval=100)
        return anim

    fig, axes = plt.subplots(ncols=2)

    anim = make_tsp_animation(coordinates, tours_sa, costs_sa, fig, axes)
    fig.tight_layout()
    anim.save('solution/tsp_sa.gif', dpi=80, writer='imagemagick')

    tour_sa_2_opt = make_k_opt(tour_sa, distances, verbose=True)
    fig, ax = plt.subplots()
    plot_cities(ax, coordinates, blue)
    plot_tour(ax, tour_sa_2_opt, coordinates, distances)
    fig.savefig('solution/sa_tour.pdf')
