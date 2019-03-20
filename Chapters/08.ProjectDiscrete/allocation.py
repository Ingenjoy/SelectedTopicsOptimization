"""
Created on Tuesday 12 March 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Files for the project on allocation problem.
"""

import json
import numpy as np

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'



with open('Data/city.json', 'r') as fh:
    data = json.load(fh)

coordinates = data['coordinates']
edges = data['edges']
vertices = set(data['vertices'])
sources = set(data['sources'])
sinks = set(data['sinks'])

capacity = 10

adj_matrix = {v : {} for v in vertices}
for w, u, v in edges:
    adj_matrix[u][v] = w

def check_solution(solution):
    """
    Evaluates a solution and returns the total cost. Raises an error if the
    solution is not valid.
    """
    assert set(solution.keys()) == set(sources), "solution does not contain all sources OR contains additional nodes..."
    total_cost = 0.0
    free_capacities = {u : capacity for u in sinks}
    for start, path in solution.items():
        current = start
        path_cost = 0.0
        for next in path:
            if next in adj_matrix[current]:  # check if path is possible
                path_cost += adj_matrix[current][next]
                current = next
            else:
                raise KeyError("error crossing from {} to {} in path for node {}".format(current, next, start))
        assert current in sinks, "path for node {} does not end in sink!".format(start)
        assert free_capacities[current] > 0, "path for node {} does not end in sink with free capacity!".format(start)
        free_capacities[current] -= 1
        total_cost += path_cost
    print("TOTAL COST: {:.2f}".format(total_cost))
    return total_cost


def generate_drunken_solution():
    """
    Generate a valid, though poor solution to the allocation problem.

    This algorithm just performs a random walk for every source until it
    encounters a sink with free capacity.
    """
    solution = {}
    cost = 0.0
    free_capacities = {u : capacity for u in sinks}
    for (i, start) in enumerate(sources):
        current = start
        if (i + 1)%100==0:
            print("Finding solution for {} / {}".format(i+1, len(sources)))
        path = []
        while True:
            next = np.random.choice(list(adj_matrix[current].keys()))
            next = int(next)
            cost += adj_matrix[current][next]
            path.append(next)
            current = next
            if current in free_capacities and free_capacities[current] > 0:
                free_capacities[current] -= 1
                break
        solution[start] = path
    return solution, cost
