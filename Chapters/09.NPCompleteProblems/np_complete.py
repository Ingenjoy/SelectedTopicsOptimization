"""
Created on Tuesday 17 April 2018
Last update: Thursday 19 April 2018

@author: Michiel Stock
michielfmstock@gmail.com

Some functions related to the NP-complete problems.
"""

import random as rd
import numpy as np

# Indiana Jones Knapsack example
# save data in a list of (name, value, weight) tuples
artifacts = [('statue 1' , 1, 2), ('statue 2', 1, 2), ('statue 3', 1, 2),
            ('tablet 1', 10, 5), ('tablet 2', 10, 5), ('golden mask', 13, 7),
            ('golden plate', 7, 3)]

capacity = 10  # kg

def generate_dynamic_programming_knapsack(items, capacity, return_dp=False):
    """
    Generates a greedy solution for the knapsack problem,
    items are gathered greedy according to the heuristic

    Inputs:
        - items : a list of tuples (item, value, weight)
        - capacity : the capacity of the knapsack

    Output:
        - solution : a tuple (value, weight, chosen set)
                containing the optimal solution
    """
    assert type(capacity) is int
    dyn_prog_matrix = np.zeros((capacity + 1, len(items) + 1))
    # fill dynamic programming matrix
    n_it = 1
    for (item, value, weight) in items:
        for n_cap in range(1, capacity + 1):
            if weight <= n_cap:  # if there is room for the item
                dyn_prog_matrix[n_cap, n_it] = max(dyn_prog_matrix[n_cap, n_it - 1],
                                                  dyn_prog_matrix[n_cap - weight, n_it - 1] + value)
            else:
                dyn_prog_matrix[n_cap, n_it] = dyn_prog_matrix[n_cap, n_it - 1]
        n_it += 1
    if return_dp:
        return dyn_prog_matrix
    # trace back
    n_it = len(items)
    value_set = dyn_prog_matrix[-1, -1]
    weight_set = 0
    chosen_set = []
    while n_it > 0:
        if dyn_prog_matrix[n_cap, n_it] == dyn_prog_matrix[n_cap, n_it - 1]:
            # this item is not chosen
            n_it -= 1
        else: # item was chosen
            item, value, weight = items[n_it - 1]
            weight_set += weight
            chosen_set.append(item)
            n_cap -= weight
            n_it -= 1
    return (value_set, weight_set, chosen_set)

def generate_set_cover_instance(n, m, n_choice=False):
    """
    Generates an instance of the set cover problem.

    Inputs:
        - n : size of the universe
        - m : number of sets
        - n_choice (default False): either an int containing the number of
                samples (with replacement) from the universe or list of ints
                with the size of every set S_k

    Output:
        - S : list of sets
        - U : the universe (could be smaller than n)
    """
    U = list(range(1,n+1))
    if not n_choice: n_choice = n
    if type(n_choice)==int:
        n_choice = [n_choice] * m
    assert len(n_choice) == m
    S = [set([rd.choice(U) for _ in range(p)]) for p in n_choice]
    U = set().union(*S)
    return S, U
