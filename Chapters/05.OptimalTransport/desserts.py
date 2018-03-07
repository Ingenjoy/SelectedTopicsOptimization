"""
Created on Sunday 22 October 2017
Last update: Wednesday 07 March 2018

@author: Michiel Stock
michielfmstock@gmail.com

Create the dessert distribution example
"""

import pandas as pd


# preferences
preferences = pd.DataFrame([
    [2, 2, 1, 0, 0],
    [0, -2, -2, -2, 2],
    [1, 2, 2, 2, -1],
    [2, 1, 0, 1, -1],
    [0.5, 2, 2, 1, 0],
    [0, 1, 1, 1, -1],
    [-2, 2, 2, 1, 1],
    [2, 1, 2, 1, -1]
], index=['Bernard', 'Jan', 'Willem', 'Hilde', 'Steffie', 'Marlies', 'Tim', 'Wouter'])

preferences.columns = ['merveilleux', 'eclair', 'chocolate mousse', 'bavarois', 'carrot cake']

cost = -preferences

C = - preferences.values

# prortions per person
portions_per_person = pd.DataFrame([[3],
                                    [3],
                                    [3],
                                    [4],
                                    [2],
                                    [2],
                                    [2],
                                    [1]],
                    index=['Bernard', 'Jan', 'Willem', 'Hilde', 'Steffie',
                                            'Marlies', 'Tim', 'Wouter'])
portions_per_person /= portions_per_person.sum()

# quantities
quantities_of_dessert = pd.DataFrame([  [4],
                                        [2],
                                        [6],
                                        [4],
                                        [4]],
                                        index=['merveilleux', 'eclair', 'chocolate mousse',
                                                   'bavarois', 'carrot cake'])

quantities_of_dessert /= quantities_of_dessert.sum()

if __name__ == '__main__':

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style('white')
    from optimal_transport import red, green, yellow, orange, blue, black
    from optimal_transport import compute_optimal_transport

    def plot_partition(P, ax, lam, d):
        partition = pd.DataFrame(P, index=preferences.index, columns=preferences.columns)
        partition.plot(kind='bar', stacked=True, ax=ax)
        ax.set_ylabel('portions')
        ax.set_title('Optimal distribution ($\lambda={}$)'.format(lam))
        print('Average preference for lambda={}: {}'.format(lam, -d))

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(7,5))
    portions_per_person.plot(kind='bar', ax=ax0, color=green)
    ax0.set_ylabel('Relative portions')
    ax0.set_title('Fraction dessert per person')

    quantities_of_dessert.plot(kind='bar', ax=ax1, color=orange)
    ax1.set_ylabel('Relative portions')
    ax1.set_title('Fraction of each dessert')

    fig.tight_layout()
    fig.savefig('Figures/dessert_distributions.png')

    fig, ax = plt.subplots()

    sns.heatmap(cost)
    ax.set_ylabel('Persons')
    ax.set_xlabel('Desserts')
    fig.savefig('Figures/dessert_cost.png')

    a = portions_per_person.values.ravel()
    b = quantities_of_dessert.values.ravel()

    # low lambda
    P_1, d = compute_optimal_transport(C, a, b, 1)
    fig, ax = plt.subplots()

    plot_partition(P_1, ax, 1, d)
    fig.tight_layout()
    fig.savefig('Figures/desserts_low_lamda.png')

    # high lambda
    P_1, d = compute_optimal_transport(C, a, b, 100)
    fig, ax = plt.subplots()

    plot_partition(P_1, ax, 100, d)
    fig.tight_layout()
    fig.savefig('Figures/desserts_high_lamda.png')
