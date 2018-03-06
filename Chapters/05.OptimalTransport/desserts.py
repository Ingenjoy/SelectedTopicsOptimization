"""
Created on Sunday 22 October 2017
Last update: Tuesday 06 March 2018

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

M = - preferences.values

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
