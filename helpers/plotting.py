"""
Created on Thursday 22 August 2017
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Some useful functions for plotting and visualisation.

Contains:
    - colors: the official colors!
    - countour plots: for the convex optimization part
"""

import matplotlib.pyplot as plt
import numpy as np

# COLORS
# ------

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

colors = {'blue' : blue,
        'green' : green,
        'yellow' : yellow,
        'orange' : orange,
        'red' : red,
        'black' : black}


# COUNTOUR PLOTS
# --------------

def plot_contour(f, xlim, ylim, ax, plot_f=True):
    '''
    Plots the contour of a 2D function to be minimized

    Inputs:
        - f: function
        - xlim, ylim: limits of the plot
        - ax: ax to use
        - plot_f: use colormap to color the plot
    '''
    xvals = np.linspace(*xlim)
    yvals = np.linspace(*ylim)
    X, Y = np.meshgrid(xvals, yvals)
    Z = np.reshape(list(map(f, zip(X.ravel().tolist(), Y.ravel().tolist()))),
                   X.shape)
    ax.contour(X, Y, Z)
    if plot_f: ax.contourf(X, Y, Z, cmap='bone')


def add_path(ax, x_steps, col=blue, label=''):
    '''
    Adds a path of an opitmization algorithm to a figure

    Inputs:
        - ax: ax to use
        - x_steps: path to plot, list of 2D vectors
        - col: color to use
        - label
    '''
    ax.plot([x[0] for x in x_steps], [x[1] for x in x_steps], c=col,
                label=label)



if __name__ == '__main__':
    pass
