# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 2016

@author: michielstock

A small demonstration of simulated annealing
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def one_dimensional_simulated_annealing(f, x0, hyperparameters):
    """
    Simple simulated annealing for a one-dimensional continous problem
    Inputs:
        - f : function to be optimized
        - x0 : starting point (float)
        - hyperparameters: dict with
                * Tmax : maximum (starting) temperature
                * Tmin : minimum (stopping) temperature
                * sigma : standard deviation for sampling a neighbor
                * r : rate of cooling
                * kT : number of iteration with fixed temperature
    Outputs:
            - xstar : obtained minimum
            - xpath : path of x-values explored
            - fbest : best function values in each iteration
            - temperatures : the temperature of each iteration
    """
    # get hyperparameters
    Tmax = hyperparameters['Tmax']
    Tmin = hyperparameters['Tmin']
    sigma = hyperparameters['sigma']
    r = hyperparameters['r']
    kT = hyperparameters['kT']

    # init outputs
    x = x0
    temp = Tmax
    xstar = x0
    fstar = f(xstar)
    xpath = [x]
    fbest = [f(x)]
    temperatures = [temp]

    while temp > Tmin:
        for _ in range(kT):
            xnew = xstar + np.random.randn() * sigma
            fnew = f(xnew)
            if np.exp((fstar - fnew) / temp) > np.random.rand():
                xstar = xnew
                fstar = fnew
        xpath.append(xstar)
        fbest.append(fstar)
        temp *= r
        temperatures.append(temp)
    return xstar, xpath, fbest, temperatures

f_toy_example = lambda x : np.abs(x * np.cos(x)) + 0.5 * np.abs(x)

def plot_SA_example(fun, x0, hyperparameters):
    """
    Demonstrate simulated annealing on a simple example
    """
    xstar, xpath, fbest, temperatures = one_dimensional_simulated_annealing(
                                    fun, x0, hyperparameters)

    fig, axes = plt.subplots(nrows=3, figsize=(5, 10))

    # plot path
    xvals = np.linspace(min(min(xpath), -10), max(max(xpath), 10), 1201)
    axes[0].plot(xvals, list(map(f_toy_example, xvals)), 'k-')
    colors = cm.rainbow(np.linspace(0, 1, len(xpath)))
    for xp, col in zip(xpath, colors):
        axes[0].axvline(x=xp, ymin=0, ymax=max(fbest), c=col)
    axes[0].set_ylabel(u'$f(x)$')
    axes[0].set_xlabel(u'$x$')
    #axes[0].set_title('Path of simulated annealing')

    axes[1].plot(temperatures)
    axes[1].set_ylabel('temperature')
    axes[1].set_xlabel('iteration')
    #axes[1].set_title('Temperature ')

    axes[2].plot(fbest)
    axes[2].set_ylabel('best function value')
    axes[2].set_xlabel('iteration')

if __name__ == '__main__':
    # simple example
    x0 = 55
    hyperparameters = {'Tmax' : 1000, 'Tmin' : 0.1,
                    'r' : 0.8, 'kT' : 10, 'sigma' : 2}

    xstar, xpath, fbest, temperatures = one_dimensional_simulated_annealing(
                                    f_toy_example, x0, hyperparameters)

    fig, axes = plt.subplots(nrows=3, figsize=(5, 10))

    # plot path
    xvals = np.linspace(min(min(xpath), -10), max(max(xpath), 10), 1201)
    axes[0].plot(xvals, list(map(f_toy_example, xvals)), 'k-')
    colors = cm.rainbow(np.linspace(0, 1, len(xpath)))
    for xp, col in zip(xpath, colors):
        axes[0].axvline(x=xp, ymin=0, ymax=max(fbest), c=col)
    axes[0].set_ylabel(u'$f(x)$')
    axes[0].set_xlabel(u'$x$')
    #axes[0].set_title('Path of simulated annealing')

    axes[1].plot(temperatures)
    axes[1].set_ylabel('temperature')
    axes[1].set_xlabel('iteration')
    #axes[1].set_title('Temperature ')

    axes[2].plot(fbest)
    axes[2].set_ylabel('best function value')
    axes[2].set_xlabel('iteration')
    #axes[2].set_title(u'Function value of current best $x$')

    fig.show()
