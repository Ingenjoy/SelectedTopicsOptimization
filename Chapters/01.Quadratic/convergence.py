"""
Created on Friday 25 August 2017
Last update: Tuesday 26 December 2017

@author: Michiel Stock
michielfmstock@gmail.com

Some functions to illustrate the convergence of the optimization algorithms
for quadratic systems
"""

import sys
sys.path.append('helpers/')
from colors import colors_list
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#sns.set_style('white')

def gd_error_decomposition(eigenvalues=[0.1, 1.5, 1.6, 1.8, 2],
                    x0=np.ones((5, 1)), n_steps=50, t='optimal', ax=None,
                    cumulative=True):
    """
    #FIX: docstring
    short description

    Inputs:
        -

    Output:
        -
    """
    if t=='optimal':
        t = 2 / (min(eigenvalues) + max(eigenvalues))
    n = len(eigenvalues)
    ev = np.reshape(eigenvalues, (-1, 1))  # vector format
    steps = np.arange(0, n_steps + 1)
    error_per_comp = (1 - t * ev) ** (2 * steps.reshape((1, -1)))
    error_per_comp *= ev * x0**2
    if ax is not None:
        colors = iter(colors_list)
        prev = np.zeros_like(steps) + 1e-10
        current = prev + 0
        for i, val in enumerate((eigenvalues)):
            label=r'$\lambda_{}=${}'.format(i+1, val)
            if cumulative:
                current += error_per_comp[i,:]
                ax.fill_between(steps+1, prev, current,
                        color=next(colors),
                        label=label)
                prev[:] = current
            else:
                ax.plot(steps+1, error_per_comp[i,:],color=next(colors),
                            label=label, lw=2)
        ax.legend(loc=0)
        if cumulative:
            ax.set_ylabel(r'$f(\mathbf{x}^{(k)})-f(\mathbf{x}^\star)$')
            #ax.set_ylim([1e-10, 5])
            ax.set_ylim([1e-10, error_per_comp[:,0].sum()])
        else:
            ax.set_ylabel(r'$(1-t\lambda_i)^{2k}\lambda_i[\mathbf{u}_i^\intercal(\mathbf{x}^{(k)} - \mathbf{x}^\star)]^2$')
        ax.set_xlabel(r'$k+1$')
        ax.set_ylim([1e-10, error_per_comp[:,0].sum()])
    return error_per_comp

def gd_convergence_and_bound(eigenvalues=[0.1, 1.5, 1.6, 1.8, 2], n_steps=50,
                        t='optimal', x0=np.ones((5, 1))):
    # condition number
    #FIX: docstring
    kappa = max(eigenvalues) / min(eigenvalues)
    c = 1 - 1/kappa
    error = gd_error_decomposition(eigenvalues=eigenvalues, n_steps=n_steps,
        t=t, x0=x0).sum(0)
    bound = np.sum([xi**2 * e for xi, e in zip(x0, eigenvalues)]) * c ** np.arange(0, n_steps+1)
    return error, bound

if __name__ == '__main__':
    # choose format of the plots
    format = 'png'

    # convergence plot

    n_steps = 1000 - 1

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))
    E = gd_error_decomposition(ax=ax0, n_steps=n_steps, cumulative=True)
    ax0.semilogx()
    ax0.set_title('Error of gradient descent\n(cumulative eigencomponents)')

    gd_error_decomposition(ax=ax1, n_steps=n_steps, cumulative=False)
    ax1.set_title('Error of gradient descent\n(individual eigencomponents)')
    ax1.loglog()
    fig.tight_layout()
    fig.savefig('Chapters/01.Quadratic/Figures/convergence_decomposition.{}'.format(format))


    # error vs bound
    kappas = [1.1, 2.5, 5, 50, 100]

    n_steps = 10000
    colors = iter(colors_list)
    fig, ax = plt.subplots()
    steps = np.arange(0, n_steps+1)+1
    for kappa in kappas:
        color = next(colors)
        error, bound = gd_convergence_and_bound(eigenvalues=[1, kappa],
                    n_steps=n_steps, x0=np.ones((2, 1)))
        ax.plot(steps, error, color=color, ls='-', label=r'$\kappa$={}'.format(kappa), lw=2)
        ax.plot(steps, bound, color=color, ls='--', lw=2)

    ax.set_ylim([1e-6, 200])
    ax.loglog()
    ax.set_title('Convergence (-) and bound (--) of GD\nfor different condition numbers')
    ax.legend(loc=0)
    ax.set_ylabel(r'$f(\mathbf{x}^{(k)})-f(\mathbf{x}^\star)$')
    ax.set_xlabel(r'$k+1$')
    fig.tight_layout()
    fig.savefig('Chapters/01.Quadratic/Figures/convergence_bound.{}'.format(format))

    plt.close('all')
