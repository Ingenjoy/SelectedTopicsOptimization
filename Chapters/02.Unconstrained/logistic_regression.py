"""
Created on Tuesday 6 February 2018
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Solution of the logistic regression exercise
"""

from unconstrained import *
from scipy.optimize import minimize
import pandas as pd  # pandas allows us to confortably work with datasets in python
from teachingtools import *

cancer_data = pd.read_csv('Data/BreastCancer.csv')  # load data
cancer_data.head()  # show first five rows

# extract response in binary encoding:
# 0 : B(egnin)
# 1 : M(alignant)
binary_response = np.array(list(map(int, cancer_data.y == 'M')), dtype=float)
binary_response = binary_response.reshape((-1, 1))  # make column vector

# extract feature matrix X
features = cancer_data.loc[:,[c[0]=='x' for c in cancer_data.columns]].values

# standarizing features
# this is needed for gradient descent to run faster
features -= features.mean(0)
features /= features.std(0)

# Assignment 1

logistic_map = lambda x : 1 / (1 + np.exp(-x))

def cross_entropy(l, p):
    ce = - np.sum(l * np.log(np.maximum(1e-10, p)))
    ce -= np.sum((1 - l) * np.log(np.maximum(1e-10, 1 - p)))
    return ce

def logistic_loss(w, y, X, lamb):
    score = X.dot(w)
    sigma = logistic_map(score)
    loss =  cross_entropy(y, sigma) + lamb * np.sum( w**2 )
    return loss

def grad_logistic_loss(w, y, X, lamb):
    score = X.dot(w)
    sigma = logistic_map(score)
    grad = - X.T.dot(y - sigma) + 2 * lamb * w
    return grad

def hess_logistic_loss(w, y, X, lamb):
    score = X.dot(w)
    sigma = logistic_map(score)
    hess = np.dot(X.T * (sigma * (1 - sigma)).T, X) + 2 * lamb * np.eye(len(w))
    return hess

if __name__ == '__main__':

    # Assignment 2

    # make functions

    l_loss = lambda w : logistic_loss(w, binary_response, features, 0.1)
    l_grad = lambda w : grad_logistic_loss(w, binary_response, features, 0.1)
    l_hess = lambda w : hess_logistic_loss(w, binary_response, features, 0.1)

    w_star_gd, w_steps_gd, f_steps_gd = gradient_descent(l_loss, np.zeros((30, 1)),
        l_grad, nu=1e-4, trace=True, alpha=0.05, beta=0.6)

    w_star_cd, w_steps_cd, f_steps_cd = coordinate_descent(l_loss, np.zeros((30, 1)),
                l_grad, nu=1e-4, trace=True, alpha=0.05, beta=0.6)

    w_star_n, w_steps_n, f_steps_n = newtons_method(l_loss, np.zeros((30, 1)), l_grad,
                l_hess, epsilon=1e-4, trace=True, alpha=0.2, beta=0.3)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(f_steps_gd)+1), f_steps_gd, color=red, label='GD', lw=2)
    ax.plot(np.arange(1, len(f_steps_cd)+1),f_steps_cd, color=green, label='CD', lw=2)
    ax.plot(np.arange(1, len(f_steps_n)+1),f_steps_n, color=blue, label='Newton', lw=2)
    ax.loglog()
    ax.set_xlabel(r'$k+1$')
    ax.legend(loc=0)
    ax.set_ylabel(r'$\mathcal{L}(\mathbf{w}^{(k)})$')
    fig.savefig('Figures/logistic_convergence.png')

    # Assignment 3

    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))

    for i, lamb in enumerate([1e-3, 1e-1, 1, 10, 100]):
        color = plt.get_cmap('hot')(i / 5)

        # make functions
        l_loss = lambda w : logistic_loss(w, binary_response, features, lamb)
        l_grad = lambda w : grad_logistic_loss(w, binary_response, features, lamb)
        l_hess = lambda w : hess_logistic_loss(w, binary_response, features, lamb)

        w_star_gd, w_steps_gd, f_steps_gd = gradient_descent(l_loss, np.zeros((30, 1)),
            l_grad, nu=1e-4, trace=True, alpha=0.05, beta=0.6)

        w_star_cd, w_steps_cd, f_steps_cd = coordinate_descent(l_loss, np.zeros((30, 1)),
                    l_grad, nu=1e-4, trace=True, alpha=0.05, beta=0.6)

        w_star_n, w_steps_n, f_steps_n = newtons_method(l_loss, np.zeros((30, 1)), l_grad,
                    l_hess, epsilon=1e-4, trace=True, alpha=0.2, beta=0.3)

        # plot results

        for ax, f_steps, name in zip(axes, [f_steps_gd, f_steps_cd, f_steps_n],
                                    ['gradient descent', 'coordinate descent',
                                    'Newton\'s method']):
            ax.plot(np.arange(1, len(f_steps)+1), f_steps, color=color, lw=2,
                        label='$\lambda=${}'.format(lamb))
            ax.loglog()
            ax.set_xlabel(r'$k+1$')
            ax.legend(loc=0)
            ax.set_title(name)
            ax.set_ylabel(r'$\mathcal{L}(\mathbf{w}^{(k)})$')
    fig.tight_layout()
    fig.savefig('Figures/logistic_effect_lambda.png')
