import time

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin


class NCA(BaseEstimator, TransformerMixin):
    r"""Neighborhood Components Analysis(NCA).

    NCA is a distance metric learning algorithm which aims to improve the
    accuracy of nearest neighbors classification compared to the standard
    Euclidean distance. The algorithm directly maximizes a stochastic variant
    of the leave-one-out k-nearest neighbors(KNN) score on the training set.
    It can also learn a low-dimensional linear transformation of data that can
    be used for data visualization and fast classification.

    """

    def __init__(self, n_components=-1, max_iter=1000, verbose=False):
        self.n_components = n_components
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y=None):
        d = X.shape[1]
        if self.n_components < 0:
            self.n_components = d
        m = self.n_components

        # Measure the total training time
        train_time = time.time()

        # Initialize A to a scaling matrix
        A = np.random.randn(d, m)

        # Run NCA
        params = {
            'method': 'L-BFGS-B',
            'fun': self.compute_gradient,
            'args': (X, y),
            'jac': True,
            'x0': A.ravel(),
            'options': dict(maxiter=self.max_iter),
            'tol': 1e-10
        }

        # counter for the number of iteration
        self.n_iter_ = 1

        # Call the optimizer
        result = minimize(**params)

        # saving the results
        self.L = result.x.reshape(d, m)
        self.train_time = time.time() - train_time

        # for debugging
        if self.verbose:
            print('%s\nTraining took %.4f seconds' % ('-'*31, self.train_time))

        return self

    def compute_gradient(self, A, X, y):
        """Compute the objective function value and gradients.

        Args:
            A (array-like, shape=[n_features * n_components]):
                The linear transformation matrix.
            X (array-like, shape=[n_examples, n_features]): Training data.
            y (array-like, shape=[n_examples]): Class labels
                for each data sample.

        Returns:
            value (float): The objective function value.
            grads (array-like, shape=[n_features * n_components]): The
                gradients.

        """

        if self.verbose and self.n_iter_ == 1:
            header = '{:>10} {:>20}'.format('Iteration', 'Objective Value')
            print('{sep}'.format(sep='-' * len(header)))
            print('{header}\n{sep}'.format(
                header=header, sep='-' * len(header)))

        # compute the label matrix
        label_matrix = y[:, np.newaxis] == y[np.newaxis, :]
        np.fill_diagonal(label_matrix, 0)

        # transform A to matrix format
        A = A.reshape(X.shape[1], -1)

        # convert inputs into tensors
        A_tensor = torch.tensor(A.astype(float), requires_grad=True)
        X_tensor = torch.tensor(X.astype(float), requires_grad=False)
        label_tensor = torch.tensor(
            label_matrix.astype(float), requires_grad=False)

        # compute the embedded matrix
        X_embedded = torch.matmul(X_tensor, A_tensor)

        # compute the distance matrix
        X_squared = torch.pow(X_embedded, 2).sum(dim=1, keepdim=True)
        Dist = X_squared - 2 * \
            torch.matmul(X_embedded, X_embedded.t()) + X_squared.t()

        # compute the log sum
        log_sum = torch.logsumexp(-Dist, dim=1, keepdim=True)
        # compute the numerator p_ij
        p_ij = torch.mul(Dist, label_tensor)

        # compute the loss
        loss = torch.sum(p_ij + log_sum)

        # backward to compute the gradients
        loss.backward()

        # update the number of iterations
        self.n_iter_ += 1

        if self.verbose:
            values_fmt = '{n_iter:>10} {loss:>20.6e}'
            print(values_fmt.format(n_iter=self.n_iter_, loss=loss.item()))

        return loss.item(), A_tensor.grad.numpy().flatten()

    def return_M(self):
        """Return the Mahalanobis matrix."""
        return self.L @ self.L.T

    def return_L(self):
        """Return the linear transformation matrix."""
        return self.L

    def transform(self, X):
        return X @ self.L


if __name__ == "__main__":
    print(NCA())
