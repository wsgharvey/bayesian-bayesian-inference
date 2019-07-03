from abc import abstractmethod
import numpy as np
from gptools.gaussian_process import GaussianProcess
from gptools.mean import MeanFunction

class StochasticProcess():
    """
    simple interface for stochastic process implementation so we can swap out package,
    use distribution over hyperparamters, use neural process etc.
    """
    @abstractmethod
    def add_data(self, X, y, derivative):
        """
        Derivative can be None if not available, else same shape as X
        """
        pass

    @abstractmethod
    def sample(self, X, n_samples):
        """
        Returns n_samples x n_dimensions array
        """


class GP(StochasticProcess):
    def __init__(self, mean_func, kernel):
        mu = MeanFunction(mean_func) if mean_func is not None else None
        self._gp = GaussianProcess(kernel, mu=mu, diag_factor=1e10)
        self.has_data = False

    def add_data(self, X, y, derivative=None):
        self._gp.add_data(X, y)
        if derivative is not None:
            assert X.shape == derivative.shape
            N, D = X.shape
            for d in range(D):
                """
                gptools has confusing docs, but it appears that, to observe
                first derivative wrt dimension d, n is 1 at element d and 0
                everywhere else
                """
                derivative_matrix = np.zeros((N, D)).astype(int)
                derivative_matrix[:, d] = 1
                self._gp.add_data(X, derivative[:, d], n=derivative_matrix)
        self.has_data = True

    def sample(self, X, n_samples):
        if not self.has_data:
            # dummy observation necessary due to apparent bug in gptools
            self._gp.add_data(np.array([[0.]*X.shape[1]]), np.array([0.]), err_y=1000000)
            self.has_data = True
        return self._gp.draw_sample(X, num_samp=n_samples).transpose()


# if __name__ == '__main__':
#     import itertools as it
#     import numpy as np
#     from gptools.kernel.squared_exponential import SquaredExponentialKernel
#     import matplotlib.pyplot as plt

#     def f(X):
#         return X[:, 0] + X[:, 1]**2

#     def df(X):
#         N, D = X.shape
#         return np.stack([np.ones(len(X)), 2*X[:, 1]]).T

#     def mean(X, n, hyper_deriv=0):
#         C = 0.000
#         p = 6
#         if sum(n) == 0:
#             return -C*(X**p).sum(axis=1)
#         elif sum(n) == 1:
#             d = np.argmax(n)
#             return -C*p*X[:, d]**(p-1)
#         else:
#             raise NotImplementedError(f"n should sum to 0 or 1, not {sum(n)}")


#     N, D = 100, 2

#     kernel = SquaredExponentialKernel(num_dim=D)

#     np.random.seed(0)
#     X = (np.random.rand(N, D)-0.5)*2*5
#     y = f(X)
#     derivative = df(X)

#     gp = GP(mean_func=mean, kernel=kernel)
#     gp.add_data(X, y, derivative)

#     Xs = np.arange(-5, 5, 0.2)
#     Xt = np.array(list(it.product(Xs, Xs)))
#     fig, axes = plt.subplots(2)
#     for y in gp.sample(Xt, 1):
#         axes[0].contour(Xs, Xs, y.reshape(len(Xs), len(Xs)))
#         axes[0].scatter(X[:, 0], X[:, 1], color='r')

#     axes[1].contour(Xs, Xs, f(Xt).reshape(len(Xs), len(Xs)))
#     plt.show()
