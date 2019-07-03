import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from gptools.kernel import SquaredExponentialKernel, SumKernel, ProductKernel
from stochastic_process import GP


def default_mean(X, n, hyper_deriv=0):
    C = 0.0001
    p = 6
    if sum(n) == 0:
        return -C*(X**p).sum(axis=1)
    elif sum(n) == 1:
        d = np.argmax(n)
        return -C*p*X[:, d]**(p-1)
    else:
        raise NotImplementedError(f"n should sum to 0 or 1, not {sum(n)}")


def get_default_kernel(n_dim):
    def squared_exp_params(sigma, lengthscale):
        return [sigma, *[lengthscale]*n_dim]

    # soak up unknown log-likelihood normalization
    params1 = squared_exp_params(1000, 10000000)
    k1 = SquaredExponentialKernel(num_dim=n_dim,
                                  initial_params=params1,
                                  fixed_params=[True for _ in params1])

    #  model of changes in log-likelihood
    params2 = squared_exp_params(30, 2)
    k2 = SquaredExponentialKernel(num_dim=n_dim,
                                  initial_params=params2,
                                  fixed_params=[True for _ in params2])
    return SumKernel(k1, k2)

class HyperDistribution():
    """
    Distribution over distributions (via a GP which parameterises the log pdf)
    """
    def __init__(self, n_dim, mean_func, kernel):
        self.n_dim = n_dim
        if mean_func is None:
            mean_func = default_mean
        if kernel is None:
            kernel = get_default_kernel(n_dim)
        self.process = GP(mean_func, kernel)
        self.set_grid(bounds=[-10., 10.], dx=0.05)
        self.obs_X = np.array([]).reshape(0, n_dim)
        self.obs_y = np.array([]).reshape(0)

    def observe(self, X, y, derivative):
        self.obs_X = np.concatenate([self.obs_X, X], axis=0)
        self.obs_y = np.concatenate([self.obs_y, y], axis=0)
        self.process.add_data(X, y, derivative)

    def set_grid(self, bounds, dx):
        """
        Set grid to be used for evaluating function to plot or approximate integral.
        """
        self.dx = dx
        self.X1 = np.arange(bounds[0], bounds[1], dx)
        self.X = np.array(list(it.product(*[self.X1]*self.n_dim)))

    def sample_unnormed_logpdf(self, n_samples):
        return self.process.sample(self.X, n_samples)

    def batch_integrate(self, batched_values, weights=None):
        """
        Naive integration of function by values given at each point in self.X
        If weights are given , will integrate product of sampled_values and weights
        batched_values - batch_size x len(self.X)
        weights - len(self.X)
        """
        if weights is None:
            return batched_values.sum(axis=1) * self.dx
        else:
            return (weights.reshape(1, -1) @ batched_values.T).reshape(-1) * self.dx

    def sample_pdf(self, n_samples, return_unnormed_logpdf_samples=False):
        unnormed_logpdf = self.sample_unnormed_logpdf(n_samples)
        unnormed_pdf = np.exp(unnormed_logpdf - unnormed_logpdf.max(axis=1, keepdims=True))
        evidence = self.batch_integrate(unnormed_pdf)
        pdf = unnormed_pdf / evidence.reshape(n_samples, 1)
        if return_unnormed_logpdf_samples:
            return pdf, unnormed_logpdf
        else:
            return pdf

    def sample_expectation(self, f, n_samples, return_unnormed_logpdf_samples=False):
        """
        f should work on batched arguments of shape: batch_size x n_dim
        """
        pdf = self.sample_pdf(n_samples, return_unnormed_logpdf_samples)
        if return_unnormed_logpdf_samples:
            pdf, unnormed_logpdf = pdf
        integral = self.batch_integrate(pdf, f(self.X))
        if return_unnormed_logpdf_samples:
            return integral, unnormed_logpdf
        else:
            return integral

    def BOED(self, f, n_samples):
        """
        TODO: optionally use gradient information when selecting design
        """
        expectation, unnormed_logpdf = self.sample_expectation(f, n_samples, True)
        def normalise(array):
            # normalise each column in `array`
            array = array - array.mean(axis=0, keepdims=True)
            eps = 1e-5
            std = array.std(axis=0, keepdims=True)
            eps = 1e-6
            std[std < eps] = eps
            return array/std
        stds = unnormed_logpdf.std(axis=0, keepdims=True)
        normed_expectation = normalise(expectation)
        normed_unnormed_logpdfs = normalise(unnormed_logpdf)
        correlations = (normed_expectation @ normed_unnormed_logpdfs) / n_samples
        optimal_index = abs(correlations).argmax()
        return self.X[optimal_index]

    def plot_pdf_samples(self, n_samples, ax):
        if self.n_dim not in [1, 2]:
            raise NotImplementedError
        for pdf in self.sample_pdf(n_samples):
            if self.n_dim == 1:
                ax.plot(self.X.reshape(-1), pdf)
            elif self.n_dim == 2:
                ax.contourf(self.X1, self.X1, pdf.reshape(len(self.X1), len(self.X1)))
        if self.n_dim == 1:
            ax.scatter(self.obs_X, [0]*len(self.obs_X), color='r', marker='x')
        elif self.n_dim == 2:
            ax.scatter(self.obs_X[:,0], self.obs_X[:,1], color='r', marker='x')

    def plot_unnormed_logpdf_samples(self, n_samples, ax):
        if self.n_dim not in [1, 2]:
            raise NotImplementedError
        for logpdf in self.sample_unnormed_logpdf(n_samples):
            if self.n_dim == 1:
                ax.plot(self.X.reshape(-1), logpdf)
            elif self.n_dim == 2:
                ax.contourf(self.X1, self.X1, logpdf.reshape(len(self.X1), len(self.X1)))
        if self.n_dim == 1:
            ax.scatter(self.obs_X, self.obs_y, color='r', marker='x')
        elif self.n_dim == 2:
            ax.scatter(self.obs_X[:,0], self.obs_X[:,1], color='r', marker='x')

    def plot_marginal_pdf(self, n_samples, ax):
        if self.n_dim not in [1, 2]:
            raise NotImplementedError
        pdf_samples = self.sample_pdf(n_samples)
        marginal = pdf_samples.mean(axis=0)
        if self.n_dim == 1:
            ax.plot(self.X.reshape(-1), marginal)
            ax.scatter(self.obs_X, [0]*len(self.obs_X), color='r', marker='x')
        elif self.n_dim == 2:
            ax.contourf(self.X1, self.X1, marginal.reshape(len(self.X1), len(self.X1)))
            ax.scatter(self.obs_X[:,0], self.obs_X[:,1], color='r', marker='x')

    def plot(self):
        fig, axes = plt.subplots(ncols=3, figsize=(16, 4))
        self.plot_marginal_pdf(1000, axes[0])
        self.plot_pdf_samples(10, axes[1])
        self.plot_unnormed_logpdf_samples(10, axes[2])
