#! /usr/bin/env python3

from matplotlib import pyplot as plt
import seaborn as sns


def plot_data_with_contours(
    data, mu, cov, num_contours=4, labels=None,
    k=2, alpha=1.0, num_points=500
):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=labels, alpha=alpha)
    # plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=alpha)
    polar_space = np.linspace(0, 2*np.pi, num_points)
    polar_coords = np.array([np.cos(polar_space), np.sin(polar_space)]).T

    for j in range(k):
        _mu = mu[j]
        _cov = cov[j]
        _chol = np.linalg.cholesky(_cov)

        for i in range(num_contours + 1):
            xy = np.concatenate(
                (
                    i*polar_coords[:, 0][:, None],
                    i*polar_coords[:, 1][:, None],
                ), axis=1
            )

            tf = (_chol @ xy.T).T + _mu
            ax.plot(tf[:, 0], tf[:, 1], '-'*(1 + (i % 2)))

    # must call plt.show() outside function.
    return fig, ax

if __name__ == "__main__":
    import numpy as np

    np.random.seed(1)
    mu = np.array([[-1, -1], [1, 1]])
    cov = np.array([[[0.1, 0.05],[0.05, 0.1]], [[0.17, -0.095],[-0.095, 0.17]]])

    data_0 = np.random.multivariate_normal(mu[0], cov[0], 500)
    data_1 = np.random.multivariate_normal(mu[1], cov[1], 500)
    data = np.concatenate((data_0, data_1), axis=0)

    f, a = plot_data_with_contours(data, mu, cov, k=2)
    plt.show()
