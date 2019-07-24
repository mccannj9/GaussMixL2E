#! /usr/bin/env python3

import numpy as np


def softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mahalanobis(data, mu, cov, k=2):
    """
    returns mahalanobis distance of each input for
    each gaussian parameterized by mu[i], cov[i].

    mu = (k, d) numpy array where k is number of mixtures
    and d is the dimension of the data.

    cov = (k, d, d) numpy array with the same setup as mu
    in terms of k and d.

    """
    mhd = np.zeros((data.shape[0], mu.shape[0]))

    for i in range(mu.shape[0]):
        _mu = mu[i, :]
        _cov = cov[i, :, :]

        mhd[:, i] = np.sum(
            (data - _mu) @ np.linalg.inv(_cov) * (data - _mu), axis=1
        )

    return np.sqrt(mhd)



if __name__ == "__main__":

    import numpy as np

    np.random.seed(1)
    mu = np.array([[-1, -1], [1, 1]])
    cov = np.array([[[0.1, 0.05],[0.05, 0.1]], [[0.17, -0.095],[-0.095, 0.17]]])

    data_0 = np.random.multivariate_normal(mu[0], cov[0], 500)
    data_1 = np.random.multivariate_normal(mu[1], cov[1], 500)
    data = np.concatenate((data_0, data_1), axis=0)

    x = np.random.randn(5)
    print("Softmax:  ", softmax(x))
    print("Softplus: ", softplus(x))
    print("Sigmoid:  ", sigmoid(x))

    mhd = mahalanobis(data, mu, cov)
    print(mhd.shape)

    from matplotlib import pyplot as plt
    plt.hist(mhd[:,0], bins=30)
    plt.show()
    plt.hist(mhd[:,1], bins=30)
    plt.show()
