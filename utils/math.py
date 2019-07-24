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
    mhd = np.zeros((data.shape[0], k))

    for i in range(k):
        _mu = mu[i, :]
        _cov = cov[i, :, :]

        mhd[:, i] = np.sum(
            (data - _mu) @ np.linalg.inv(_cov) * (data - _mu), axis=1
        )

    return np.sqrt(mhd)

if __name__ == "__main__":
    x = np.random.randn(5)
    print("Softmax:  ", softmax(x))
    print("Softplus: ", softplus(x))
    print("Sigmoid:  ", sigmoid(x))
