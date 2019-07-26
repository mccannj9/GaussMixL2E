#! /usr/bin/env python3

import numpy as np

def build_toy_dataset(N, K=None, D=None, pi=None):
    pi = np.array([0.4, 0.6])
    mus = [[2, 2], [-2, -2]]
    stds = [[[0.1, 0.0], [0.0, 0.1]], [[0.1, 0.0], [0.0, 0.1]]]
    x = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, 0:2] = np.random.multivariate_normal(mus[k], stds[k])
        x[n, 2] = k
    return x


def build_toy_dataset_with_outliers(N, K=None, D=None, pi=None):
    pi = np.array([0.375, 0.575, 0.05])
    mus = [[1, 1], [-1, -1], [-5.5, -5.5]]
    stds = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
    x = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, 0:2] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
        x[n, 2] = k
    return x

if __name__ == "__main__":
    pass
