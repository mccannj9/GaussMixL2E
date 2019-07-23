#! /usr/bin/env python3

import numpy as np


def softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    x = np.random.randn(5)
    print("Softmax:  ", softmax(x))
    print("Softplus: ", softplus(x))
    print("Sigmoid:  ", sigmoid(x))
