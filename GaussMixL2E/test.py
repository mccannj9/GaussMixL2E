#! /usr/bin/env python3

from utils.misc import stochastic_mini_batch
from utils.simulate import build_toy_dataset


from layer.model import GaussianMixture

data = build_toy_dataset(500)
real_labels = data[:, 2]
# data = data[:, 0:2]

GMM = GaussianMixture(2, 2)
GMM(sum_to_one=False, learning_rate=0.01)

print(data.shape)
test = GMM.fit(data[:, 0:2])
