#! /usr/bin/env python3

from matplotlib import pyplot as plt
import seaborn as sns


def plot_data_with_contours(
    data, mu, cov, num_contours=4, labels=None,
    k=2, alpha=1.0, num_points=500
):
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=alpha)
    polar_space = np.linspace(0, 2*np.pi, num_points)
    polar_coords = np.array([np.cos(polar_space), np.sin(polar_space)]).T

    for i in range(num_contours + 1):
        pass

if __name__ == "__main__":
    pass
