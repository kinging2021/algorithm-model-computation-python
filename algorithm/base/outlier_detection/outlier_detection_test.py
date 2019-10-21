import numpy as np
import matplotlib.pyplot as plt

from data_sample.scatter_data import get_scatter_data
from .outlier_dection import get_outlier_labels


def test_PCA_outlier_detection():
    data = get_scatter_data()
    X = np.array(data).T
    label = get_outlier_labels(X, 0.002)

    plt.scatter(X[label == 0][:, 0], X[label == 0][:, 1])
    plt.scatter(X[label == 1][:, 0], X[label == 1][:, 1], color='r')
    plt.show()
