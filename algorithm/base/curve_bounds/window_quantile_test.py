from .window_quantile import get_curves, get_resample_curves
from data_sample.scatter_data import get_scatter_data

import matplotlib.pyplot as plt


def draw(data, x, mean, upper, lower, title="figure"):
    plt.scatter(data[0], data[1], marker='.', color='#0000ba')
    plt.plot(x, mean)
    plt.plot(x, upper)
    plt.plot(x, lower)
    plt.title(title)
    plt.show()


def test_curves():
    data = get_scatter_data()
    x, mean, upper, lower = get_curves(data[0], data[1])
    draw(data, x, mean, upper, lower, "origin figure")
    return


def test_resample_curves():
    data = get_scatter_data()
    x, mean, upper, lower = get_resample_curves(data[0], data[1])
    draw(data, x, mean, upper, lower, "resample figure")
    return
