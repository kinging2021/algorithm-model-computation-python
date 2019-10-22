from data_sample.scatter_data import get_scatter_data

from .window_quantile import get_curves, get_resample_curves
from .utils import draw


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
