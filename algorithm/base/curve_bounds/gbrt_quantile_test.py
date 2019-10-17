from data_sample.scatter_data import get_scatter_data, get_scatter_data_3D

from .gbrt_quantile import gbrt_curve_1D, gbrt_surface
from .utils import draw, draw_surface


def test_gbrt_curves():
    data = get_scatter_data()
    x, mean, upper, lower = gbrt_curve_1D(data[0], data[1])
    draw(data, x, mean, upper, lower, "gbrt figure")
    return


def test_gbrt_surface():
    data = get_scatter_data_3D()
    surface = gbrt_surface(data[0], data[1], data[2])
    draw_surface(surface[0], surface[1], surface[2])
