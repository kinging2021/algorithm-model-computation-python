import numpy as np

from .smoother import moving_average, lowess_smooth
from .utils import check_results


def test_one_dim_MA():
    check_results(moving_average([1, 2, 3], 1), np.array([1., 2., 3.]))
    check_results(moving_average([1, 2, 3], 2), np.array([1.5, 2, 2.5]))
    check_results(moving_average([1, 2, 3], 3), np.array([1.5, 2, 2.5]))


def test_two_dim_MA():
    check_results(moving_average([[1, 2, 3], [1, 2, 3]], 3), np.array([1.5, 2, 2.5]))
    check_results(moving_average([[2, 2, 2], [1, 2, 3]], 3), np.array([1.5, 2, 2.5]))
    check_results(moving_average([[2, 2, 2], [3, 2, 1]], 3), np.array([2.5, 2, 1.5]))
    check_results(moving_average([[1, 2, 3, 8, 9], [3, 4, 5, 2, 1]], 3, new_sample_size=5),
                  np.array([[1.8, 3.4, 5, 6.6, 8.2], [3.5, 5, 5, 5, 2]]))


def test_lowess():
    check_results(lowess_smooth([1, 5, 3.5, 4, 5])[:, 1], np.array([1, 5, 3.5, 4, 5]))
