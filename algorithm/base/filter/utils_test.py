import numpy as np

from .utils import check_results


def test_check_works():
    check_results(1, 1)
    check_results([1, 3], [1, 3])
    check_results(np.array([[1, 2], [1, 2]]), np.array([[1, 2], [1, 2]]))
    check_results({"a": 1 + 1E-10}, {"a": 1.})
    return
