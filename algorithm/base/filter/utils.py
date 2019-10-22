import numpy as np


def check_results(x, y):
    print(x,y)
    assert (type(x) == type(y))
    if isinstance(x, dict):
        assert (len(x.keys()) == len(y.keys()))
        for k in x.keys():
            check_results(x[k], y[k])
    elif isinstance(x, list) or isinstance(x, tuple):
        assert (len(x) == len(y))
        for i in range(len(x)):
            check_results(x[i], y[i])
    elif isinstance(x, np.ndarray):
        assert (x.ndim == y.ndim)
        for i in range(x.shape[0]):
            check_results(x[i], y[i])
    elif isinstance(x, float):
        assert (abs(x - y) < 1E-8)
    elif isinstance(x, int) or isinstance(x, np.int32):
        assert (x == y)
    else:
        raise Exception("type %s not known" % type(x))
