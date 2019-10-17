import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from algorithm.base.filter.smoother import moving_average
from algorithm.base.filter.denoiser import wavelet_filter
from .utils import sort_by_x


def gbrt_curve_1D(x, y, lower_quantile=0.01, upper_quantile=0.99):
    x = np.array(x).reshape(-1)
    x, y = sort_by_x(x, y)
    x = x.reshape((-1, 1))

    clf = GradientBoostingRegressor(loss='quantile', alpha=upper_quantile,
                                    n_estimators=250, max_depth=3,
                                    learning_rate=.1, min_samples_leaf=9,
                                    min_samples_split=9)

    clf.fit(x, y)
    y_upper = clf.predict(x)
    clf.set_params(alpha=lower_quantile)
    clf.fit(x, y)
    y_lower = clf.predict(x)
    clf.set_params(loss='ls')
    clf.fit(x, y)
    y_pred = clf.predict(x)

    window_size = len(x) // 10
    x = x.reshape(-1)
    y_upper = moving_average(wavelet_filter(y_upper), window_size)
    y_lower = moving_average(wavelet_filter(y_lower), window_size)
    y_mean = moving_average(wavelet_filter(y_pred), window_size)
    return x, y_upper, y_lower, y_mean


def gbrt_surface(x, y, z):
    regressor = GradientBoostingRegressor(loss='ls',
                                          n_estimators=250, max_depth=3,
                                          learning_rate=.1, min_samples_leaf=9,
                                          min_samples_split=9)

    model = regressor.fit(np.array([x, y]).T, z)

    def pooling(array, r):
        a = array.copy()
        shape = a.shape
        for i in range(r, shape[0] - r):
            for j in range(r, shape[1] - r):
                a[i][j] = np.mean(array[i - r:i + r + 1, j - r:j + r + 1])
        return a

    def fun(x, y):
        return model.predict(np.array([x, y]).T)

    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)
    print(min_x, min_y)
    x = np.arange(min_x, max_x, (max_x - min_x) / 100)
    y = np.arange(min_y, max_y, (max_y - min_y) / 100)
    x, y = np.meshgrid(x, y)
    zs = np.array(fun(np.ravel(x), np.ravel(y)))
    z = zs.reshape(x.shape)
    z = pooling(z, 5)
    return x, y, z
