import numpy as np

from algorithm.base.regression.poly_ridge import PolyRidge
from algorithm.base.filter.smoother import moving_average
from algorithm.base.filter.denoiser import wavelet_filter
from .utils import sort_by_x


def get_curves(x, y, window_size=20, lower_quantile=0.01, upper_quantile=0.99):
    x,y = sort_by_x(x,y)

    rg = PolyRidge(alpha=1)
    rg.fit(x, y)
    y_hat = rg.predict(x)

    y_upper = []
    y_lower = []
    r = window_size // 2
    for i in range(len(x)):
        begin = max(i - r, 0)
        end = min(i + r, len(x), i * 2 - begin)
        begin = max(begin, 2 * i - end)
        if begin == end:
            mean = y_hat[begin]
        else:
            mean = np.mean(y_hat[begin:end])
        upper = [mean]
        lower = [mean]
        for j in range(begin, end):
            if y[j] > mean:
                upper.append(y[j])
            elif y[j] < mean:
                lower.append(y[j])
            else:
                continue

        upper.sort()
        lower.sort()
        y_upper.append(np.percentile(upper, upper_quantile * 100))
        y_lower.append(np.percentile(lower, lower_quantile * 100))

    y_mean = moving_average(y_hat, window_size)
    y_upper = moving_average(y_upper, window_size)
    y_lower = moving_average(y_lower, window_size)
    return x, y_mean, y_upper, y_lower


def get_resample_curves(x, y, window_size=20, lower_quantile=0.01, upper_quantile=0.99, sample_num=100):

    x,y = sort_by_x(x,y)

    rg = PolyRidge(alpha=1)
    rg.fit(x, y)

    xnew = np.arange(x[0], x[-1], (x[-1] - x[0]) / sample_num)
    y_hat_new = rg.predict(xnew)

    y_upper = []
    y_lower = []

    r = window_size // 2

    for i in range(len(xnew)):
        begin = max(i - r, 0)
        end = min(i + r, len(xnew), i * 2 - begin)
        begin = max(begin, 2 * i - end)
        if begin == end:
            mean = y_hat_new[begin]
        else:
            mean = np.mean(y_hat_new[begin:end])
        upper = [mean]
        lower = [mean]
        for j in range(len(y)):
            if x[j] < xnew[begin]:
                continue
            elif x[j] >= xnew[end - 1]:
                break
            if y[j] > mean:
                upper.append(y[j])
            elif y[j] < mean:
                lower.append(y[j])
            else:
                continue

        upper.sort()
        lower.sort()
        y_upper.append(np.percentile(upper, upper_quantile * 100))
        y_lower.append(np.percentile(lower, lower_quantile * 100))

    y_mean = moving_average(wavelet_filter(y_hat_new), r)
    y_upper = moving_average(wavelet_filter(y_upper), r)
    y_lower = moving_average(wavelet_filter(y_lower), r)
    return xnew, y_mean, y_upper, y_lower
