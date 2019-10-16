import numpy as np

import statsmodels.api as sm

from algorithm.exception import ParamError, DataError


def _ma_smooth(y, min_num):
    r = min_num // 2
    y_mean = []
    for i in range(len(y)):
        begin = max(i - r, 0)
        end = min(i + r + 1, len(y))
        mean = np.mean(y[begin:end])
        y_mean.append(mean)
    return np.array(y_mean)


def moving_average(data, window_size, new_sample_size=-1):
    data = np.array(data)
    if data.ndim == 1:
        # window = np.ones(int(window_size)) / float(window_size)
        # return np.convolve(data, window, 'same')
        return _ma_smooth(data, window_size)
    else:
        if data.ndim > 2:
            raise ParamError("calling moving_average, data dim should not be greater than 2")
        if data.shape[0] != 2:
            raise ParamError("calling moving_average, the first dim of data should be 2")
        # in this case data[0] should be sorted
        if new_sample_size == -1:
            # window = np.ones(int(window_size)) / float(window_size)
            # return np.convolve(data[1], window, 'same')
            return _ma_smooth(data[1], window_size)
        else:
            # primarily for plot
            if new_sample_size < window_size:
                raise ParamError("number of new sample size should not less than window size")
            newx = np.arange(data[0][0], data[0][-1], (data[0][-1] - data[0][0]) / new_sample_size)
            x = []
            y = []
            tmp = []
            data_index = 0
            newx = np.hstack([newx, data[0][-1]])
            for i in range(len(newx) - 1):
                while True:
                    if data_index >= data.shape[1]:
                        break
                    if data[0][data_index] >= newx[i + 1]:
                        break
                    tmp.append(data[1][data_index])
                    data_index += 1
                x.append((newx[i] + newx[i + 1]) / 2)
                if len(tmp) == 0:
                    y.append(y[-1])
                else:
                    y.append(np.mean(tmp))
                    tmp = []

            return np.array([x, y])


def lowess_smooth(data):
    data = np.array(data)
    if data.ndim == 1:
        data = np.array([range(len(data)), data])
    elif data.ndim > 2:
        raise ParamError("calling lowess, data dim should not be greater than 2")
    elif data.shape[0] != 2:
        raise ParamError("calling lowess, the first dim of data should be 2")
    # TODO check data[0] should be sorted
    lowess = sm.nonparametric.lowess
    results = lowess(data[1], data[0], frac=1. / 3, it=0)  # it=0 貌似更光滑一些
    return results
