import numpy as np
from geomdl import BSpline
from geomdl import knotvector

from algorithm.exception import ParamError


# EnergyEfficiencyCurve

class EECurve(object):
    def __init__(self,
                 data,
                 x_window,
                 min_num_sample,
                 degree,
                 out_size,
                 bound_scale=1.0,
                 clamped=True):

        self.data = data
        self.x_window = x_window
        self.min_num_sample = min_num_sample
        self.degree = degree
        self.out_size = out_size
        self.bound_scale = bound_scale
        self.clamped = clamped

        self.sample_points_expected = None
        self.eval_points_expected = None

        self.sample_points_lower = None
        self.eval_points_lower = None

        self.sample_points_upper = None
        self.eval_points_upper = None

        self.__arg_check()

    def evaluate(self):
        def get_evalpts(data):
            curve = BSpline.Curve()
            curve.degree = self.degree
            curve.ctrlpts = data.tolist()
            curve.knotvector = knotvector.generate(curve.degree, len(curve.ctrlpts),
                                                   clamped=self.clamped)
            curve.delta = 1.0 / self.out_size
            curve.evaluate()
            return np.array(curve.evalpts)[:, 0:2]

        self.eval_points_expected = get_evalpts(self.sample_points_expected)
        self.eval_points_lower = get_evalpts(self.sample_points_lower)
        self.eval_points_upper = get_evalpts(self.sample_points_upper)

    def sampling(self):
        self._sampling_expected()
        self._sampling_lower_upper()

    def _sampling_expected(self):
        j = 0
        step = self.data[j, 0] + self.x_window
        points = []
        for i in range(self.data.shape[0]):
            if i - j >= self.min_num_sample and self.data[i, 0] >= step:
                if self.data.shape[0] - i <= self.min_num_sample:
                    points.append(self.__select_expected_points(self.data[j:, :]))
                else:
                    points.append(self.__select_expected_points(self.data[j:i, :]))
                    j = i
                    step += self.x_window
                    while self.data[i, 0] >= step:
                        step += self.x_window
        self.sample_points_expected = np.vstack(points)

    def _sampling_lower_upper(self):
        j = 0
        step = self.data[j, 0] + self.x_window
        points_lower = []
        points_upper = []
        for i in range(self.data.shape[0]):
            if self.data[i, 0] >= step:
                if self.data.shape[0] - i <= self.min_num_sample:
                    p_lower, p_upper = self.__get_lower_upper_points(self.data[j:, :])
                else:
                    p_lower, p_upper = self.__get_lower_upper_points(self.data[j:i, :])
                    j = i
                    step += self.x_window
                    while self.data[i, 0] >= step:
                        step += self.x_window
                points_lower.append(p_lower)
                points_upper.append(p_upper)
        if len(points_lower) < self.degree + 1:
            # Number of control points should be at least degree + 1
            raise ParamError('The input data is not enough or x_window is too wide, '
                             'please check input data, x_window and min_num_sample')

        if self.data[0, 0] < points_upper[0][0]:
            data = self.data[self.data[:, 0] < points_upper[0][0]]
            y = np.max([data[:, 1].max(), points_upper[0][1]])
            points_upper[0] = [self.data[0, 0], y]

        if self.data[0, 0] < points_lower[0][0]:
            data = self.data[self.data[:, 0] < points_lower[0][0]]
            y = np.min([data[:, 1].min(), points_lower[0][1]])
            points_lower[0] = [self.data[0, 0], y]

        if self.data[-1, 0] > points_upper[-1][0]:
            data = self.data[self.data[:, 0] > points_upper[-1][0]]
            y = np.max([data[:, 1].max(), points_upper[-1][1]])
            points_upper[-1] = [self.data[-1, 0], y]

        if self.data[-1, 0] > points_lower[-1][0]:
            data = self.data[self.data[:, 0] > points_lower[-1][0]]
            y = np.min([data[:, 1].min(), points_lower[-1][1]])
            points_lower[-1] = [self.data[-1, 0], y]

        self.sample_points_lower = np.vstack(points_lower)
        self.sample_points_upper = np.vstack(points_upper)

    @staticmethod
    def __select_expected_points(data):
        distance_x = np.square(data[:, 0] - data[:, 0].mean())
        distance_y = np.square(data[:, 1] - data[:, 1].mean())
        index = np.argmin(distance_x + distance_y)
        return data[index, :]

    def __get_lower_upper_points(self, data):
        y = data[:, 1]
        y_min = y.min()
        y_max = y.max()
        q1 = np.percentile(y, 25, interpolation='nearest')
        q2 = np.percentile(y, 50, interpolation='nearest')
        q3 = np.percentile(y, 75, interpolation='nearest')
        x = data[np.where(y == q2)[0][0], 0]
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        if upper >= y_max:
            p_upper = data[np.where(y == y_max)[0][0]]
        else:
            p_upper = np.array([x, upper])

        if lower <= y_min:
            p_lower = data[np.where(y == y_min)[0][0]]
        else:
            p_lower = np.array([x, lower])

        p_lower[1] = self.bound_scale * (p_lower[1] - q2) + q2
        p_upper[1] = self.bound_scale * (p_upper[1] - q2) + q2

        return p_lower, p_upper

    def __arg_check(self):
        if self.degree < 3:
            raise ParamError('Parameter \'degree\' should be at least 3')
        if self.min_num_sample < 1:
            raise ParamError('Parameter \'min_num_sample\' should be at least 1')
        if self.data.shape[0] < self.degree + 1:
            # Number of control points should be at least degree + 1
            raise ParamError('The input data is not enough or x_window is too wide, '
                             'please check input data, x_window and min_num_sample')
