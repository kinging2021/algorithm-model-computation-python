import numpy as np
from pyod.models.abod import ABOD
from .ee_curve import EECurve

from algorithm.exception import ParamError, DataError


class EECurveGSB(EECurve):
    def __init__(self,
                 x,
                 y,
                 x_range,
                 y_range,
                 bounds=(False, False, False, False),
                 outliers_fraction=0.005,
                 x_window=5.0,
                 min_num_sample=5,
                 degree=6,
                 out_size=100,
                 bound_scale=1.0,
                 clamped=True):

        self.__data_check(x, y)
        super(EECurveGSB, self).__init__(
            np.vstack((x, y)).T, x_window, min_num_sample,
            degree, out_size, bound_scale, clamped
        )

        self.x_range = x_range
        self.y_range = y_range
        self.bounds = bounds
        self.outliers_fraction = outliers_fraction

    def get_result(self):
        expected_x_min, expected_y_min = self.eval_points_expected.min(axis=0)
        expected_x_max, expected_y_max = self.eval_points_expected.max(axis=0)
        lower_x_min, lower_y_min = self.eval_points_lower.min(axis=0)
        lower_x_max, lower_y_max = self.eval_points_lower.max(axis=0)
        upper_x_min, upper_y_min = self.eval_points_upper.min(axis=0)
        upper_x_max, upper_y_max = self.eval_points_upper.max(axis=0)
        return {
            'expected': self.eval_points_expected.tolist(),
            'expected_range': {
                'x': [expected_x_min, expected_x_max],
                'y': [expected_y_min, expected_y_max],
            },
            'lower': self.eval_points_lower.tolist(),
            'lower_range': {
                'x': [lower_x_min, lower_x_max],
                'y': [lower_y_min, lower_y_max],
            },
            'upper': self.eval_points_upper.tolist(),
            'upper_range': {
                'x': [upper_x_min, upper_x_max],
                'y': [upper_y_min, upper_y_max],
            },
        }

    def process(self):
        self.clean_data()
        self.sampling()
        self.evaluate()

    def clean_data(self):
        self._remove_out_range()
        self._remove_outliers()

    def _remove_outliers(self):
        index_list = []
        j = 0
        step = self.data[j, 0] + self.x_window
        for i in range(self.data.shape[0]):
            if i - j >= self.min_num_sample and self.data[i, 0] >= step:
                index_list.append(self.box_outliers_index(self.data[j:i, 1]))
                j = i
                step += self.x_window
                while self.data[i, 0] >= step:
                    step += self.x_window
        index_list.append(self.box_outliers_index(self.data[j:, 1]))
        self.data = self.data[~np.hstack(index_list)]

        clf = ABOD(contamination=self.outliers_fraction)
        clf.fit(self.data)
        y_pred = clf.predict(self.data)
        self.data = self.data[~np.array(y_pred, dtype=np.bool)]

    def _remove_out_range(self):
        # y
        self.data = self.data[self.data[:, 1].argsort()]
        index = np.ones(self.data.shape[0]).astype(bool)
        if self.bounds[2]:
            index = (self.data[:, 1] >= self.y_range[0]) & index
        else:
            index = (self.data[:, 1] > self.y_range[0]) & index
        if self.bounds[3]:
            index = (self.data[:, 1] <= self.y_range[1]) & index
        else:
            index = (self.data[:, 1] < self.y_range[1]) & index
        self.data = self.data[index]
        # x
        # self.data sorted by self.data[:, 0] (sorted by x)
        self.data = self.data[self.data[:, 0].argsort()]
        index = np.ones(self.data.shape[0]).astype(bool)
        if self.bounds[0]:
            index = (self.data[:, 0] >= self.x_range[0]) & index
        else:
            index = (self.data[:, 0] > self.x_range[0]) & index
        if self.bounds[1]:
            index = (self.data[:, 0] <= self.x_range[1]) & index
        else:
            index = (self.data[:, 0] < self.x_range[1]) & index
        self.data = self.data[index]

    @staticmethod
    def __data_check(x, y):
        if len(x) != len(y):
            raise DataError('The dimensions of x and y must match exactly')
        if len(x) == 0:
            raise DataError('The dimension of x can not be zero')
        if len(y) == 0:
            raise DataError('The dimension of y can not be zero')

    @staticmethod
    def box_outliers_index(nums, iqr_coef=1.5):
        q1 = np.percentile(nums, 25, interpolation='nearest')
        q3 = np.percentile(nums, 75, interpolation='nearest')
        iqr = q3 - q1
        index = (nums > q3 + iqr_coef * iqr) | (nums < q1 - iqr_coef * iqr)
        return index

    @staticmethod
    def normal_dist_outliers_index(nums, sigma_coef=2.0):
        mu = nums.mean()
        sigma = nums.std()
        index = (nums > mu + sigma_coef * sigma) | (nums < mu - sigma_coef * sigma)
        return index


def call(*args, **kwargs):
    param = kwargs.get('param')
    if param is None:
        raise ParamError('Missing required parameter in the JSON body: param')
    for p in ['x', 'y', 'x_range', 'y_range']:
        if p not in param.keys():
            raise ParamError('Required parameter \'%s\' not found in \'param\'' % p)

    s = EECurveGSB(
        x=param['x'],
        y=param['y'],
        x_range=param['x_range'],
        y_range=param['y_range'],

        x_window=param.get('x_window', 5.0),
        degree=param.get('degree', 6),
        out_size=param.get('out_size', 100),

        bounds=param.get('bounds', (False, False, False, False)),
        outliers_fraction=param.get('outliers_fraction', 0.005),
        min_num_sample=param.get('min_num_sample', 5),
        bound_scale=param.get('bound_scale', 1.0),
        clamped=param.get('clamped', True),
    )

    s.process()
    return s.get_result()
