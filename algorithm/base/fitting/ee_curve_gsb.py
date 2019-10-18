import numpy as np
from pyod.models.abod import ABOD
from .ee_curve import EECurve

from algorithm.exception import ParamError, DataError


class EECurveGSB(EECurve):
    def __init__(self, x,
                 y,
                 x_range,
                 y_range,
                 bounds=None,
                 outliers_fraction=0.005,
                 x_window=5,
                 min_num_sample=5,
                 degree=6,
                 out_size=100,
                 clamped=True):

        self.__data_check(x, y)
        super(EECurveGSB, self).__init__(
            np.vstack((x, y)).T, x_window, min_num_sample, degree, out_size, clamped
        )

        self.x = x
        self.y = y
        self.x_range = x_range
        self.y_range = y_range
        self.bounds = bounds
        self.outliers_fraction = outliers_fraction
        self.__reset_default()

    def get_result(self):
        return {
            'x': self.eval_points[:, 0].tolist(),
            'y': self.eval_points[:, 1].tolist(),
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

    def __reset_default(self):
        if self.bounds is None:
            self.bounds = (False, False, False, False)
        if self.min_num_sample is None:
            self.min_num_sample = 5
        if self.degree is None:
            self.degree = 6
        if self.out_size is None:
            self.out_size = 100
        if self.x_window is None:
            self.x_window = 5
        if self.clamped is None:
            self.clamped = True
        if self.outliers_fraction is None:
            self.outliers_fraction = 0.005

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

        x_window=param.get('x_window'),
        degree=param.get('degree'),
        out_size=param.get('out_size'),

        bounds=param.get('bounds'),
        outliers_fraction=param.get('outliers_fraction'),
        min_num_sample=param.get('min_num_sample'),
        clamped=param.get('clamped')
    )

    s.process()
    return s.get_result()
