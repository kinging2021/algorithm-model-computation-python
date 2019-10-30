import numpy as np
from pyod.models.abod import ABOD
from .ee_surface import EESurface

from algorithm.exception import ParamError, DataError


class EESurfaceGSB(EESurface):
    def __init__(self,
                 x,
                 y,
                 z,
                 x_range,
                 y_range,
                 z_range,
                 bounds=(False, False, False, False, False, False),
                 outliers_fraction=0.005,
                 x_window=5.0,
                 y_window=1.0,
                 min_num_sample=5,
                 degree_x=6,
                 degree_y=3,
                 out_size=1000,
                 bound_scale=1.0,
                 clamped=True):

        self.__data_check(x, y, z)
        super(EESurfaceGSB, self).__init__(
            np.vstack((x, y, z)).T, x_window, y_window,
            min_num_sample, degree_x, degree_y,
            out_size, bound_scale, clamped
        )

        self.x = x
        self.y = y
        self.z = z
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.bounds = bounds
        self.outliers_fraction = outliers_fraction

    def get_result(self):
        expected_x_min, expected_y_min, expected_z_min = self.eval_points_expected.min(axis=0)
        expected_x_max, expected_y_max, expected_z_max = self.eval_points_expected.max(axis=0)
        lower_x_min, lower_y_min, lower_z_min = self.eval_points_lower.min(axis=0)
        lower_x_max, lower_y_max, lower_z_max = self.eval_points_lower.max(axis=0)
        upper_x_min, upper_y_min, upper_z_min = self.eval_points_upper.min(axis=0)
        upper_x_max, upper_y_max, upper_z_max = self.eval_points_upper.max(axis=0)
        return {
            'expected': self.eval_points_expected.tolist(),
            'expected_range': {
                'x': [expected_x_min, expected_x_max],
                'y': [expected_y_min, expected_y_max],
                'z': [expected_z_min, expected_z_max],
            },
            'lower': self.eval_points_lower.tolist(),
            'lower_range': {
                'x': [lower_x_min, lower_x_max],
                'y': [lower_y_min, lower_y_max],
                'z': [lower_z_min, lower_z_max],
            },
            'upper': self.eval_points_upper.tolist(),
            'upper_range': {
                'x': [upper_x_min, upper_x_max],
                'y': [upper_y_min, upper_y_max],
                'z': [upper_z_min, upper_z_max],
            },
        }

    def process(self):
        self.clean_data()
        self.sampling()
        self.regressing()
        self.evaluate()

    def clean_data(self):
        self._remove_out_range()
        self._remove_outliers()

    def _remove_outliers(self):
        data_list = []
        x = self.data[:, 0]
        x_cursor = x.min()
        x_max = x.max()
        while x_cursor <= x_max:
            data_ = self.data[(x >= x_cursor) & (x < x_cursor + self.x_window)]
            y_ = data_[:, 1]
            y_cursor = y_.min()
            y_max = y_.max()
            while y_cursor <= y_max:
                data = data_[(y_ >= y_cursor) & (y_ < y_cursor + self.y_window)]
                if data.shape[0] != 0:
                    index = ~self.box_outliers_index(data[:, 2])
                    data_list.append(data[index])
                y_cursor += self.y_window
            x_cursor += self.x_window
        self.data = np.vstack(data_list)

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
        # z
        self.data = self.data[self.data[:, 2].argsort()]
        index = np.ones(self.data.shape[0]).astype(bool)
        if self.bounds[4]:
            index = (self.data[:, 2] >= self.z_range[0]) & index
        else:
            index = (self.data[:, 2] > self.z_range[0]) & index
        if self.bounds[5]:
            index = (self.data[:, 2] <= self.z_range[1]) & index
        else:
            index = (self.data[:, 2] < self.z_range[1]) & index
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
    def __data_check(x, y, z):
        if len(x) != len(y) or len(y) != len(z):
            raise DataError('The dimensions of x, y and z must match exactly')
        if len(x) == 0:
            raise DataError('The dimension of x can not be zero')
        if len(y) == 0:
            raise DataError('The dimension of y can not be zero')
        if len(z) == 0:
            raise DataError('The dimension of z can not be zero')

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
    for p in ['x', 'y', 'z', 'x_range', 'y_range', 'z_range']:
        if p not in param.keys():
            raise ParamError('Required parameter \'%s\' not found in \'param\'' % p)

    s = EESurfaceGSB(
        x=param['x'],
        y=param['y'],
        z=param['z'],
        x_range=param['x_range'],
        y_range=param['y_range'],
        z_range=param['z_range'],

        x_window=param.get('x_window', 5.0),
        y_window=param.get('y_window', 1.0),
        degree_x=param.get('degree_x', 6),
        degree_y=param.get('degree_y', 3),
        out_size=param.get('out_size', 1000),

        bounds=param.get('bounds', (False, False, False, False, False, False)),
        outliers_fraction=param.get('outliers_fraction', 0.005),
        min_num_sample=param.get('min_num_sample', 5),
        bound_scale=param.get('bound_scale', 1.0),
        clamped=param.get('clamped', True)
    )

    s.process()
    return s.get_result()
