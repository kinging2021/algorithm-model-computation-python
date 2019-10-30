import numpy as np
from geomdl import BSpline
from geomdl import knotvector
from geomdl.exceptions import GeomdlException
from scipy.interpolate import interp1d

from .ee_curve import EECurve
from algorithm.exception import ParamError


# EnergyEfficiencySurface

class EESurface(object):
    def __init__(self,
                 data,
                 x_window,
                 y_window,
                 min_num_sample,
                 degree_x,
                 degree_y,
                 out_size,
                 bound_scale=1.0,
                 clamped=True):

        self.data = data
        self.x_window = x_window
        self.y_window = y_window
        self.out_size = out_size
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.bound_scale = bound_scale
        self.clamped = clamped
        self.min_num_sample = min_num_sample

        self.regression_pos = None

        self.sample_points_expected = None
        self.regression_points_expected = None
        self.eval_points_expected = None

        self.sample_points_lower = None
        self.regression_points_lower = None
        self.eval_points_lower = None

        self.sample_points_upper = None
        self.regression_points_upper = None
        self.eval_points_upper = None

        self.__arg_check()

    def evaluate(self):

        def get_evalpts(ctrl_pts):
            surf = BSpline.Surface()
            surf.degree_u = self.degree_y
            surf.degree_v = self.degree_x

            ctrl_pts = ctrl_pts[:, [1, 0, 2]]  # [X, Y, Z] -> [Y, X, Z]
            ctrl_pts = ctrl_pts[np.lexsort((ctrl_pts[:, 1], ctrl_pts[:, 0]))]

            num_u = np.unique(ctrl_pts[:, 0]).shape[0]
            num_v = int(ctrl_pts.shape[0] / num_u)

            try:
                surf.set_ctrlpts(ctrl_pts.tolist(), num_u, num_v)
            except GeomdlException:
                raise ParamError('The input data is not enough or x_window and y_window is too wide, '
                                 'please check input data, x_window, y_window and min_num_sample')
            surf.knotvector_u = knotvector.generate(surf.degree_u, num_u, clamped=self.clamped)
            surf.knotvector_v = knotvector.generate(surf.degree_v, num_v, clamped=self.clamped)
            surf.delta = 1.0 / np.sqrt(self.out_size)
            surf.evaluate()
            eval_points = np.array(surf.evalpts)
            eval_points = eval_points[:, [1, 0, 2]]  # [Y, X, Z] -> [X, Y, Z]

            return eval_points

        ctrl_pts = np.vstack((self.sample_points_expected, self.regression_points_expected))
        self.eval_points_expected = get_evalpts(ctrl_pts)

        ctrl_pts = np.vstack((self.sample_points_lower, self.regression_points_lower))
        self.eval_points_lower = get_evalpts(ctrl_pts)

        ctrl_pts = np.vstack((self.sample_points_upper, self.regression_points_upper))
        self.eval_points_upper = get_evalpts(ctrl_pts)

    def sampling(self):
        regression_pos = []
        points_expected = []
        points_lower = []
        points_upper = []
        x = self.data[:, 0]
        x_cursor = x.min()
        x_max = x.max()
        y = self.data[:, 1]
        y_min = y.min()
        y_max = y.max()
        while x_cursor <= x_max:
            data_ = self.data[(x >= x_cursor) & (x < x_cursor + self.x_window)]
            y_ = data_[:, 1]
            y_cursor = y_min
            while y_cursor <= y_max:
                data = data_[(y_ >= y_cursor) & (y_ < y_cursor + self.y_window)]
                if data.shape[0] >= self.min_num_sample:
                    p_expected, p_lower, p_upper = self._get_sample_points(data)

                    tmp_y = y_cursor + self.y_window / 2
                    p_expected[1] = tmp_y
                    p_lower[1] = tmp_y
                    p_upper[1] = tmp_y

                    points_expected.append(p_expected)
                    points_lower.append(p_lower)
                    points_upper.append(p_upper)
                else:
                    regression_pos.append([x_cursor + self.x_window / 2,
                                           y_cursor + self.y_window / 2])
                y_cursor += self.y_window
            x_cursor += self.x_window

        self.regression_pos = np.vstack(regression_pos)
        self.sample_points_lower = np.vstack(points_lower)
        self.sample_points_upper = np.vstack(points_upper)
        self.sample_points_expected = np.vstack(points_expected)

    def _get_sample_points(self, data):
        distance_x = np.square(data[:, 0] - data[:, 0].mean())
        # distance_y = np.square(data[:, 1] - data[:, 1].mean())
        distance_z = np.square(data[:, 2] - data[:, 2].mean())
        index = np.argmin(distance_x + distance_z)
        p_expected = data[index, :].copy()

        z = data[:, 2]
        z_min = z.min()
        z_max = z.max()
        q1 = np.percentile(z, 25, interpolation='nearest')
        q3 = np.percentile(z, 75, interpolation='nearest')
        q2 = np.percentile(z, 50, interpolation='nearest')
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        x, y = data[np.where(z == q2)[0][0], 0:2]
        if upper >= z_max:
            p_upper = data[np.where(z == z_max)[0][0]].copy()
        else:
            p_upper = np.array([x, y, upper])

        if lower <= z_min:
            p_lower = data[np.where(z == z_min)[0][0]].copy()
        else:
            p_lower = np.array([x, y, lower])

        delta = self.bound_scale * (p_lower[2] - q2)
        p_lower[2] = q2 + delta

        delta = self.bound_scale * (p_upper[2] - q2)
        p_upper[2] = q2 + delta

        return p_expected, p_lower, p_upper

    def regressing(self):
        r = EECurve(self.data[:, [0, 2]], self.x_window, self.min_num_sample,
                    self.degree_x, self.out_size)
        r.sampling()
        r.evaluate()

        # expected
        x = r.eval_points_expected[:, 0]
        y = r.eval_points_expected[:, 1]
        r_expected = interp1d(x, y)
        index = (self.regression_pos[:, 0] >= x[0]) & (self.regression_pos[:, 0] <= x[-1])
        pos = self.regression_pos[index]
        z = r_expected(pos[:, 0])
        self.regression_points_expected = np.vstack((pos.T, z)).T
        index = (self.sample_points_expected[:, 0] >= x[0]) & (self.sample_points_expected[:, 0] <= x[-1])
        self.sample_points_expected = self.sample_points_expected[index]

        # lower
        x = r.eval_points_lower[:, 0]
        y = r.eval_points_lower[:, 1]
        r_lower = interp1d(x, y)
        index = (self.regression_pos[:, 0] >= x[0]) & (self.regression_pos[:, 0] <= x[-1])
        pos = self.regression_pos[index]
        z = r_lower(pos[:, 0])
        self.regression_points_lower = np.vstack((pos.T, z)).T
        index = (self.sample_points_lower[:, 0] >= x[0]) & (self.sample_points_lower[:, 0] <= x[-1])
        self.sample_points_lower = self.sample_points_lower[index]

        # upper
        x = r.eval_points_upper[:, 0]
        y = r.eval_points_upper[:, 1]
        r_upper = interp1d(x, y)
        index = (self.regression_pos[:, 0] >= x[0]) & (self.regression_pos[:, 0] <= x[-1])
        pos = self.regression_pos[index]
        z = r_upper(pos[:, 0])
        self.regression_points_upper = np.vstack((pos.T, z)).T
        index = (self.sample_points_upper[:, 0] >= x[0]) & (self.sample_points_upper[:, 0] <= x[-1])
        self.sample_points_upper = self.sample_points_upper[index]

    def __arg_check(self):
        if self.degree_x < 3:
            raise ParamError('Parameter \'degree_x\' should be at least 3')
        if self.degree_y < 3:
            raise ParamError('Parameter \'degree_y\' should be at least 3')
        if self.min_num_sample < 1:
            raise ParamError('Parameter \'min_num_sample\' should be at least 1')
