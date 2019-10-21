import numpy as np
from geomdl import BSpline
from geomdl import knotvector


# Energy efficiency bound curve
class EEBoundCurve(object):

    def __init__(self,
                 data,
                 x_window,
                 min_num_sample,
                 degree,
                 out_size,
                 iqr_coef=1.8,
                 clamped=True):

        self.data = data
        self.x_window = x_window
        self.degree = degree
        self.out_size = out_size
        self.iqr_coef = iqr_coef
        self.clamped = clamped
        self.min_num_sample = min_num_sample

        self.sample_points_min = None
        self.sample_points_max = None
        self.eval_points_min = None
        self.eval_points_max = None

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

        self.eval_points_min = get_evalpts(self.sample_points_min)
        self.eval_points_max = get_evalpts(self.sample_points_max)

    def sampling(self):
        j = 0
        step = self.data[j, 0] + self.x_window
        points_min = []
        points_max = []
        for i in range(self.data.shape[0]):
            if self.data[i, 0] >= step:
                if self.data.shape[0] - i < self.min_num_sample:
                    p_min, p_max = self._get_min_max_points(self.data[j:, :])
                else:
                    p_min, p_max = self._get_min_max_points(self.data[j:i, :])
                    j = i
                    step += self.x_window
                    while self.data[i, 0] >= step:
                        step += self.x_window
                points_min.append(p_min)
                points_max.append(p_max)

        if self.data[0, 0] < points_max[0][0, 0]:
            data = self.data[self.data[:, 0] < points_max[0][0, 0]]
            y = np.max([data[:, 1].max(), points_max[0][0, 1]])
            points_max[0] = [self.data[0, 0], y]

        if self.data[0, 0] < points_min[0][0, 0]:
            data = self.data[self.data[:, 0] < points_min[0][0, 0]]
            y = np.min([data[:, 1].min(), points_min[0][0, 1]])
            points_min[0] = [self.data[0, 0], y]

        if self.data[-1, 0] > points_max[-1][0, 0]:
            data = self.data[self.data[:, 0] > points_max[-1][0, 0]]
            y = np.max([data[:, 1].max(), points_max[-1][0, 1]])
            points_max[-1] = [self.data[-1, 0], y]

        if self.data[-1, 0] > points_min[-1][0, 0]:
            data = self.data[self.data[:, 0] > points_min[-1][0, 0]]
            y = np.min([data[:, 1].min(), points_min[-1][0, 1]])
            points_min[-1] = [self.data[-1, 0], y]

        self.sample_points_min = np.vstack(points_min)
        self.sample_points_max = np.vstack(points_max)

    def _get_min_max_points(self, data):
        y = data[:, 1]
        y_min = y.min()
        y_max = y.max()
        q1 = np.percentile(y, 25, interpolation='nearest')
        q3 = np.percentile(y, 75, interpolation='nearest')
        tmp = np.percentile(y, 50, interpolation='nearest')
        x = data[np.where(y == tmp)[0][0], 0]
        iqr = q3 - q1
        upper = q3 + self.iqr_coef * iqr
        lower = q1 - self.iqr_coef * iqr
        if upper >= y_max:
            p_max = data[np.where(y == y_max)[0][0]]
        else:
            p_max = np.array([x, upper])

        if lower <= y_min:
            p_min = data[np.where(y == y_min)[0][0]]
        else:
            p_min = np.array([x, lower])

        return p_min, p_max
