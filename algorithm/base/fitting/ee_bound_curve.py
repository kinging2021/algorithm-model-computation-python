import numpy as np
from geomdl import BSpline
from geomdl import knotvector


# Energy efficiency bound curve
class EEBoundCurve(object):

    def __init__(self, data, x_window, min_num_sample, x_interval,
                 degree, out_size, clamped=True):
        self.data = data
        self.x_window = x_window
        self.out_size = out_size
        self.degree = degree
        self.clamped = clamped
        self.min_num_sample = min_num_sample
        self.x_interval = x_interval

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

    def sampling_min_max(self):
        # x_interval = self.x_window // ((self.x_range[1] - self.x_range[0]) / 100)
        j = 0
        step = self.data[j, 0] + self.x_window
        points_min = []
        points_max = []
        for i in range(self.data.shape[0]):
            if i - j >= self.min_num_sample and self.data[i, 0] >= step:
                p_min, p_max = self._get_min_max_points(self.data[j:i, :], self.x_interval)
                points_min.append(p_min)
                points_max.append(p_max)
                j = i
                step += self.x_window
                while self.data[i, 0] >= step:
                    step += self.x_window
        p_min, p_max = self._get_min_max_points(self.data[j:, :], self.x_interval)
        points_min.append(p_min)
        points_max.append(p_max)

        self.sample_points_min = np.vstack(points_min)
        self.sample_points_max = np.vstack(points_max)

    @staticmethod
    def _get_min_max_points(data, x_interval):
        x = data[:, 0]
        sections = [int(x.min())]
        while sections[-1] <= x.max():
            sections.append(sections[-1] + x_interval)
        sections = sections[1:-1]
        points_min = []
        points_max = []
        for each in np.split(data, sections):
            points_min.append(each[np.argmin(each[:, 1])])
            points_max.append(each[np.argmax(each[:, 1])])
        p_min = np.vstack(points_min).mean(axis=0)
        p_max = np.vstack(points_max).mean(axis=0)
        return p_min, p_max
