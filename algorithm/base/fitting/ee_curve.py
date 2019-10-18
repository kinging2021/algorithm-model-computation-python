import numpy as np
from geomdl import BSpline
from geomdl import knotvector


# EnergyEfficiencyCurve

class EECurve(object):
    def __init__(self,
                 data,
                 x_window,
                 min_num_sample,
                 degree,
                 out_size,
                 clamped=True):

        self.data = data
        self.x_window = x_window
        self.out_size = out_size
        self.degree = degree
        self.clamped = clamped
        self.min_num_sample = min_num_sample
        self.sample_points = None
        self.eval_points = None

    def evaluate(self):
        curve = BSpline.Curve()
        curve.degree = self.degree
        curve.ctrlpts = self.sample_points.tolist()
        curve.knotvector = knotvector.generate(curve.degree, len(curve.ctrlpts),
                                               clamped=self.clamped)
        curve.delta = 1.0 / self.out_size
        curve.evaluate()
        self.eval_points = np.array(curve.evalpts)[:, 0:2]

    def sampling(self):
        j = 0
        step = self.data[j, 0] + self.x_window
        points = []
        for i in range(self.data.shape[0]):
            if i - j >= self.min_num_sample and self.data[i, 0] >= step:
                if self.data.shape[0] - i < self.min_num_sample:
                    points.append(self._select_sample_points(self.data[j:, :]))
                else:
                    points.append(self._select_sample_points(self.data[j:i, :]))
                    j = i
                    step += self.x_window
                    while self.data[i, 0] >= step:
                        step += self.x_window
        self.sample_points = np.vstack(points)

    @staticmethod
    def _select_sample_points(data):
        distance_x = np.square(data[:, 0] - data[:, 0].mean())
        distance_y = np.square(data[:, 1] - data[:, 1].mean())
        index = np.argmin(distance_x + distance_y)
        return data[index, :]

