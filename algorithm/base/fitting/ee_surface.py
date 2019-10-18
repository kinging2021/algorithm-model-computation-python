import numpy as np
from geomdl import BSpline
from geomdl import knotvector
from sklearn.neighbors import KNeighborsRegressor


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
                 clamped=True):

        self.data = data
        self.x_window = x_window
        self.y_window = y_window
        self.out_size = out_size
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.clamped = clamped
        self.min_num_sample = min_num_sample
        self.sample_points = None
        self.eval_points = None
        self.regression_points = None

    def evaluate(self):
        surf = BSpline.Surface()

        ctrl_pts = np.vstack((self.sample_points, self.regression_points))
        ctrl_pts = ctrl_pts[:, [1, 0, 2]]  # [X, Y, Z] -> [Y, X, Z]
        ctrl_pts = ctrl_pts[np.lexsort((ctrl_pts[:, 1], ctrl_pts[:, 0]))]
        ctrl_pts = ctrl_pts.tolist()

        x_range, y_range, _ = np.ptp(self.data, axis=0)
        num_u = int(y_range // self.y_window + 1)
        num_v = int(x_range // self.x_window + 1)
        surf.degree_u = self.degree_y
        surf.degree_v = self.degree_x

        surf.set_ctrlpts(ctrl_pts, num_u, num_v)
        surf.knotvector_u = knotvector.generate(surf.degree_u, num_u, clamped=self.clamped)
        surf.knotvector_v = knotvector.generate(surf.degree_v, num_v, clamped=self.clamped)
        surf.delta = 1.0 / np.sqrt(self.out_size)
        surf.evaluate()
        eval_points = np.array(surf.evalpts)
        self.eval_points = eval_points[:, [1, 0, 2]]  # [Y, X, Z] -> [X, Y, Z]

        # todo: data for web front
        # surf.tessellate()
        # self.faces = surf.tessellator.faces
        # self.vertices = surf.tessellator.vertices
        # from geomdl.visualization.VisPlotly import VisSurface, VisConfig
        # vis_comp = VisSurface(config=VisConfig(ctrlpts=False))
        # surf.vis = vis_comp
        # surf.render()

    def sampling(self):
        points = []
        regression_points = []

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
                if data.shape[0] != 0:
                    tmp_y = y_cursor + self.y_window / 2
                    points.append(self._select_sample_points(data, tmp_y))
                else:
                    regression_points.append([x_cursor + self.x_window / 2,
                                              y_cursor + self.y_window / 2])
                y_cursor += self.y_window
            x_cursor += self.x_window
        self.sample_points = np.vstack(points)
        self.regression_points = np.vstack(regression_points)

    def regressing(self):
        test_in = self.regression_points.copy()

        feature = self.data[:, 0:2].copy()
        value = self.data[:, 2].copy()

        knr_1 = KNeighborsRegressor(weights="distance")
        knr_1.fit(feature[:, 0].reshape(-1, 1), value)
        predict = knr_1.predict(test_in[:, 0].reshape(-1, 1))

        # x_max, y_max, _ = self.data.max(axis=0)
        # x_min, y_min, _ = self.data.min(axis=0)
        # if y_max != y_min:
        #     scaler = (x_max - x_min) / (y_max - y_min)
        #     knr_2 = KNeighborsRegressor(weights="distance")
        #     feature[:, 1] = (feature[:, 1] - y_min) * scaler
        #     knr_2.fit(feature, value)
        #     test_in[:, 1] *= (test_in[:, 1] - y_min) * scaler
        #     predict_2 = knr_2.predict(test_in)
        #     predict = (predict + predict_2) / 2

        tmp = np.vstack((self.regression_points.T, predict)).T
        self.regression_points = tmp

    @staticmethod
    def _select_sample_points(data, y):
        distance_x = np.square(data[:, 0] - data[:, 0].mean())
        # distance_y = np.square(data[:, 1] - data[:, 1].mean())
        distance_z = np.square(data[:, 2] - data[:, 2].mean())
        index = np.argmin(distance_x + distance_z)
        d = data[index, :].copy()
        d[1] = y
        return d
