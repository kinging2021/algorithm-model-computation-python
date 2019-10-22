import numpy as np
from geomdl import BSpline
from geomdl import knotvector
from sklearn.neighbors import KNeighborsRegressor


# Energy efficiency bound surface

class EEBoundSurface(object):
    def __init__(self,
                 data,
                 x_window,
                 y_window,
                 min_num_sample,
                 degree_x,
                 degree_y,
                 out_size,
                 scale=1.0,
                 clamped=True):

        self.data = data
        self.x_window = x_window
        self.y_window = y_window
        self.out_size = out_size
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.scale = scale
        self.clamped = clamped
        self.min_num_sample = min_num_sample
        self.sample_points_min = None  # [X, Y, X, delta_Z]
        self.sample_points_max = None  # [X, Y, X, delta_Z]
        self.regression_points_min = None
        self.regression_points_max = None
        self.eval_points_min = None
        self.eval_points_max = None
        self.__regression_pos = None

    def evaluate(self):
        x_range, y_range, _ = np.ptp(self.data, axis=0)
        num_y = int(y_range // self.y_window + 1)
        num_x = int(x_range // self.x_window + 1)

        def get_evalpts(ctrl_pts):
            #
            x_max, y_max, _ = ctrl_pts.max(axis=0)
            x_min, y_min, _ = ctrl_pts.min(axis=0)
            scaler = (x_max - x_min) / (y_max - y_min)
            ctrl_pts[:, 1] = (ctrl_pts[:, 1] - y_min) * scaler
            #

            surf = BSpline.Surface()
            ctrl_pts = ctrl_pts[:, [1, 0, 2]]  # [X, Y, Z] -> [Y, X, Z]
            ctrl_pts = ctrl_pts[np.lexsort((ctrl_pts[:, 1], ctrl_pts[:, 0]))]
            ctrl_pts = ctrl_pts.tolist()

            surf.degree_u = self.degree_y
            surf.degree_v = self.degree_x

            surf.set_ctrlpts(ctrl_pts, num_y, num_x)
            surf.knotvector_u = knotvector.generate(surf.degree_u, num_y, clamped=self.clamped)
            surf.knotvector_v = knotvector.generate(surf.degree_v, num_x, clamped=self.clamped)
            surf.delta = 1.0 / np.sqrt(self.out_size)
            surf.evaluate()
            eval_points = np.array(surf.evalpts)
            eval_points = eval_points[:, [1, 0, 2]]  # [Y, X, Z] -> [X, Y, Z]

            # todo: data for web front
            # surf.tessellate()
            # self.faces = surf.tessellator.faces
            # self.vertices = surf.tessellator.vertices
            # from geomdl.visualization.VisPlotly import VisSurface, VisConfig
            # vis_comp = VisSurface(config=VisConfig(ctrlpts=True))
            # surf.vis = vis_comp
            # surf.render()

            return eval_points

        ctrl_pts = np.vstack((self.sample_points_min, self.regression_points_min))[:, 0:3]
        self.eval_points_min = get_evalpts(ctrl_pts)
        ctrl_pts = np.vstack((self.sample_points_max, self.regression_points_max))[:, 0:3]
        self.eval_points_max = get_evalpts(ctrl_pts)

    def sampling(self):
        points_min = []
        points_max = []
        regression_pos = []

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
                    tmp_y = y_cursor + self.y_window / 2
                    p_min, p_max = self._get_min_max_points(data, tmp_y)
                    points_min.append(p_min)
                    points_max.append(p_max)
                else:
                    regression_pos.append([x_cursor + self.x_window / 2,
                                           y_cursor + self.y_window / 2])
                y_cursor += self.y_window
            x_cursor += self.x_window

        self.sample_points_min = np.vstack(points_min)
        self.sample_points_max = np.vstack(points_max)
        self.__regression_pos = np.vstack(regression_pos)

    def regressing(self):

        feature = self.data[:, 0:2].copy()
        value = self.data[:, 2].copy()

        knr_1 = KNeighborsRegressor(weights="distance")
        knr_1.fit(feature[:, 0].reshape(-1, 1), value)

        # sort data by X
        self.sample_points_min = self.sample_points_min[self.sample_points_min[:, 0].argsort()]
        self.sample_points_max = self.sample_points_max[self.sample_points_max[:, 0].argsort()]

        def get_delta_z(data, y, x):
            # data := [X, Y, Z], and data should be sorted by X
            d = data[data[:, 1] == y]
            if d.size == 0:
                return None
            index_less = np.where(d[:, 0] <= x)[0]
            if index_less.size == 0:
                return d[0][3]
            elif index_less.size == d.shape[0]:
                return d[-1][3]
            else:
                return (d[index_less[-1]][3] + d[index_less[-1] + 1][3]) / 2

        def get_mean(n1, n2):
            if n1 is None and n2 is None:
                return 0
            if n1 is None:
                return n2
            elif n2 is None:
                return n1
            else:
                return (n1 + n2) / 2

        points_min = []
        points_max = []
        for i in range(self.__regression_pos.shape[0]):
            pred_z = knr_1.predict(self.__regression_pos[i][0].reshape(-1, 1))

            y_left = self.__regression_pos[i, 1] - self.y_window
            y_right = self.__regression_pos[i, 1] + self.y_window

            z_left = get_delta_z(self.sample_points_min, y_left, self.__regression_pos[i, 0])
            z_right = get_delta_z(self.sample_points_min, y_right, self.__regression_pos[i, 0])
            delta_z = get_mean(z_left, z_right)
            points_min.append(np.hstack((self.__regression_pos[i], pred_z + delta_z, delta_z)))

            z_left = get_delta_z(self.sample_points_max, y_left, self.__regression_pos[i, 0])
            z_right = get_delta_z(self.sample_points_max, y_right, self.__regression_pos[i, 0])
            delta_z = get_mean(z_left, z_right)
            points_max.append(np.hstack((self.__regression_pos[i], pred_z + delta_z, delta_z)))

        self.regression_points_min = np.vstack(points_min)
        self.regression_points_max = np.vstack(points_max)

    def _get_min_max_points(self, data, y):
        z = data[:, 2]
        z_min = z.min()
        z_max = z.max()
        q1 = np.percentile(z, 25, interpolation='nearest')
        q3 = np.percentile(z, 75, interpolation='nearest')
        q2 = np.percentile(z, 50, interpolation='nearest')
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        x = data[np.where(z == q2)[0][0], 0]
        if upper >= z_max:
            p_max = data[np.where(z == z_max)[0][0]].copy()
            p_max[1] = y
        else:
            p_max = np.array([x, y, upper])

        if lower <= z_min:
            p_min = data[np.where(z == z_min)[0][0]].copy()
            p_min[1] = y
        else:
            p_min = np.array([x, y, lower])

        delta = self.scale * (p_min[2] - q2)
        p_min[2] = q2 + delta
        p_min = np.hstack((p_min, delta))

        delta = self.scale * (p_max[2] - q2)
        p_max[2] = q2 + delta
        p_max = np.hstack((p_max, delta))

        return p_min, p_max
