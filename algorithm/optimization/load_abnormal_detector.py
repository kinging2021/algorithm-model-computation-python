import pandas as pd
import numpy as np


class LoadAbnormalDetection(object):
    def __init__(self, data: dict):
        # 此处load为负载率的时序数据，模型每天运行一次，检测数据的死值和波动异常
        self.time = data['time']
        self.load = data['load']

    def abnormal_detector(self):
        data = pd.DataFrame(self.time, self.load, columns=['time', 'load'])
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time').sort_index()
        data = data.diff()
        # 设定死值的阈值p1
        p1 = 5
        if data[data == 0].count() > p1:
            print('数据有死值')
            return 1
        # 设定数据波动的阈值p2
        p2 = 10
        if data[np.abs(data) > p2].count() >10:
            print('负荷波动过于剧烈')
            return 1


def call(*args, **kwargs):
    detector = LoadAbnormalDetection(data=kwargs)
    detector.abnormal_detector()