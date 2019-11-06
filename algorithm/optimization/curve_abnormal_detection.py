import datetime
import pandas as pd
import numpy as np
import algorithm.exception


class AbnormalDetection(object):
    def __init__(self, data: dict):
        self.load = data['load']
        self.production = data['production']

    def abnormal_detector(self):
        data = pd.DataFrame(self.load, self.production, columns=['load', 'production'])
        data_diff = data.diff()
        data_diff['slope'] = data_diff['production']/data_diff['load']
        # 设定斜率过大的阈值p1
        p1 = 3
        if data_diff[data_diff['slope'] > p1].count() > p1:
            return algorithm.exception.DataError()
        # 计算斜率之间的差值，正常曲线斜率差值应该大于等于0
        slope_diff = data_diff['slope'].diff()
        if slope_diff[slope_diff < 0.1].count() > 0:
            return algorithm.exception.DataError()


def call(*args, **kwargs):
    detector = AbnormalDetection(data=kwargs)
    detector.abnormal_detector()
