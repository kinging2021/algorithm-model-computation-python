import pandas as pd
import numpy as np


class ElectricityOffset(object):
    def __init__(self, data: dict):
        self.time = data['time']
        self.load = data['load']
        self.time_price = data['time_price']
        self.price = data['price']

    def optimization(self):
        # 将数据转为Dataframe
        c = {'负荷时间':self.time, '负荷量':self.load}
        d = {'电价时间':self.time_price, '价格':self.price}
        data = pd.DataFrame(c)
        data_price = pd.DataFrame(d)
        data['负荷时间'] = pd.to_datetime(data['负荷时间'])
        data['hour'] = data['负荷时间'].dt.hour
        number_day = data['负荷时间'].dt.day.nunique()

        p = {}
        # 向前平移0-12个小时
        for i in range(12):
            data_offset = data.copy()
            data_offset['hour'] = data_offset['hour'] - i
            data_offset['hour'] = data_offset['hour'].apply(lambda x: 24 + x if x < 0 else x)
            data_total = pd.merge(data_offset, data_price, left_on='hour', right_on='电价时间')
            data_total['expense_hour'] = data_total['负荷量'] * data_total['价格']
            expense = data_total['expense_hour'].sum()
            p[-i] = expense
        # 向后平移1-12个小时
        for i in range(1, 12):
            data_offset = data.copy()
            data_offset['hour'] = data_offset['hour'] + i
            data_offset['hour'] = data_offset['hour'].apply(lambda x: x - 24 if x > 23 else x)
            data_total = pd.merge(data_offset, data_price, left_on='hour', right_on='电价时间')
            data_total['expense_hour'] = data_total['负荷量'] * data_total['价格']
            expense = data_total['expense_hour'].sum()
            p[i] = expense

        expense_series = pd.Series(p)
        expense_diff = expense_series - expense_series[0]
        expense_diff_average = expense_diff / number_day
        print(expense_series)
        return expense_diff_average.to_dict()


def call(*args, **kwargs):
    optimizer = ElectricityOffset(data=kwargs)
    result = optimizer.optimization()
    return result


