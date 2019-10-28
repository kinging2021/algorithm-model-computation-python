import numpy as np
from scipy.interpolate import griddata

from algorithm.exception import ParamError, DataError
from data_handler.api_data import HistoryResult


class EERange2d(object):
    def __init__(self, call_code, x, y):
        self.call_code = call_code
        self.x = x
        self.y = y

    def get_data(self):
        result = HistoryResult(self.call_code).get_result()
        if result['code'] != 200:
            raise DataError('Failed to get history result. call_code: %s, code: %d'
                            % self.call_code, result['code'])
        return result['data']

    def get_range(self):
        data = self.get_data()
        ee_range = {
            'expected': None,
            'upper': None,
            'lower': None,
        }
        xi = [(self.x, self.y)]
        try:
            d = np.array(data['expected'])
            ee_range['expected'] = griddata(d[:, 0:2], d[:, 2], xi, method='linear')[0]
        except:
            pass
        try:
            d = np.array(data['lower'])
            ee_range['lower'] = griddata(d[:, 0:2], d[:, 2], xi, method='linear')[0]
        except:
            pass
        try:
            d = np.array(data['upper'])
            ee_range['upper'] = griddata(d[:, 0:2], d[:, 2], xi, method='linear')[0]
        except:
            pass
        return ee_range


def call(*args, **kwargs):
    param = kwargs.get('param')
    if param is None:
        raise ParamError('Missing required parameter in the JSON body: param')
    for p in ['call_code', 'x', 'y']:
        if p not in param.keys():
            raise ParamError('Required parameter \'%s\' not found in \'param\'' % p)

    s = EERange2d(
        call_code=param['call_code'],
        x=param['x'],
        y=param['y'],
    )

    return s.get_range()
