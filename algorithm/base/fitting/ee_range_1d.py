import numpy as np
from scipy.interpolate import interp1d

from algorithm.exception import ParamError, DataError
from data_handler.api_data import HistoryResult


class EERange1d(object):
    def __init__(self, call_code, x):
        self.call_code = call_code
        self.x = x

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
        try:
            d = np.array(data['expected'])
            func = interp1d(d[:, 0], d[:, 1])
            ee_range['expected'] = func(self.x)
        except:
            pass
        try:
            d = np.array(data['lower'])
            func = interp1d(d[:, 0], d[:, 1])
            ee_range['lower'] = func(self.x)
        except:
            pass
        try:
            d = np.array(data['upper'])
            func = interp1d(d[:, 0], d[:, 1])
            ee_range['upper'] = func(self.x)
        except:
            pass
        return ee_range


def call(*args, **kwargs):
    param = kwargs.get('param')
    if param is None:
        raise ParamError('Missing required parameter in the JSON body: param')
    for p in ['call_code', 'x']:
        if p not in param.keys():
            raise ParamError('Required parameter \'%s\' not found in \'param\'' % p)

    s = EERange1d(
        call_code=param['call_code'],
        x=param['x'],
    )

    return s.get_range()
