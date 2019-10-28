import numpy as np

from algorithm.exception import ParamError, DataError
from utils.tool import get_arg


class EEHistogram(object):
    def __init__(self,
                 x,
                 bins,
                 x_range=None,
                 density=True):

        self.x = np.array(x)
        self.bins = bins
        self.x_range = get_arg(x_range, (self.x.min(), self.x.max()))
        self.density = get_arg(density, True)

    def get_result(self):
        result = np.histogram(self.x, bins=self.bins,
                              range=self.x_range, density=self.density)
        return {
            'hist': result[0],
            'bin_edges': result[1],
        }


def call(*args, **kwargs):
    param = kwargs.get('param')
    if param is None:
        raise ParamError('Missing required parameter in the JSON body: param')
    for p in ['x', 'x_range', 'bins']:
        if p not in param.keys():
            raise ParamError('Required parameter \'%s\' not found in \'param\'' % p)

    s = EEHistogram(
        x=param['x'],
        bins=param['bins'],
        x_range=param.get('x_range'),
        density=param.get('density'),
    )

    return s.get_result()
