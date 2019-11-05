# -*- coding: utf-8 -*-
from .multi_device_load_linprog_1d import call as call_linprog
from .multi_device_load_searcher_1d import call as call_search
from algorithm.exception import ParamError


def call(*args, **kwargs):
    # 参数示例
    # param = {
    #     'method': '' # 方法名, linprog | search
    #     'target_output': 24,
    #     'devices': [
    #         {  # device 1
    #             'rated_output': 8,
    #             'load_min': 30,
    #             'load_max': 85,
    #             'cost_data': []  # 能效曲线散点
    #         },
    #         {  # device 2
    #             'rated_output': 6,
    #             'load_min': 35,
    #             'load_max': 85,
    #             'cost_data': []  # 能效曲线散点
    #         }
    #     ],
    # }
    param = kwargs.get('param')
    if param is None:
        raise ParamError('Missing required parameter in the JSON body: param')

    method = param.get('method')
    if method is None:
        raise ParamError('Required parameter \'method\' not found in \'param\'')
    elif method == 'search':
        return call_search(param=param)
    elif method == 'linprog':
        return call_linprog(param=param)
    else:
        raise ParamError('Unknown method: \'%s\'' % method)

