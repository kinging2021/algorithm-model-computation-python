# -*- coding: utf-8 -*-

import itertools
import numpy as np

from algorithm.exception import ParamError


class MultiDeviceLoadSearcher1D(object):

    def __init__(self, devices, target_output, schedule_num=5):
        self.devices = devices
        self.target_output = target_output
        self.schedule_num = schedule_num
        self.schedule_list = [None] * self.schedule_num
        self._feasible_list = []

        self.__arg_check()

    def calc_feasible_solution(self):
        rated_sum = 0
        for device in self.devices:
            rated_sum += device['rated_output'] * device['load_max'] / 100
        if rated_sum < self.target_output:
            msg = 'The maximum rated output of all device is less than \'target_output\'. \n'
            for i, device in enumerate(self.devices):
                if device.get('load_max_input'):
                    msg += 'device[%d][\'load_max\'] is %d, ' \
                           'but the max load in device[%d][\'cost_data\'] is %d \n' \
                           % (i, device['load_max_input'], i, device['load_max'])
            raise ParamError(msg)

        for device in self.devices:
            feasible = []
            load_min_calc = self.target_output - (rated_sum - device['rated_output'] * device['load_max'] / 100)
            data = np.array(device['cost_data'])
            if load_min_calc < 0:
                feasible.append([0, 0])
                index = (data[:, 0] >= device['load_min']) & (data[:, 0] <= device['load_max'])
            else:
                index = (data[:, 0] >= max(device['load_min'], load_min_calc / device['rated_output'])) \
                        & (data[:, 0] <= device['load_max'])
            data = data[index]
            data[:, 0] = data[:, 0] * device['rated_output'] / 100
            data[:, 1] *= device.get('cost_coef', 1)
            feasible += data.tolist()

            self._feasible_list.append(feasible)

    def search_schedule(self):
        cost_list = [float('inf')] * self.schedule_num
        for data in itertools.product(*self._feasible_list):
            data = np.array(data)
            output_sum, cost_sum = data.sum(axis=0)
            if output_sum >= self.target_output:
                pos = np.searchsorted(cost_list, cost_sum)
                if pos >= self.schedule_num:
                    continue
                cost_list[pos] = cost_sum
                self.schedule_list[pos] = {
                    'loads': data[:, 0].tolist(),
                    'output_sum': output_sum,
                    'cost_sum': cost_sum,
                }
        for schedule in self.schedule_list:
            for i in range(len(self.devices)):
                schedule['loads'][i] = schedule['loads'][i] / self.devices[i]['rated_output'] * 100

    def get_result(self):
        self.calc_feasible_solution()
        self.search_schedule()

        return self.schedule_list

    def __arg_check(self):
        for device in self.devices:
            data = np.array(device['cost_data'])
            load_max_real = data[:, 0].max()
            if device['load_max'] > load_max_real:
                device['load_max_input'] = device['load_max']
                device['load_max'] = load_max_real


def call(*args, **kwargs):
    # 参数示例
    # param = {
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

    for p in ['target_output', 'devices']:
        if p not in param.keys():
            raise ParamError('Required parameter \'%s\' not found in \'param\'' % p)

    if len(param['devices']) == 0:
        raise ParamError('Parameter \'devices\' is empty.')

    for each in param['devices']:
        for k in ['rated_output', 'load_min', 'load_max', 'cost_data']:
            if k not in each.keys():
                raise ParamError('Required parameter \'%s\' not found in item of \'devices\'' % k)

    s = MultiDeviceLoadSearcher1D(
        devices=param['devices'],
        target_output=param['target_output'],
        schedule_num=param.get('schedule_num', 5),
    )
    return s.get_result()
