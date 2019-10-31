# -*- coding: utf-8 -*-

import itertools
import numpy as np
from scipy.optimize import linprog
from algorithm.exception import ParamError


class MultiDeviceLoadLinprog1D(object):

    def __init__(self, devices, target_output, schedule_num=5):
        self.devices = devices
        self.target_output = target_output
        self.schedule_num = schedule_num
        self.schedule_list = [None] * self.schedule_num
        self._bins = []
        self._rated_outputs = np.array([device['rated_output'] for device in self.devices])

        self.__arg_check()

    def get_result(self):
        self.get_bins()
        self.solve()
        return self.schedule_list

    def get_bins(self):
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

        bins = [0]
        for device in self.devices:
            load_min_calc = self.target_output - (rated_sum - device['rated_output'] * device['load_max'] / 100)
            data = np.array(device['cost_data'])
            if load_min_calc < 0:
                device['required'] = False
                index = (data[:, 0] >= device['load_min']) & (data[:, 0] <= device['load_max'])
            else:
                device['required'] = True
                index = (data[:, 0] >= max(device['load_min'], load_min_calc / device['rated_output'])) \
                        & (data[:, 0] <= device['load_max'])
            data = data[index]
            data[:, 0] /= 100
            data = data[data[:, 0].argsort()]
            device['x'] = data[:, 0]
            device['y'] = data[:, 1]
            bins += device['x'].tolist()
        self._bins = list(set(bins))
        self._bins.sort()

    def solve(self):
        cost_list = [float('inf')] * self.schedule_num

        for bins_right in itertools.product(self._bins, repeat=len(self.devices)):
            if not self.__check_feasible_region(bins_right):
                continue
            output_sum = (np.array(bins_right) * self._rated_outputs).sum()
            if output_sum < self.target_output:
                continue

            kwargs, bias, unused_device = self._get_linprog_param(bins_right)
            result = linprog(**kwargs)
            if result.success:
                cost_sum = result.fun + bias
                pos = np.searchsorted(cost_list, cost_sum)
                if pos >= self.schedule_num:
                    continue
                cost_list[pos] = cost_sum
                loads = list(result.x)
                for i in unused_device:
                    loads.insert(i, 0)
                self.schedule_list[pos] = {
                    'loads': loads,
                    'output_sum': output_sum,
                    'cost_sum': cost_sum,
                }

    def _get_linprog_param(self, bins_right):
        unused_device = []
        # 成本方程 c := [a1, a2, a3]
        #         bias := sum(['b1', 'b2', 'b3'])
        c = []
        bias = 0
        # 边界约束 bounds := [[x0_min, x0_max], [x1_min, x1_max], [x2_min, x2_max]]
        bounds = []
        # 产量约束 A_ub := [[rated_output_1, rated_output_2, rated_output_3]]
        #         b_ub := [target_output]
        A_ub = [[]]
        b_ub = [self.target_output]
        for i, bin_right in enumerate(bins_right):
            if bin_right == 0:
                unused_device.append(i)
                continue
            pos = np.searchsorted(self.devices[i]['x'], bin_right)
            x1 = self.devices[i]['x'][pos - 1]
            x2 = self.devices[i]['x'][pos]
            y1 = self.devices[i]['y'][pos - 1]
            y2 = self.devices[i]['y'][pos]
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            c.append(k)
            bias += b
            bin_left = self._bins[self._bins.index(bins_right) - 1]
            bounds.append((bin_left, bin_right))
            A_ub[0].append(self.devices[i]['rated_output'])
        kwargs = {
            'c': c,
            'A_ub': A_ub,
            'b_ub': b_ub,
            'bounds': bounds,
        }
        return kwargs, bias, unused_device

    def __check_feasible_region(self, bins_right):
        for i, bin_ in enumerate(bins_right):
            if bin_ == 0:
                if self.devices[i]['required']:
                    return False
                else:
                    return True
            if bin_ <= self.devices[i]['load_min']:
                return False

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

    s = MultiDeviceLoadLinprog1D(
        devices=param['devices'],
        target_output=param['target_output'],
        schedule_num=param.get('schedule_num', 5),
    )
    return s.get_result()
