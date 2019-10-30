import bisect
import itertools
import numpy as np

from algorithm.exception import ParamError


class MultiDeviceLoadSearcher1D(object):
    def __init__(self, devices, target_output, schedule_num=3):
        self.devices = devices
        self.target_output = target_output
        self.schedule_num = schedule_num
        self.schedule_list = [None] * self.schedule_num
        self._feasible_list = []

        self.__arg_check()

    def calc_feasible_solution(self):
        rated_sum = 0
        for boiler in self.devices:
            rated_sum += boiler['rated_output'] * boiler['load_max'] / 100
        if rated_sum < self.target_output:
            raise ParamError('The maximum rated output of all boiler is less than \'target_output\'')

        for boiler in self.devices:
            feasible = []
            output_min = self.target_output - (rated_sum - boiler['rated_output'] * boiler['load_max'] / 100)
            data = np.array(boiler['cost_data'])
            if output_min < 0:
                feasible.append([0, 0])
                index = (data >= boiler['load_min']) & (data <= boiler['load_max'])
            else:
                index = (data >= max(boiler['load_min'], output_min / boiler['rated_output']))\
                        & (data <= boiler['load_max'])
            data = data[index]
            data[:, 0] = data[:, 0] * boiler['rated_output'] / 100
            data[:, 1] *= boiler.get('cost_coef', 1)
            feasible += data.tolist()

            self._feasible_list.append(feasible)

    def search_schedule(self):
        cost_list = [float('inf')] * self.schedule_num
        for data in itertools.product(*self._feasible_list):
            data = np.array(data)
            output_sum, cost_sum = data.sum(axis=0)
            if output_sum >= self.target_output:
                pos = bisect.bisect(cost_list, cost_sum)
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
        pass


def call(*args, **kwargs):
    param = kwargs.get('param')
    if param is None:
        raise ParamError('Missing required parameter in the JSON body: param')

    for p in ['devices', 'target_output']:
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
