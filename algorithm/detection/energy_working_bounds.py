import datetime

import numpy as np

from common.model_loader import ModelLoader
from data_handler.api_data import jinmailang
from algorithm.exception import ParamError, DataError
from algorithm.base.cluster.energy_status_cluster import EnergyStatusCluster


class EnergyWorkingBounds(object):
    def __init__(self, asset):
        self.model = asset
        return

    def calculate(self, date):
        model = self.model
        days_len = model["sample_days_length"]
        lower = model["lower_percentile"]
        upper = model["upper_percentile"]
        target_date = date - datetime.timedelta(days=days_len)
        data = jinmailang.get_jinmailang_df(target_date, days_len)
        bounds = {}
        for column in model["columns"]:
            energy_cluster = EnergyStatusCluster(data,
                                                 column,
                                                 model["working_threshold"],
                                                 0,
                                                 0)
            energy_cluster.analyze()
            column_data = data[data["tag"] == 1][column]
            ret = np.percentile(column_data, [lower, upper])
            bounds[column] = list(ret)
        return bounds


def call(param: dict, model_url: str):
    if param is None:
        raise ParamError('Missing required parameter in the JSON body: param')
    date = param.get('date')

    if not date:
        raise ParamError('Required parameter \'date\' not found in \'param\'')

    asset = ModelLoader.load_file(model_url)
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    bound_detector = EnergyWorkingBounds(asset)
    bounds = bound_detector.calculate(date)
    result = {"energy_bounds": bounds}

    return result
