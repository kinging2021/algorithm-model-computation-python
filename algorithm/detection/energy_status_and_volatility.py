import datetime
# from typing import Dict, Any

import numpy as np

from common.model_loader import ModelLoader
from algorithm.exception import ParamError, DataError
from algorithm.base.cluster.energy_status_cluster import EnergyStatusCluster


class EnergyStatus(object):
    uptime_str = "up_time"
    downtime_str = "down_time"
    down_hour_str = "down_hours"
    down_nums_str = "down_nums"
    energy_mode_str = "energy_mode"
    rest_mode_str = "休息"
    double_day_mode_str = "白班+夜班"

    def __init__(self, asset: dict):
        """

        :param asset: 包含 model, model dict 包含 factory 代表数据源，
        energy_mode 即不同的聚类中心, mode_column 表示用那一列作为工作模式聚类，status_column 表示用哪一列做停机时间识别，
        小于 status_ignore_time 表示忽略的停机，小于 status_break_time 表示计入停机时间的停机，否则认为是下班了; 这两项单位统一为分钟
        working_threshold 是一个 [0,1] 的小数，表示工作模式识别阈值离聚类中心的距离，越大识别出来的工作时间越短。
        """
        self.model = asset
        self._check_model()
        self.data = None

    def detect(self, target_date):
        data = self._get_data(target_date)
        self.data = data
        energe_mode = self._get_energy_mode()
        if energe_mode != EnergyStatus.rest_mode_str:
            energy_status = self._get_energy_status(energe_mode)
        else:
            energy_status = {EnergyStatus.uptime_str: None,
                             EnergyStatus.downtime_str: None,
                             EnergyStatus.down_hour_str: "24",
                             EnergyStatus.down_nums_str: 1}
            data["tag"] = 0
        energy_status[EnergyStatus.energy_mode_str] = energe_mode
        return energy_status

    def _get_energy_status(self, mode):
        model = self.model
        if mode == EnergyStatus.double_day_mode_str:
            data = self.data
        else:
            data = self.data.iloc[:len(self.data) // 2]
        energy_cluster = EnergyStatusCluster(data,
                                             model["status_column"],
                                             model["working_threshold"],
                                             model["status_ignore_time"],
                                             model["status_break_time"])
        energy_cluster.analyze()
        energy_time = energy_cluster.get_energy_time()
        break_info = energy_cluster.get_break_info()
        if energy_time[0] is None:
            energy_time[0] = self.data.iloc[0]["datetime"]
        if energy_time[1] is None:
            energy_time[1] = self.data.iloc[-1]["datetime"]

        energy_status = {EnergyStatus.uptime_str: datetime.datetime.strftime(energy_time[0], "%Y-%m-%d %H:%M:%S"),
                         EnergyStatus.downtime_str: datetime.datetime.strftime(energy_time[1], "%Y-%m-%d %H:%M:%S"),
                         EnergyStatus.down_hour_str: break_info[1],
                         EnergyStatus.down_nums_str: break_info[0]}
        return energy_status

    def _get_data(self, target_date):
        if self.model["factory"] == "jinmailang":
            try:
                from data_handler.api_data import jinmailang
                data = jinmailang.get_jinmailang_df(
                    target_date)
                two_days_len = 12 * 24 * 2
                if len(data) != two_days_len:
                    raise DataError("data length %s error, should be %s" %
                                    (len(data), two_days_len))
            except Exception as e:
                raise DataError(str(e))
        else:
            raise DataError("can't find factory %s" % self.model["factory"])
        return data

    def _get_energy_mode(self):
        mode_data = np.array(self.data[self.model["mode_column"]])
        mode_data = mode_data[:len(mode_data) // 2]
        centroids_dict = self.model["energy_mode"]
        min_distance = np.inf
        match_mode = None
        for mode in centroids_dict:
            distance = centroids_dict[mode] - mode_data
            distance = np.sum(distance ** 2)
            if distance < min_distance:
                min_distance = distance
                match_mode = mode
        return match_mode

    def _check_model(self):
        if not "factory" in self.model:
            raise DataError("factory must be in model")
        if not "energy_mode" in self.model:
            raise DataError("energy_mode must be in model")
        if not "mode_column" in self.model:
            raise DataError("mode_column must be in model")
        if not "status_column" in self.model:
            raise DataError("status_column must be in model")
        if not "status_ignore_time" in self.model:
            raise DataError("status_ignore_time must be in model")
        if not "status_break_time" in self.model:
            raise DataError("status_break_time must be in model")
        if not "working_threshold" in self.model:
            raise DataError("working_threshold must be in model")

        pass

    def get_data(self):
        return self.data


class EnergyVolatility(object):
    def __init__(self, asset, data):
        self.model = asset
        self.data = data
        return

    def calculate(self, date):
        data = self.data
        model = self.model
        if "tag" not in self.data.columns:
            energy_cluster = EnergyStatusCluster(data,
                                                 model["status_column"],
                                                 model["working_threshold"],
                                                 0,
                                                 0)
            energy_cluster.analyze()

        volatility_weight_dict = self.model["volatility_weight"]
        df_work = data[(data["tag"] == 1) &
                       (((data["day"] == date.day) & (data["hour"] > 5))
                        | ((data["hour"] < 7) & (data["day"] != date.day)))]
        total_vol = 0
        for key in volatility_weight_dict:
            if len(df_work) == 0:
                vol = 0
            else:
                vol = np.std(df_work[key])
            total_vol += vol * volatility_weight_dict[key]

        return total_vol


def call(param: dict, model_url: str):
    if param is None:
        raise ParamError('Missing required parameter in the JSON body: param')
    date = param.get('date')

    if not date:
        raise ParamError('Required parameter \'date\' not found in \'param\'')

    asset = ModelLoader.load(model_url)
    time_detector = EnergyStatus(asset=asset)
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    time_status = time_detector.detect(target_date=date)
    volatility_detector = EnergyVolatility(asset, time_detector.get_data())
    volatility_status = volatility_detector.calculate(date)
    result = {"working status": time_status, "volatility": volatility_status}

    return result
