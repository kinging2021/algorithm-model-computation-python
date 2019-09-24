import datetime
import numpy as np
import keras

from data_handler.api_data import IndicatorData
from common.model_loader import ModelLoader


class OneDayLSTM(object):
    def __init__(self, asset: dict):
        self.model = asset['model']
        self.scaler = asset['scaler']
        self.system_code = asset['system_code']
        self.calc_code = asset['calc_code']

    def forecast(self, target_date):
        data = self._get_data(target_date)
        result = self._get_result(data)
        return result

    def _get_data(self, target_date: datetime):
        start_time = target_date + datetime.timedelta(days=-5)
        end_time = target_date + datetime.timedelta(days=-1)

        ds = IndicatorData(self.system_code, self.calc_code)
        raw_data = ds.get_data(start_time, end_time, 'day')
        results = raw_data['data'][0]['result']

        # todo: data check
        return [item['value'] for item in results]

    def _get_result(self, data):
        input_x = self.scaler.transform(np.asarray(data).reshape((-1, 1)))
        predict_y = self.model.predict(input_x.reshape((1, -1, 1)))
        result = self.scaler.inverse_transform(predict_y.reshape(-1, 1))
        return result.tolist()[0][0]


def call(param: dict, model_url: str):
    # todo: load model from file server
    # asset = ModelLoader.load(model_url)
    keras.backend.clear_session()
    asset = ModelLoader.load_file('./static/kelun_power_forecast_one_day.pkl')
    forecaster = OneDayLSTM(asset=asset)
    date = datetime.datetime.strptime(param.get('date'), '%Y-%m-%d')
    result = forecaster.forecast(target_date=date)
    return result
