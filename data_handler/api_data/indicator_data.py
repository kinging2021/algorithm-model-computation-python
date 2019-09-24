import requests
from datetime import datetime


class IndicatorData(object):
    url = 'http://bigdata-api-fnw-release.topaas.enncloud.cn/internal/bigdata/calculation/get_result'

    def __init__(self, system_code, calc_code, device_id=''):
        self.system_code = system_code
        self.calc_code = calc_code
        self.device_id = device_id

    def get_data(self, start_time: datetime, end_time: datetime, time_type):
        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
        }
        param = {
            "calcInstenceParams": [
                {
                    "systemCode": self.system_code,
                    "calcCode": self.calc_code,
                    "deviceId": self.device_id
                }
            ],
            "timeType": time_type,
            "startTime": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "endTime": end_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        r = requests.post(self.url, json=param, headers=headers)
        return r.json()
