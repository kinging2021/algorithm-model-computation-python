import requests
from conf import HISTORY_RESULT_API


class HistoryResult(object):
    url = HISTORY_RESULT_API

    def __init__(self, call_code):
        self.call_code = call_code

    def get_result(self):
        params = {
            "callCode": self.call_code
        }
        r = requests.get(self.url, params)
        return r.json()
