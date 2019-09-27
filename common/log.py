import sys
import time
import logging
import socket
import json
from common.kafka import elk_producer
from conf import ELK_KAFKA_TOPIC, ELK_APP_ID, ELK_KAFKA_ENABLE


class LoggerHandlerToKafKa(logging.Handler):
    def __init__(self, topic, app_id, version=1, enable=True):
        super(LoggerHandlerToKafKa, self).__init__()

        self.producer = elk_producer
        self.topic = topic
        self.app_id = app_id
        self.hostname = socket.gethostname()
        self.version = version
        self.enable = enable

    def emit(self, record):
        if not self.enable:
            return
        data = {
            "@timestamp": self.__get_time(record.created),
            "@version": self.version,
            "TOPIC": self.topic,
            "appId": self.app_id,
            "hostname": self.hostname,

            "logger": record.name,

            "module": record.module,
            "lineno": record.lineno,
            "level": record.levelname,
            "message": record.msg,
            "exc_info": str(record.exc_info),
            "stack_info": record.stack_info,
        }

        self.producer.send(self.topic, json.dumps(data).encode('utf-8'))

    @staticmethod
    def __get_time(created_time):
        msecs = (created_time - int(created_time)) * 1000
        time_format = '%Y-%m-%dT%H:%M:%S' + '.%03d' % msecs + '%z'
        s = time.strftime(time_format, time.localtime(created_time))
        s = '%s:%s' % (s[:-2], s[-2:])
        return s


# 日志管理
logger = logging.getLogger('logger')

formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter

kafka_handler = LoggerHandlerToKafKa(ELK_KAFKA_TOPIC, ELK_APP_ID, enable=ELK_KAFKA_ENABLE)

logger.addHandler(kafka_handler)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)
