# -*- coding: utf-8 -*-

# hdfs config
HDFS_URL = 'http://10.39.48.25:50070'
HDFS_ROOT = ''
# HDFS_ROOT = '/algorithm/model/'

# ELK kafka config
ELK_KAFKA_ENABLE = False
ELK_KAFKA_BOOTSTRAP_SERVERS = [
    '10.39.40.25:9092',
    '10.39.40.29:9092',
    '10.39.40.30:9092'
]
ELK_KAFKA_TOPIC = 'fnw_logging_kafka_topic'
ELK_APP_ID = 'bigdata_algorithm-model-computation-python'


# flask config
class FlaskDevelopmentConfig(object):
    DEBUG = True
    ENV = 'development'


FLASK_CONFIG = FlaskDevelopmentConfig

# history result api
HISTORY_RESULT_API = 'http://algorithm-service-platform-backend.bigdata-dev.onrancher.enncloud.cn/internal/algorithm/get_result'
