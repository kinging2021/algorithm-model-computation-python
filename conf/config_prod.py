# -*- coding: utf-8 -*-

# hdfs config
HDFS_URL = 'http://snhost22.master.cluster.enn.cn:50070'
HDFS_ROOT = ''
# HDFS_ROOT = '/algorithm/model/'

# ELK kafka config
ELK_KAFKA_ENABLE = True
ELK_KAFKA_BOOTSTRAP_SERVERS = [
    '10.39.40.25:9092',
    '10.39.40.29:9092',
    '10.39.40.30:9092'
]
ELK_KAFKA_TOPIC = 'fnw_logging_kafka_topic_prod'
ELK_APP_ID = 'bigdata_algorithm-model-computation-python'


# flask config
class FlaskProductionConfig(object):
    DEBUG = False
    ENV = 'production'


FLASK_CONFIG = FlaskProductionConfig

# history result api
HISTORY_RESULT_API = 'http://algorithm-service-platform-backend.prod-algorithm-service.fnwrancher.enncloud.cn/internal/algorithm/get_result'
