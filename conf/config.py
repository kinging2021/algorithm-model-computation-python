# -*- coding: utf-8 -*-

# hdfs config
HDFS_URL = 'http://10.39.48.25:50070'
HDFS_ROOT = ''
# HDFS_ROOT = '/algorithm/model/'

# kafka config
KAFKA_BOOTSTRAP_SERVERS = [
    'broker1:1234'
]


# flask config
class FlaskConfig(object):
    pass
    '''
    此处为通用配置,子类分别定义专用配置
    '''


class FlaskDevelopmentConfig(FlaskConfig):
    DEBUG = True
    ENV = 'development'


class FlaskProductionConfig(FlaskConfig):
    DEBUG = False
    ENV = 'production'


FLASK_CONFIG = {
    'development': FlaskDevelopmentConfig,
    'production': FlaskProductionConfig
}
