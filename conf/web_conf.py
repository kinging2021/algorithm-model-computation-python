# -*- coding: utf-8 -*-

# Config里面为通用配置,子类分别定义专用配置

import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    pass
    '''
    此处为配置内容
    '''


class DevelopmentConfig(Config):
    DEBUG = True
    ENV = 'development'


class ProductionConfig(Config):
    DEBUG = False
    ENV = 'production'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}
