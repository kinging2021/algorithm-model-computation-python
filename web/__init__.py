from flask import Flask
from conf.web_conf import config
from . import handlers, extensions


def create_app(config_name='DevelopmentConfig'):
    """
    :param config_name: 环境名
    :return: app应用
    """

    app = Flask(__name__)

    app.config.from_object(config[config_name])

    handlers.init_app(app)

    extensions.init_app(app)

    return app
