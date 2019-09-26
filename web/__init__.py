from flask import Flask
from conf import FLASK_CONFIG
from . import handlers, extensions


def create_app():

    app = Flask(__name__)

    app.config.from_object(FLASK_CONFIG)

    handlers.init_app(app)

    extensions.init_app(app)

    return app
