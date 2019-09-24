import os
from web import create_app
from common.log import logger

config_name = os.environ.get('CONFIG_NAME') or 'development'
logger.info('Using config: ' + config_name)
app = create_app(config_name)

if __name__ == '__main__':
    app.run()
