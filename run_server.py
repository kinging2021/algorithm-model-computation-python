from web import create_app
from conf import config_name
from common.log import logger

logger.info('Using config: ' + config_name)
app = create_app()
logger.info('Flask app created')

if __name__ == '__main__':
    app.run()
