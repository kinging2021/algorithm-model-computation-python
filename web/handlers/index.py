from web.handlers.base import BaseResource
from common.log import logger


class IndexView(BaseResource):

    def get(self):
        logger.info('IndexView calling...')
        return 'Hello!'



