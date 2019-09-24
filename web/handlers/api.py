from flask_restful import Api
from .executor import Executor
from .index import IndexView

api = Api()

api.add_resource(IndexView, '/')
api.add_resource(Executor, '/api/algorithm/executor')
