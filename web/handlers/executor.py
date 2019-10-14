import importlib.util
from flask import jsonify
from web.handlers.base import BaseResource
from common.log import logger
from algorithm.exception import ParamError, ModuleError


class Executor(BaseResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser.add_argument('className', required=True, type=str, location=['json'])
        self.parser.add_argument('modelURL', required=False, type=str, location=['json'])
        self.parser.add_argument('param', required=False, type=dict, location=['json'])
        self.ret = {'result': None, 'code': 0, 'msg': ''}

    def post(self):
        data = self.parser.parse_args()
        module_name = data.get('className')

        try:
            module_spec = importlib.util.find_spec(module_name)
            if module_spec is None:
                raise ModuleError('Module not found: %s' % module_name)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        except Exception as e:
            self.ret['code'] = 500
            self.ret['msg'] = 'Module error: %s, %s' % (module_name, repr(e))
            logger.error(self.ret['msg'], exc_info=True, stack_info=True)
            return jsonify(self.ret)

        try:
            result = module.call(param=data.get('param'),
                                 model_url=data.get('modelURL'))
            self.ret['code'] = 200
            self.ret['msg'] = 'OK'
            self.ret['result'] = result
            return jsonify(self.ret)
        except ParamError as e:
            self.ret['code'] = 400
            self.ret['msg'] = 'Exception raised when calling module: %s, %s' % (module_name, repr(e))
            logger.error(self.ret['msg'], exc_info=True, stack_info=True)
            return jsonify(self.ret)
        except Exception as e:
            self.ret['code'] = 500
            self.ret['msg'] = 'Exception raised when calling module: %s, %s' % (module_name, repr(e))
            logger.error(self.ret['msg'], exc_info=True, stack_info=True)
            return jsonify(self.ret)
