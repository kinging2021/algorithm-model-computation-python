import importlib.util
from flask import request, jsonify
from web.handlers.base import BaseResource
from common.log import logger


class Executor(BaseResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser.add_argument('className', required=True, type=str, location=['json'])
        self.parser.add_argument('modelURL', required=True, type=str, location=['json'])
        self.parser.add_argument('param', required=True, location=['json'])
        self.ret = {'result': None, 'code': 0, 'msg': ''}

    def post(self):
        data = request.get_json(True)
        module_name = data.get('className')
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            self.ret['code'] = -1
            self.ret['msg'] = 'Module not found: %s' % module_name
            logger.error(self.ret['msg'], exc_info=True, stack_info=True)
            return jsonify(self.ret)

        try:
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        except Exception:
            self.ret['code'] = -1
            self.ret['msg'] = 'Module loading failed: %s' % module_name
            logger.error(self.ret['msg'], exc_info=True, stack_info=True)
            return jsonify(self.ret)

        try:
            result = module.call(param=data.get('param'),
                                 model_url=data.get('modelURL'))
            self.ret['code'] = 0
            self.ret['msg'] = 'OK'
            self.ret['result'] = result
            return jsonify(self.ret)
        except Exception:
            self.ret['code'] = -2
            self.ret['msg'] = 'Exception raised when calling module: %s' % module_name
            logger.error(self.ret['msg'], exc_info=True, stack_info=True)
            return jsonify(self.ret)
