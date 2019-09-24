from flask_restful import Resource, reqparse


class BaseResource(Resource):
    def __init__(self, *args, **kwargs):
        super(BaseResource, self).__init__(*args, **kwargs)
        self.parser = reqparse.RequestParser()
