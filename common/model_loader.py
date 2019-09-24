import pickle
from .hdfs import client


class ModelLoader(object):
    hdfs_client = client

    def __init__(self):
        pass

    def load(self, model_url):
        """Load model from hdfs

        :param model_url:
        :return:
        """
        with self.hdfs_client.read(model_url) as f:
            obj = pickle.load(f)
            return obj

    @staticmethod
    def load_file(file):
        """Load model from file

        :param model_url:
        :return:
        """
        with open(file, 'rb') as f:
            obj = pickle.load(f)
            return obj
