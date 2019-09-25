from hdfs.client import Client
from conf import HDFS_URL, HDFS_ROOT


client = Client(url=HDFS_URL, root=HDFS_ROOT)
