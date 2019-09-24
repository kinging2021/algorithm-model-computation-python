from kafka import KafkaProducer
from conf.kafka_conf import bootstrap_servers


producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
