from kafka import KafkaProducer
from conf import ELK_KAFKA_BOOTSTRAP_SERVERS


elk_producer = KafkaProducer(bootstrap_servers=ELK_KAFKA_BOOTSTRAP_SERVERS)
