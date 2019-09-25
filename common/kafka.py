from kafka import KafkaProducer
from conf import KAFKA_BOOTSTRAP_SERVERS


producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
