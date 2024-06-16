from kafka import KafkaProducer
import json
from binance_api.ohlc_data import get_ohlc_data
import time

def start_producer(topic_name='ohlc_data'):
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    while True:
        ohlc_data = get_ohlc_data()
        producer.send(topic_name, ohlc_data)
        print(f"Sent: {ohlc_data}")
        time.sleep(60)  # Envoyer des données chaque minute

