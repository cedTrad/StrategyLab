from kafka import KafkaConsumer
import json
import pandas as pd
from ml_model.process import process_batches

def start_consumer():
    consumer = KafkaConsumer(
        'ohlc_data',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

    data_list = []

    for message in consumer:
        ohlc_data = message.value
        data_list.append(ohlc_data)
        
        if len(data_list) >= 30:
            df = pd.DataFrame(data_list)
            predictions = process_batches(df)
            print(f"Predictions: {predictions}")
            data_list = []
