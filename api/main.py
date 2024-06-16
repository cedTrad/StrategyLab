from fastapi import FastAPI, Request
from kafka import KafkaProducer
import json

app = FastAPI()

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

@app.post("/send_data")
async def send_data(request: Request):
    json_data = await request.json()
    producer.send('ohlc_data', json_data)
    return {"status": "data sent"}

# Lancer l'application FastAPI
# Utilise `uvicorn` pour lancer l'application, par exemple :
# uvicorn api.main:app --reload
