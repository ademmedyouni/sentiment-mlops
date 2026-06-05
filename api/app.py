import os
import time

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

MODEL_PATH = "models/model.pkl"

app = FastAPI(title="Sentiment API", version="1.0")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Run 'dvc repro' first to train the model."
    )

model = joblib.load(MODEL_PATH)

REQUEST_COUNT   = Counter("predictions_total", "Total predictions", ["sentiment"])
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Latency in seconds")


class Review(BaseModel):
    text: str


class Prediction(BaseModel):
    sentiment: str
    confidence: float
    label: int


@app.post("/predict", response_model=Prediction)
async def predict(review: Review):
    start = time.time()
    proba = model.predict_proba([review.text])[0]
    label = int(proba.argmax())
    confidence = float(proba.max())
    sentiment = "positive" if label == 1 else "negative"
    REQUEST_COUNT.labels(sentiment=sentiment).inc()
    REQUEST_LATENCY.observe(time.time() - start)
    return Prediction(sentiment=sentiment, confidence=confidence, label=label)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
