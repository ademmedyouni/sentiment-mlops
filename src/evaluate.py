import json
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)

data_path = "data/processed/train_clean.csv"
model_path = "models/model.pkl"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Run preprocess.py first.")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found. Run train.py first.")

df = pd.read_csv(data_path)
model = joblib.load(model_path)
preds = model.predict(df["text"])

metrics = {
    "accuracy":  accuracy_score(df["label"], preds),
    "precision": precision_score(df["label"], preds),
    "recall":    recall_score(df["label"], preds),
    "f1":        f1_score(df["label"], preds),
}

print(classification_report(df["label"], preds, target_names=["negative", "positive"]))
print(json.dumps(metrics, indent=2))

# Save evaluation report
with open("evaluation.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
