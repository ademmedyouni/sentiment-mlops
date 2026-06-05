import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    data_path = "data/processed/train_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run preprocess.py first.")

    df = pd.read_csv(data_path)
    params = load_params()

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=params["features"]["max_features"],
            ngram_range=tuple(params["features"]["ngram_range"]),
        )),
        ("clf", LogisticRegression(
            max_iter=params["model"]["max_iter"],
            C=params["model"]["C"],
        )),
    ])

    mlflow.set_experiment("sentiment-classifier")

    with mlflow.start_run():
        mlflow.log_params({
            "model_type": params["model"]["type"],
            "max_iter":   params["model"]["max_iter"],
            "C":          params["model"]["C"],
            "max_features": params["features"]["max_features"],
            "ngram_range":  str(params["features"]["ngram_range"]),
        })

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        metrics = {
            "accuracy":  accuracy_score(y_val, preds),
            "precision": precision_score(y_val, preds),
            "recall":    recall_score(y_val, preds),
            "f1":        f1_score(y_val, preds),
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/model.pkl")

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
