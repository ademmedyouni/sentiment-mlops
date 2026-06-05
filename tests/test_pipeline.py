import json
import joblib
import pandas as pd
import pytest


def test_data_schema():
    df = pd.read_csv("data/processed/train_clean.csv")
    assert "text" in df.columns, "Missing 'text' column"
    assert "label" in df.columns, "Missing 'label' column"
    assert df["label"].isin([0, 1]).all(), "Labels must be 0 or 1"
    assert len(df) > 1000, "Dataset too small"


def test_model_loads():
    model = joblib.load("models/model.pkl")
    assert model is not None


def test_model_predicts_valid_label():
    model = joblib.load("models/model.pkl")
    result = model.predict(["this movie was absolutely amazing"])
    assert result[0] in [0, 1], "Prediction must be 0 or 1"


def test_model_predicts_positive():
    model = joblib.load("models/model.pkl")
    result = model.predict(["this film was brilliant and wonderful"])
    assert result[0] == 1, "Expected positive sentiment"


def test_model_predicts_negative():
    model = joblib.load("models/model.pkl")
    result = model.predict(["this movie was terrible and boring"])
    assert result[0] == 0, "Expected negative sentiment"


def test_minimum_accuracy():
    with open("metrics.json") as f:
        m = json.load(f)
    assert m["accuracy"] > 0.85, f"Accuracy {m['accuracy']:.3f} below threshold 0.85"


def test_minimum_f1():
    with open("metrics.json") as f:
        m = json.load(f)
    assert m["f1"] > 0.85, f"F1 {m['f1']:.3f} below threshold 0.85"
