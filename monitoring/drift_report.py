"""
Run this script to generate a data drift report.
Usage: python monitoring/drift_report.py
Output: monitoring/drift_report.html  (open in browser)
"""
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REFERENCE_PATH = "data/processed/train_clean.csv"
PRODUCTION_LOGS = "monitoring/production_logs.csv"
OUTPUT_HTML = "monitoring/drift_report.html"

if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"{REFERENCE_PATH} not found. Run dvc repro first.")

# Reference = training data sample
reference = pd.read_csv(REFERENCE_PATH).sample(500, random_state=42)

# Current = production traffic logs (fallback to test set for demo)
if os.path.exists(PRODUCTION_LOGS):
    current = pd.read_csv(PRODUCTION_LOGS).tail(500)
    print(f"Using production logs: {PRODUCTION_LOGS}")
else:
    test_path = "data/raw/test.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"No production logs found at {PRODUCTION_LOGS} and no test.csv fallback."
        )
    current = pd.read_csv(test_path).sample(500, random_state=99)
    print(f"No production logs found — using test.csv as simulation.")

# Build and save report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

os.makedirs("monitoring", exist_ok=True)
report.save_html(OUTPUT_HTML)
print(f"Drift report saved to: {OUTPUT_HTML}")
print("Open it in your browser to see the analysis.")
