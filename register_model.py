"""
Run this after training to register the best run in the MLflow Model Registry.
Usage: python register_model.py
"""
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "sentiment-classifier"
client = MlflowClient()

# Find best run by F1 score
runs = mlflow.search_runs(
    experiment_names=["sentiment-classifier"],
    order_by=["metrics.f1 DESC"],
)

if runs.empty:
    raise RuntimeError(
        "No runs found. Run 'python src/train.py' or 'dvc repro' first."
    )

best_run_id = runs.iloc[0]["run_id"]
best_f1     = runs.iloc[0]["metrics.f1"]
print(f"Best run ID : {best_run_id}")
print(f"Best F1     : {best_f1:.4f}")

# Register model
model_uri = f"runs:/{best_run_id}/model"
mv = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registered  : version {mv.version}")

# Promote to Staging
client.transition_model_version_stage(
    name=MODEL_NAME, version=mv.version, stage="Staging"
)
print("Stage       : Staging")

# Promote to Production
client.transition_model_version_stage(
    name=MODEL_NAME, version=mv.version, stage="Production"
)
print("Stage       : Production")
print(f"\nModel '{MODEL_NAME}' v{mv.version} is now in Production!")
print("Open MLflow UI to verify: mlflow ui -> Models tab")
