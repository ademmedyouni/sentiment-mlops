# Sentiment MLOps Pipeline

A production-grade MLOps pipeline for sentiment classification using the IMDb dataset.

## Stack
| Stage | Tool |
|-------|------|
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Model registry | MLflow Model Registry |
| CI/CD | GitHub Actions |
| Serving | FastAPI + Docker |
| Monitoring | Evidently + Prometheus + Grafana |
| Deployment | Render (free tier) |

## Quick start (Windows — Git Bash)

### 1. Install packages
```bash
pip install -r requirements.txt
```

### 2. Download data
```bash
python download_data.py
```

### 3. Set up DVC remote and track data
```bash
mkdir -p ~/dvc-remote
dvc remote add -d local_remote ~/dvc-remote
dvc add data/raw/train.csv data/raw/test.csv
dvc push
git add data/raw/train.csv.dvc data/raw/test.csv.dvc .dvc/config
git commit -m "data: add IMDb dataset"
```

### 4. Run the full pipeline
```bash
dvc repro
```
This runs: preprocess → train → evaluate

### 5. View experiments in MLflow
```bash
mlflow ui
# Open http://localhost:5000
```

### 6. Register best model
```bash
python register_model.py
```

### 7. Run the API locally
```bash
uvicorn api.app:app --reload
# Open http://localhost:8000/docs
```

### 8. Run with Docker
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

### 9. Run full monitoring stack
```bash
docker compose up -d
# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
```

### 10. Generate drift report
```bash
python monitoring/drift_report.py
# Open monitoring/drift_report.html in browser
```

## Project structure
```
sentiment-mlops/
├── data/
│   ├── raw/          # DVC-tracked CSVs
│   └── processed/    # cleaned data
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   └── app.py        # FastAPI app
├── models/
│   └── model.pkl     # trained model
├── monitoring/
│   ├── drift_report.py
│   └── prometheus.yml
├── tests/
│   ├── test_pipeline.py
│   └── test_smoke.py
├── .github/workflows/
│   ├── ci.yml        # runs on every push
│   └── retrain.yml   # runs on data/params change
├── dvc.yaml          # pipeline stages
├── params.yaml       # hyperparameters
├── Dockerfile
├── docker-compose.yml
└── register_model.py
```

## CI/CD

- **ci.yml** — triggers on every `git push`. Runs syntax checks + tests.
- **retrain.yml** — triggers when `data/**/*.dvc` or `params.yaml` changes. Retrains model automatically.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /predict | Returns sentiment + confidence |
| GET | /health | Health check |
| GET | /metrics | Prometheus metrics |
| GET | /docs | Interactive API docs |
