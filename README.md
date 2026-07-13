# Loan Approval ML Project — MLOps + DevOps End-to-End

> Predict whether a loan application will be approved using Machine Learning, served via Flask API, containerized with Docker, and deployed to a local Kubernetes cluster using Kind.

![Python](https://img.shields.io/badge/python-3.11-blue) ![Docker](https://img.shields.io/badge/docker-ready-blue) ![Kubernetes](https://img.shields.io/badge/kubernetes-kind-blue) ![CI](https://github.com/sandeep14394413/loan-approval-mlops/actions/workflows/ci.yml/badge.svg)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Algorithms Used](#algorithms-used)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Train the Models](#train-the-models)
- [Run the API Locally](#run-the-api-locally)
- [Docker](#docker)
- [Kind + Kubernetes Deployment](#kind--kubernetes-deployment)
- [API Reference](#api-reference)
- [Model Performance Report](#model-performance-report)
- [MLOps and DevOps Practices](#mlops-and-devops-practices)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project builds a complete end-to-end machine learning pipeline to predict whether a loan application should be approved.

It demonstrates:
- Structured ML project layout
- Multi-model training and comparison
- Clean preprocessing pipeline using scikit-learn
- Model serving via REST API
- Docker containerization
- Local Kubernetes deployment using Kind
- Automated CI with GitHub Actions

---

## Dataset

**Source:** [Applicant Details For Loan Approve — Kaggle](https://www.kaggle.com/datasets/yaminh/applicant-details-for-loan-approve?resource=download)

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0, 1, 2, 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numeric | Monthly income of applicant |
| CoapplicantIncome | Numeric | Monthly income of co-applicant |
| LoanAmount | Numeric | Loan amount (thousands) |
| Loan_Amount_Term | Numeric | Term in months |
| Credit_History | Numeric | 1 = good history, 0 = bad |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| **Loan_Status** | **Target** | **Y = Approved, N = Rejected** |

> **Note:** Download the CSV from Kaggle and place it at `data/raw/applicant-details-for-loan-approve.csv`.

---

## Algorithms Used

| Algorithm | Type | Key Strengths |
|---|---|---|
| Decision Tree | Tree-based | Interpretable, fast |
| Random Forest | Ensemble | High accuracy, handles missing values |
| Support Vector Machine | Kernel-based | Effective in high-dimensional spaces |

All models are evaluated on the same test split. The best model (by F1-score) is saved as `models/best_model.joblib` and used for inference.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| ML Library | scikit-learn |
| API Server | Flask + Gunicorn |
| Containerization | Docker |
| Local K8s | Kind |
| Orchestration | Kubernetes |
| CI Pipeline | GitHub Actions |
| Model Persistence | joblib |

---

## Project Structure

```bash
loan-approval-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── data/
│   ├── raw/                    # Place Kaggle CSV here
│   └── processed/              # Auto-generated processed data
├── k8s/
│   ├── deployment.yaml         # Kubernetes Deployment manifest
│   ├── service.yaml            # Kubernetes Service manifest
│   └── kind-config.yaml        # Kind cluster config (1 control-plane + 1 worker)
├── models/                     # Saved model artifacts (.joblib)
├── notebooks/
│   └── exploration.ipynb       # EDA and experiments
├── reports/
│   └── model_metrics.md        # Auto-generated after training
├── scripts/
│   ├── download_data.md        # Instructions to get the dataset
│   ├── run_local.sh            # Run training + API locally
│   └── deploy_kind.sh          # Full Kind deployment script
├── src/
│   ├── __init__.py
│   ├── app.py                  # Flask REST API
│   ├── config.py               # Paths and feature config
│   ├── data_preprocessing.py   # Feature pipeline builder
│   ├── train.py                # Model training entrypoint
│   ├── evaluate.py             # Metrics evaluation
│   ├── predict.py              # Load model and predict
│   └── utils.py                # Save/load artifacts
├── tests/
│   └── test_api.py             # Basic API test
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/sandeep14394413/loan-approval-mlops.git
cd loan-approval-mlops
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the CSV from [Kaggle](https://www.kaggle.com/datasets/yaminh/applicant-details-for-loan-approve?resource=download) and place it at:

```bash
data/raw/applicant-details-for-loan-approve.csv
```

---

## Train the Models

```bash
python -m src.train
```

This will:
- Load and clean the dataset
- Build a preprocessing pipeline (imputation + scaling + encoding)
- Train Decision Tree, Random Forest, and SVM
- Evaluate and compare all 3 models
- Save each model to `models/`
- Save the best model as `models/best_model.joblib`
- Write performance metrics to `reports/model_metrics.md`

---

## Run the API Locally

```bash
python -m src.app
```

The API will be available at `http://localhost:5000`.

---

## Docker

### Build the image

```bash
docker build -t loan-approval-mlops:latest .
```

### Run the container

```bash
docker run -p 5000:5000 loan-approval-mlops:latest
```

### Test it

```bash
curl http://localhost:5000/health
```

---

## Kind + Kubernetes Deployment

Make sure you have [Kind](https://kind.sigs.k8s.io/) and `kubectl` installed.

### Step 1 — Create the Kind cluster

```bash
kind create cluster --name loan-cluster --config k8s/kind-config.yaml
```

### Step 2 — Build and load the Docker image

```bash
docker build -t loan-approval-mlops:latest .
kind load docker-image loan-approval-mlops:latest --name loan-cluster
```

### Step 3 — Apply Kubernetes manifests

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Step 4 — Verify pods are running

```bash
kubectl get pods
kubectl get svc
```

### Step 5 — Forward port and test

```bash
kubectl port-forward service/loan-approval-service 5000:5000
```

Now open `http://localhost:5000/health` in your browser or test with curl.

---

## API Reference

### Health Check

```
GET /health
```

Response:
```json
{"status": "ok"}
```

### Predict

```
POST /predict
Content-Type: application/json
```

Request body:
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

Response:
```json
{
  "prediction": 1,
  "loan_approved": true,
  "approval_probability": 0.87
}
```

---

## Model Performance Report

After running training, check `reports/model_metrics.md` for detailed per-model results.

Typical expected comparison:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Decision Tree | ~78% | ~75% | ~80% | ~77% |
| Random Forest | ~82% | ~80% | ~84% | ~82% |
| SVM | ~80% | ~78% | ~82% | ~80% |

> Actual numbers will vary based on the dataset version. The best model is automatically selected.

---

## MLOps and DevOps Practices

| Practice | Implementation |
|---|---|
| Modular codebase | Separate files for preprocessing, training, evaluation, serving |
| Reproducibility | Fixed `random_state`, pinned `requirements.txt` |
| Pipeline abstraction | scikit-learn `Pipeline` combines preprocessing + model |
| Model artifact management | `joblib` saves each model + best model |
| REST API serving | Flask with `/health` and `/predict` endpoints |
| Containerization | Dockerfile with Gunicorn for production serving |
| Local Kubernetes | Kind cluster with Deployment + ClusterIP Service |
| Health probes | Kubernetes readiness and liveness probes on `/health` |
| Resource limits | CPU and memory requests/limits defined in Deployment |
| CI pipeline | GitHub Actions — install deps and run tests on every push |
| Automated testing | pytest test for health endpoint |
| Clean `.gitignore` | Data and model binaries excluded from git |

---

## Future Improvements

- [ ] Add experiment tracking with **MLflow**
- [ ] Add data validation with **Great Expectations** or **Pydantic**
- [ ] Add model monitoring and drift detection with **Evidently**
- [ ] Replace Flask with **FastAPI** for auto-generated API docs
- [ ] Add **Helm chart** for production Kubernetes deployment
- [ ] Add CD pipeline to push to remote cluster on merge to main
- [ ] Add **Prometheus metrics** endpoint and **Grafana** dashboard
- [ ] Add hyperparameter tuning with `GridSearchCV` or **Optuna**
- [ ] Containerize model training job as a separate Kubernetes Job

---

## Author

**Sandeep Upadhayay**  
DevOps Engineer | SRE  
[GitHub](https://github.com/sandeep14394413)
