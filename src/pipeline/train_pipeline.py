"""
Stage 3 - Model Training Pipeline  (with MLflow tracking)

Reads from:  data/processed/cleaned.csv
Outputs:
  models/<name>_pipeline.joblib
  models/best_model.joblib
  reports/metrics.json
  reports/model_metrics.md
  mlruns/                        <-- MLflow local tracking store

MLflow integration:
  - Every model gets its own child run nested under a parent run
    for the current pipeline execution.
  - Hyperparameters, all metrics, and the best model artifact are
    logged so every historical run is fully reproducible.
  - Experiment name : loan-approval-training
  - Run name        : pipeline-<git sha or timestamp>
  - Best model is registered in the MLflow Model Registry as
    'LoanApprovalBestModel' with a 'Staging' alias.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import TARGET_COLUMN, MODEL_DIR, REPORT_PATH
from src.data_preprocessing import (
    build_preprocessor,
    get_features_and_target,
    clean_target,
)
from src.evaluate import evaluate_model
from src.utils import save_artifact

CLEANED_DATA_PATH   = Path("data/processed/cleaned.csv")
METRICS_JSON_PATH   = Path("reports/metrics.json")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME     = "loan-approval-training"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Annual_Income" in df.columns and "Work_Experience" in df.columns:
        df["income_per_exp_year"] = df["Annual_Income"] / (df["Work_Experience"] + 1)
    if "Applicant_Age" in df.columns and "Work_Experience" in df.columns:
        df["age_minus_experience"] = df["Applicant_Age"] - df["Work_Experience"]
    if "Years_in_Current_Employment" in df.columns and "Work_Experience" in df.columns:
        df["employment_stability"] = df["Years_in_Current_Employment"] / (df["Work_Experience"] + 1)
    if "Years_in_Current_Residence" in df.columns and "Applicant_Age" in df.columns:
        df["residence_stability"] = df["Years_in_Current_Residence"] / (df["Applicant_Age"] + 1)
    return df


# ---------------------------------------------------------------------------
# ImbPipeline builder
# ---------------------------------------------------------------------------
def make_imb_pipeline(classifier, smote_ratio=1.0):
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(sampling_strategy=smote_ratio, k_neighbors=5, random_state=42)),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------
def _log_params(model_name: str, classifier) -> None:
    """Log classifier hyperparameters to the active MLflow run."""
    params = classifier.get_params(deep=False)
    for k, v in params.items():
        try:
            mlflow.log_param(f"{model_name}.{k}", v)
        except Exception:
            pass  # skip un-serialisable params silently


def _log_metrics(model_name: str, metrics: dict) -> None:
    """Log all numeric evaluation metrics to the active MLflow run."""
    scalar_keys = [
        "accuracy", "precision", "recall", "f1_score",
        "f1_class_0", "f1_class_1", "cv_f1_mean", "cv_f1_std",
    ]
    for key in scalar_keys:
        if key in metrics:
            mlflow.log_metric(f"{model_name}.{key}", float(metrics[key]))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def run_training():
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {CLEANED_DATA_PATH}.\n"
            "Run Stage 2 (clean_data) first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    df = clean_target(df)
    df = engineer_features(df)
    X, y = get_features_and_target(df)

    # Class distribution
    class_counts  = y.value_counts().to_dict()
    total         = len(y)
    majority      = max(class_counts.values())
    minority      = min(class_counts.values())
    scale_pos     = round(majority / minority, 2)

    print(f"Dataset : {total} rows | Features: {X.shape[1]}")
    print(f"Columns : {list(X.columns)}")
    print("Target distribution:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  Class {cls}: {cnt} rows ({cnt / total * 100:.1f}%)")
    print(f"  Imbalance ratio : {scale_pos}:1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

    # Model definitions
    candidate_models = {
        "decision_tree": DecisionTreeClassifier(
            random_state=42, max_depth=12, min_samples_leaf=2,
            min_samples_split=4, class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            random_state=42, n_estimators=400, max_depth=15,
            min_samples_leaf=2, min_samples_split=4,
            max_features="sqrt", class_weight="balanced", n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42, n_estimators=300, learning_rate=0.05,
            max_depth=5, min_samples_leaf=3, subsample=0.8,
            max_features="sqrt",
        ),
        "xgboost": XGBClassifier(
            random_state=42, n_estimators=400, learning_rate=0.05,
            max_depth=6, min_child_weight=3, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos,
            eval_metric="logloss", n_jobs=-1, verbosity=0,
        ),
        "svm": SVC(
            kernel="rbf", C=10.0, gamma="scale", probability=True,
            random_state=42, class_weight="balanced",
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            solver="adam", alpha=0.0005, batch_size=128,
            learning_rate="adaptive", learning_rate_init=0.001,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42, verbose=False,
        ),
    }

    smote_ratio_map = {"gradient_boosting": 0.5, "xgboost": 0.5}
    candidate_pipelines = {
        name: make_imb_pipeline(model, smote_ratio=smote_ratio_map.get(name, 1.0))
        for name, model in candidate_models.items()
    }

    # -----------------------------------------------------------------------
    # MLflow setup
    # -----------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    git_sha = os.getenv("GITHUB_SHA", "local")
    build_id = (
        os.getenv("BUILD_ID")
        or os.getenv("GITHUB_RUN_ID")
        or os.getenv("GITHUB_RUN_NUMBER")
        or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    )
    run_name = f"build-{build_id}-{git_sha[:8] if git_sha != 'local' else 'local'}"

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results       = {}
    best_name     = None
    best_score    = -1
    best_pipeline = None
    best_run_id   = None

    # Parent run wraps all model child runs for this pipeline execution
    with mlflow.start_run(run_name=f"pipeline-{run_name}") as parent_run:

        mlflow.log_param("dataset_rows",    total)
        mlflow.log_param("dataset_features", X.shape[1])
        mlflow.log_param("train_rows",       len(X_train))
        mlflow.log_param("test_rows",        len(X_test))
        mlflow.log_param("imbalance_ratio",  scale_pos)
        mlflow.log_param("git_sha",          git_sha)
        mlflow.log_param("build_id",         build_id)
        mlflow.set_tag("build_id",           build_id)
        mlflow.set_tag("git_sha",            git_sha)
        mlflow.set_tag("pipeline_stage",     "model-training")

        for name, pipeline in candidate_pipelines.items():
            print(f"\n{'=' * 55}")
            print(f"Training : {name}")
            print(f"{'=' * 55}")

            # Each model gets its own nested child run
            with mlflow.start_run(run_name=name, nested=True) as child_run:

                mlflow.set_tag("model_name",  name)
                mlflow.set_tag("pipeline_run", parent_run.info.run_id)
                mlflow.set_tag("build_id", build_id)
                mlflow.set_tag("git_sha", git_sha)
                mlflow.log_param("smote_ratio", smote_ratio_map.get(name, 1.0))

                # Log classifier hyperparameters
                _log_params(name, candidate_models[name])

                # Train
                pipeline.fit(X_train, y_train)
                metrics = evaluate_model(pipeline, X_test, y_test)

                cv_f1 = cross_val_score(
                    pipeline, X, y, cv=cv,
                    scoring="f1_weighted", n_jobs=-1,
                )
                metrics["cv_f1_mean"] = round(float(cv_f1.mean()), 4)
                metrics["cv_f1_std"]  = round(float(cv_f1.std()),  4)
                results[name] = metrics

                # Log all metrics to MLflow
                _log_metrics(name, metrics)

                # Log the pipeline as an MLflow artifact
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path=f"model_{name}",
                    registered_model_name=None,   # register only the best below
                    input_example=X_test.head(3),
                )

                print(f"  Accuracy      : {metrics['accuracy']}")
                print(f"  Precision (w) : {metrics['precision']}")
                print(f"  Recall    (w) : {metrics['recall']}")
                print(f"  F1 (weighted) : {metrics['f1_score']}  <-- gate threshold")
                print(f"  F1 Class 0    : {metrics['f1_class_0']}")
                print(f"  F1 Class 1    : {metrics['f1_class_1']}")
                print(f"  CV F1         : {metrics['cv_f1_mean']} +/- {metrics['cv_f1_std']}")
                print(f"  MLflow run_id : {child_run.info.run_id}")

                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                save_artifact(pipeline, MODEL_DIR / f"{name}_pipeline.joblib")

                if metrics["f1_score"] > best_score:
                    best_score    = metrics["f1_score"]
                    best_name     = name
                    best_pipeline = pipeline
                    best_run_id   = child_run.info.run_id

        # Log best model summary to parent run
        mlflow.log_param("best_model",   best_name)
        mlflow.log_metric("best_f1",     best_score)
        mlflow.set_tag("best_run_id",    best_run_id)

    # Save best model artifact
    save_artifact(best_pipeline, MODEL_DIR / "best_model.joblib")

    # Register best model in MLflow Model Registry
    model_uri = f"runs:/{best_run_id}/model_{best_name}"
    try:
        reg = mlflow.register_model(
            model_uri=model_uri,
            name="LoanApprovalBestModel",
        )
        print(f"\nMLflow registry : LoanApprovalBestModel v{reg.version}")
    except Exception as e:
        print(f"MLflow registry skipped (no registry backend): {e}")

    print(f"\n{'=' * 55}")
    print(f"Best model    : {best_name}")
    print(f"Best F1       : {best_score} (weighted)")
    print(f"MLflow UI     : mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"{'=' * 55}")

    # Save metrics JSON
    metrics_out = {
        "best_model":  best_name,
        "best_f1":     best_score,
        "mlflow_run":  parent_run.info.run_id,
        "build_id":    build_id,
        "git_sha":     git_sha,
        "models": {
            name: {k: v for k, v in m.items() if k != "classification_report"}
            for name, m in results.items()
        },
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    METRICS_JSON_PATH.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"Metrics JSON  : {METRICS_JSON_PATH}")

    # Save markdown report
    lines = [
        "# Model Performance Report\n\n",
        f"**Best model:** `{best_name}` — weighted F1 = {best_score}\n",
        f"**Build ID:** `{build_id}`\n",
        f"**Git SHA:** `{git_sha}`\n",
        f"**MLflow parent run:** `{parent_run.info.run_id}`\n\n",
        "---\n\n",
    ]
    for name, m in results.items():
        lines.append(f"## {name.replace('_', ' ').title()}\n\n")
        lines.append("| Metric | Value | CV Mean +/- Std |\n")
        lines.append("|---|---|---|\n")
        lines.append(f"| Accuracy        | {m['accuracy']}   | - |\n")
        lines.append(f"| Precision (w)   | {m['precision']}  | - |\n")
        lines.append(f"| Recall (w)      | {m['recall']}     | - |\n")
        lines.append(f"| F1 (weighted)   | {m['f1_score']}   | {m['cv_f1_mean']} +/- {m['cv_f1_std']} |\n")
        lines.append(f"| F1 Class 0      | {m['f1_class_0']} | - |\n")
        lines.append(f"| F1 Class 1      | {m['f1_class_1']} | - |\n")
        lines.append("\n**Classification Report:**\n```\n")
        lines.append(m["classification_report"])
        lines.append("```\n\n---\n\n")

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text("".join(lines), encoding="utf-8")
    print(f"Markdown report: {REPORT_PATH}")


if __name__ == "__main__":
    run_training()
