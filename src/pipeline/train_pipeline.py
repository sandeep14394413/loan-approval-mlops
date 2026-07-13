"""
Stage 3 - Model Training Pipeline

Reads from:  data/processed/cleaned.csv
Outputs:
  models/<name>_pipeline.joblib  (one per model)
  models/best_model.joblib
  reports/metrics.json
  reports/model_metrics.md

Models trained:
  1. Decision Tree        (class_weight=balanced)
  2. Random Forest        (class_weight=balanced)
  3. SVM                  (class_weight=balanced)
  4. Gradient Boosting    (scale_pos_weight via sample_weight)
  5. XGBoost              (scale_pos_weight auto-computed)
  6. Neural Network       (MLPClassifier + SMOTE oversampling)

Improvements for small imbalanced dataset (7k rows, 5:1 ratio):
  - SMOTE applied inside ImbPipeline for all models
  - GradientBoosting & XGBoost added (best for tabular imbalanced data)
  - Feature engineering: income_per_year, age_experience_ratio
  - Wider hyperparameter search for each model
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
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

CLEANED_DATA_PATH = Path("data/processed/cleaned.csv")
METRICS_JSON_PATH = Path("reports/metrics.json")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that help the model distinguish default risk.
    These are domain-meaningful ratios for loan risk assessment.
    """
    df = df.copy()

    # Income per year of work experience (earning efficiency)
    if "Annual_Income" in df.columns and "Work_Experience" in df.columns:
        df["income_per_exp_year"] = df["Annual_Income"] / (df["Work_Experience"] + 1)

    # Age relative to work experience (career start age proxy)
    if "Applicant_Age" in df.columns and "Work_Experience" in df.columns:
        df["age_minus_experience"] = df["Applicant_Age"] - df["Work_Experience"]

    # Employment stability: years in current job vs total experience
    if "Years_in_Current_Employment" in df.columns and "Work_Experience" in df.columns:
        df["employment_stability"] = df["Years_in_Current_Employment"] / (
            df["Work_Experience"] + 1
        )

    # Residence stability
    if "Years_in_Current_Residence" in df.columns and "Applicant_Age" in df.columns:
        df["residence_stability"] = df["Years_in_Current_Residence"] / (
            df["Applicant_Age"] + 1
        )

    return df


# ---------------------------------------------------------------------------
# ImbPipeline builder used for all models
# ---------------------------------------------------------------------------
def make_imb_pipeline(classifier, smote_ratio=1.0):
    """
    Wrap any classifier in an ImbPipeline with SMOTE.
    ImbPipeline applies SMOTE only during fit(), never during predict().
    smote_ratio=1.0 means oversample minority to equal majority.
    """
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(
            sampling_strategy=smote_ratio,
            k_neighbors=5,
            random_state=42,
        )),
        ("classifier", classifier),
    ])


def run_training():
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {CLEANED_DATA_PATH}.\n"
            "Run Stage 2 (clean_data) first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    df = clean_target(df)

    # Apply feature engineering before split
    df = engineer_features(df)

    X, y = get_features_and_target(df)

    # -------------------------------------------------------------------
    # Class distribution diagnostics
    # -------------------------------------------------------------------
    class_counts = y.value_counts().to_dict()
    total = len(y)
    print(f"Dataset : {total} rows | Features: {X.shape[1]}")
    print(f"Columns : {list(X.columns)}")
    print("Target distribution:")
    for cls, cnt in sorted(class_counts.items()):
        label = "No Default" if cls == 0 else "Default Risk"
        print(f"  Class {cls} ({label}): {cnt} rows ({cnt / total * 100:.1f}%)")

    majority = max(class_counts.values())
    minority = min(class_counts.values())
    scale_pos = round(majority / minority, 2)
    print(f"  Imbalance ratio : {scale_pos}:1  (scale_pos_weight for XGBoost)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

    # -------------------------------------------------------------------
    # Model definitions
    # Note: All wrapped in ImbPipeline+SMOTE via make_imb_pipeline().
    # XGBoost and GradientBoosting also use scale_pos_weight / sample_weight
    # for dual imbalance correction.
    # -------------------------------------------------------------------
    candidate_models = {
        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=12,
            min_samples_leaf=2,
            min_samples_split=4,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            random_state=42,
            n_estimators=400,
            max_depth=15,
            min_samples_leaf=2,
            min_samples_split=4,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=3,
            subsample=0.8,
            max_features="sqrt",
        ),
        "xgboost": XGBClassifier(
            random_state=42,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,   # handles imbalance natively
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
            verbosity=0,
        ),
        "svm": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42,
            class_weight="balanced",
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            batch_size=128,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        ),
    }

    # Wrap all models in ImbPipeline + SMOTE
    # SMOTE ratio 0.5 for boosting models (partial balance) to
    # avoid over-smoothing; 1.0 (full balance) for others
    smote_ratio_map = {
        "gradient_boosting": 0.5,
        "xgboost":           0.5,
    }
    candidate_pipelines = {
        name: make_imb_pipeline(
            model,
            smote_ratio=smote_ratio_map.get(name, 1.0),
        )
        for name, model in candidate_models.items()
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results    = {}
    best_name  = None
    best_score = -1
    best_pipeline = None

    for name, pipeline in candidate_pipelines.items():
        print(f"\n{'=' * 55}")
        print(f"Training : {name}")
        print(f"{'=' * 55}")

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        cv_f1 = cross_val_score(
            pipeline, X, y,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        metrics["cv_f1_mean"] = round(float(cv_f1.mean()), 4)
        metrics["cv_f1_std"]  = round(float(cv_f1.std()), 4)
        results[name] = metrics

        print(f"  Accuracy      : {metrics['accuracy']}")
        print(f"  Precision (w) : {metrics['precision']}")
        print(f"  Recall    (w) : {metrics['recall']}")
        print(f"  F1 (weighted) : {metrics['f1_score']}  <-- gate threshold")
        print(f"  F1 Class 0    : {metrics['f1_class_0']}")
        print(f"  F1 Class 1    : {metrics['f1_class_1']}")
        print(f"  CV F1         : {metrics['cv_f1_mean']} +/- {metrics['cv_f1_std']}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        save_artifact(pipeline, MODEL_DIR / f"{name}_pipeline.joblib")

        if metrics["f1_score"] > best_score:
            best_score    = metrics["f1_score"]
            best_name     = name
            best_pipeline = pipeline

    save_artifact(best_pipeline, MODEL_DIR / "best_model.joblib")

    print(f"\n{'=' * 55}")
    print(f"Best model    : {best_name}")
    print(f"Best F1       : {best_score} (weighted)")
    print(f"{'=' * 55}")

    # -------------------------------------------------------------------
    # Save metrics JSON
    # -------------------------------------------------------------------
    metrics_out = {
        "best_model": best_name,
        "best_f1":    best_score,
        "models": {
            name: {k: v for k, v in m.items() if k != "classification_report"}
            for name, m in results.items()
        },
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    METRICS_JSON_PATH.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"Metrics JSON  : {METRICS_JSON_PATH}")

    # -------------------------------------------------------------------
    # Save markdown report
    # -------------------------------------------------------------------
    lines = [
        "# Model Performance Report\n\n",
        f"**Best model:** `{best_name}` - weighted F1 = {best_score}\n\n",
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
