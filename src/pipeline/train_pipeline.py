"""
Stage 3 - Model Training Pipeline

Reads from:  data/processed/cleaned.csv
Outputs:
  models/decision_tree_pipeline.joblib
  models/random_forest_pipeline.joblib
  models/svm_pipeline.joblib
  models/neural_network_pipeline.joblib
  models/best_model.joblib
  reports/metrics.json
  reports/model_metrics.md

Models trained:
  - Decision Tree
  - Random Forest
  - Support Vector Machine
  - Neural Network (MLPClassifier)

Class imbalance strategy:
  - Decision Tree / Random Forest / SVM  : class_weight='balanced'
  - Neural Network (MLPClassifier)       : SMOTE oversampling via
    imbalanced-learn Pipeline before the classifier step.
    MLPClassifier does NOT support class_weight or sample_weight
    in scikit-learn 1.5.x, so SMOTE is the correct alternative.

Model selection: highest weighted F1 on held-out test set.
5-fold CV F1 (weighted) is computed for all models.
"""

import json
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


def build_nn_pipeline():
    """
    Neural Network pipeline using imbalanced-learn ImbPipeline.
    SMOTE is applied AFTER preprocessing (on transformed numeric data)
    and BEFORE the MLP classifier.

    ImbPipeline is a drop-in replacement for sklearn Pipeline that
    correctly applies resamplers only during fit(), not predict().
    """
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(
            sampling_strategy="minority",
            k_neighbors=5,
            random_state=42,
        )),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )),
    ])


def run_training():
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {CLEANED_DATA_PATH}.\n"
            "Run Stage 2 (clean_data) first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    df = clean_target(df)
    X, y = get_features_and_target(df)

    # -------------------------------------------------------------------
    # Class distribution diagnostics
    # -------------------------------------------------------------------
    class_counts = y.value_counts().to_dict()
    total = len(y)
    print(f"Dataset : {total} rows | Features: {list(X.columns)}")
    print("Target distribution:")
    for cls, cnt in sorted(class_counts.items()):
        label = "No Default" if cls == 0 else "Default Risk"
        print(f"  Class {cls} ({label}): {cnt} rows ({cnt / total * 100:.1f}%)")

    majority = max(class_counts.values())
    minority = min(class_counts.values())
    imbalance_ratio = round(majority / minority, 2)
    print(f"  Imbalance ratio : {imbalance_ratio}:1")
    if imbalance_ratio > 3:
        print("  NOTE: Imbalanced dataset. DT/RF/SVM use class_weight='balanced'.")
        print("        Neural Network uses SMOTE oversampling.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(X_train)} rows | Test : {len(X_test)} rows")

    # -------------------------------------------------------------------
    # Model definitions - standard sklearn Pipeline for DT / RF / SVM
    # -------------------------------------------------------------------
    sklearn_models = {
        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_leaf=3,
            min_samples_split=6,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            random_state=42,
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            min_samples_split=6,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "svm": SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
            probability=True,
            random_state=42,
            class_weight="balanced",
        ),
    }

    candidate_pipelines = {
        name: Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ])
        for name, model in sklearn_models.items()
    }

    # Neural network uses ImbPipeline with SMOTE
    candidate_pipelines["neural_network"] = build_nn_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_name = None
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
        print(f"  F1 (weighted) : {metrics['f1_score']}")
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
