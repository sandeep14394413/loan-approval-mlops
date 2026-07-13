"""
Training entrypoint.

Usage:
    python -m src.train

This script:
  1. Loads and cleans the dataset
  2. Splits into train / test
  3. Builds a preprocessing + model pipeline for each algorithm
  4. Trains and evaluates Decision Tree, Random Forest, and SVM
  5. Saves every pipeline to models/
  6. Saves the best model (by F1 score) as models/best_model.joblib
  7. Writes a performance report to reports/model_metrics.md
"""

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import MODEL_DIR, RAW_DATA_PATH, REPORT_PATH
from src.data_preprocessing import (
    build_preprocessor,
    clean_target,
    get_features_and_target,
    load_data,
)
from src.evaluate import evaluate_model
from src.utils import save_artifact


def train_all_models():
    # ------------------------------------------------------------------ #
    # 1. Load and prepare data
    # ------------------------------------------------------------------ #
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found.\n"
            f"Please download from Kaggle and place at:\n  {RAW_DATA_PATH}"
        )

    print("Loading dataset...")
    df = load_data(RAW_DATA_PATH)
    df = clean_target(df)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ------------------------------------------------------------------ #
    # 2. Define candidate models
    # ------------------------------------------------------------------ #
    candidate_models = {
        "decision_tree": DecisionTreeClassifier(
            random_state=42, max_depth=5, min_samples_leaf=5
        ),
        "random_forest": RandomForestClassifier(
            random_state=42, n_estimators=200, max_depth=8, min_samples_leaf=3
        ),
        "svm": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42
        ),
    }

    # ------------------------------------------------------------------ #
    # 3. Train, evaluate, save
    # ------------------------------------------------------------------ #
    results = {}
    best_name = None
    best_score = -1
    best_pipeline = None

    for name, model in candidate_models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        print(f"  Accuracy : {metrics['accuracy']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall   : {metrics['recall']}")
        print(f"  F1 Score : {metrics['f1_score']}")

        save_artifact(pipeline, MODEL_DIR / f"{name}_pipeline.joblib")

        if metrics["f1_score"] > best_score:
            best_score = metrics["f1_score"]
            best_name = name
            best_pipeline = pipeline

    # ------------------------------------------------------------------ #
    # 4. Save best model
    # ------------------------------------------------------------------ #
    save_artifact(best_pipeline, MODEL_DIR / "best_model.joblib")
    print(f"\nBest model: {best_name} (F1 = {best_score})")

    # ------------------------------------------------------------------ #
    # 5. Write performance report
    # ------------------------------------------------------------------ #
    lines = [
        "# Model Performance Report\n",
        f"**Best model selected:** `{best_name}` (F1 = {best_score})\n",
        "---\n",
    ]
    for name, metrics in results.items():
        lines.append(f"## {name.replace('_', ' ').title()}\n")
        lines.append(f"| Metric | Score |\n|---|---|")
        lines.append(f"\n| Accuracy | {metrics['accuracy']} |")
        lines.append(f"\n| Precision | {metrics['precision']} |")
        lines.append(f"\n| Recall | {metrics['recall']} |")
        lines.append(f"\n| F1 Score | {metrics['f1_score']} |\n\n")
        lines.append("**Classification Report:**\n```\n")
        lines.append(metrics["classification_report"])
        lines.append("```\n---\n")

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text("".join(lines), encoding="utf-8")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    train_all_models()
