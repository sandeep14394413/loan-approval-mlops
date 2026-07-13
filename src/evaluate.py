"""
Model evaluation utilities.

Uses weighted averaging for F1 / precision / recall so that
both the majority class (0 = no default) and the minority class
(1 = default risk) contribute proportionally to the score.

This prevents F1=0.0 caused by models that predict only the
majority class to achieve high accuracy on imbalanced datasets.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained pipeline on the test set.
    Returns a dict of metrics.

    averaging="weighted" accounts for class imbalance:
      - weights each class's score by its support (number of true instances)
      - prevents F1=0.0 on imbalanced datasets where majority class dominates
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_class_0": round(f1_score(y_test, y_pred, pos_label=0, average="binary", zero_division=0), 4),
        "f1_class_1": round(f1_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0), 4),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics
