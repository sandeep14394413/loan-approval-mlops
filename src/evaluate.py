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
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics
