"""
Stage 4 - Performance Gate

Reads reports/metrics.json and checks whether the best model's
weighted F1 score meets the minimum threshold set by F1_THRESHOLD.

Blocks deployment if the score is below threshold.
Sets GitHub Actions outputs: best_model, best_f1.

Models evaluated: Decision Tree, Random Forest, SVM, Neural Network
"""

import json
import os
import sys
from pathlib import Path

METRICS_PATH = Path("reports/metrics.json")
F1_THRESHOLD  = float(os.getenv("F1_THRESHOLD", "0.70"))


def main():
    if not METRICS_PATH.exists():
        print(f"ERROR: {METRICS_PATH} not found. Run model training first.")
        sys.exit(1)

    data       = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best_model = data.get("best_model", "unknown")
    best_f1    = float(data.get("best_f1", 0.0))
    models     = data.get("models", {})

    print(f"\nPerformance Gate Check")
    print(f"  Best model : {best_model}")
    print(f"  Best F1    : {best_f1}")
    print(f"  Threshold  : {F1_THRESHOLD}")
    print()

    model_order = ["decision_tree", "random_forest", "svm", "neural_network"]
    for name in model_order:
        if name not in models:
            continue
        m      = models[name]
        f1     = m.get("f1_score", 0.0)
        acc    = m.get("accuracy", 0.0)
        f1_c1  = m.get("f1_class_1", 0.0)
        marker = "<<< BEST" if name == best_model else ""
        print(f"  {name:<22} F1(w)={f1:.4f}  Acc={acc:.4f}  F1(default)={f1_c1:.4f}  {marker}")

    print()

    # Write GitHub Actions outputs
    github_output = os.getenv("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"best_model={best_model}\n")
            fh.write(f"best_f1={best_f1}\n")

    if best_f1 >= F1_THRESHOLD:
        print(f"[PASS] Best model F1 ({best_f1}) >= threshold ({F1_THRESHOLD}).")
        print(f"       Proceeding to Docker build and deployment.")
        sys.exit(0)
    else:
        print(f"[FAIL] Best model F1 ({best_f1}) is below threshold ({F1_THRESHOLD}).")
        print(f"       Deployment blocked. Improve the model or lower the threshold.")
        sys.exit(1)


if __name__ == "__main__":
    main()
