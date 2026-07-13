"""
Stage 4 — Performance Gate

Reads:  reports/metrics.json
Checks: best model F1 score >= F1_THRESHOLD environment variable

If the threshold is not met, the pipeline exits with code 1
and blocks the Docker build + deployment stages.

Also sets GitHub Actions step outputs:
  best_model   — name of the best performing model
  best_f1      — F1 score of the best model
"""

import json
import os
import sys
from pathlib import Path

METRICS_PATH = Path("reports/metrics.json")


def run_gate():
    threshold = float(os.environ.get("F1_THRESHOLD", "0.70"))

    if not METRICS_PATH.exists():
        print(f"[FAIL] Metrics file not found: {METRICS_PATH}")
        sys.exit(1)

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best_model = metrics["best_model"]
    best_f1 = float(metrics["best_f1"])

    print(f"\nPerformance Gate Check")
    print(f"  Best model : {best_model}")
    print(f"  Best F1    : {best_f1}")
    print(f"  Threshold  : {threshold}")
    print()

    # Print all model scores for visibility
    for name, m in metrics.get("models", {}).items():
        marker = "<<< BEST" if name == best_model else ""
        print(f"  {name:20s}  F1={m['f1_score']}  Acc={m['accuracy']}  {marker}")

    # Write GitHub Actions step outputs
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"best_model={best_model}\n")
            f.write(f"best_f1={best_f1}\n")

    if best_f1 < threshold:
        print(f"\n[FAIL] Best model F1 ({best_f1}) is below threshold ({threshold}).")
        print("       Deployment blocked. Improve the model or lower the threshold.")
        sys.exit(1)

    print(f"\n[PASS] Performance gate passed. F1 {best_f1} >= {threshold}")


if __name__ == "__main__":
    run_gate()
