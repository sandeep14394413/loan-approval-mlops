import joblib
from pathlib import Path


def save_artifact(obj, path):
    """Persist a Python object to disk using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print(f"Saved: {path}")


def load_artifact(path):
    """Load a joblib artifact from disk."""
    return joblib.load(path)
