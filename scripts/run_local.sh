#!/usr/bin/env bash
set -e

echo "==> Activating virtual environment..."
source .venv/bin/activate

echo "==> Training models..."
python -m src.train

echo "==> Starting Flask API..."
python -m src.app
