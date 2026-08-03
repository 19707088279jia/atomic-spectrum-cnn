#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/generate_synthetic_data.py --config configs/quickstart.yaml
echo "Setup complete. Train with: python scripts/train.py --config configs/quickstart.yaml"
