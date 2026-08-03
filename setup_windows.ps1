$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/generate_synthetic_data.py --config configs/quickstart.yaml
Write-Host "Setup complete. Train with: python scripts/train.py --config configs/quickstart.yaml"
