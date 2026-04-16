#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/project"
DATA_DIR="$PROJECT_ROOT/Data/eegmat"
MODELS_DIR="$PROJECT_ROOT/models"
MODELS_READY_FILE="$MODELS_DIR/dataset_info.pkl"
AUTO_BOOTSTRAP="${AUTO_BOOTSTRAP:-false}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

has_eeg_data() {
    compgen -G "$DATA_DIR/Subject*_*.edf" > /dev/null
}

bootstrap_assets() {
    echo "=== Bootstrapping EEG data and trained models ==="

    cd "$PROJECT_ROOT"
    if has_eeg_data; then
        echo "EEG data already present. Skipping download."
    else
        echo "Downloading EEG data..."
        python download_data.py
    fi

    echo "Training models..."
    cd "$PROJECT_ROOT/backend"
    python train_models.py

    echo "=== Bootstrap complete ==="
}

if [ -f "$MODELS_READY_FILE" ]; then
    echo "=== Trained models found. Skipping bootstrap. ==="
elif [ "$AUTO_BOOTSTRAP" = "true" ]; then
    bootstrap_assets
else
    echo "=== No trained models found. Starting API without bootstrap. ==="
    echo "Set AUTO_BOOTSTRAP=true to download data and train models on startup."
fi

echo "Starting FastAPI server on ${HOST}:${PORT}..."
exec uvicorn api:app --host "$HOST" --port "$PORT"
