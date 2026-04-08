#!/bin/bash
set -e

INIT_MARKER="/project/models/.initialized"

if [ ! -f "$INIT_MARKER" ]; then
    echo "=== First run: setting up data and models ==="

    # Download EEG data
    echo "Downloading data..."
    cd /project
    python download_data.py

    # Train models
    echo "Training models..."
    cd /project/backend
    python train_models.py

    # Mark as initialized
    touch "$INIT_MARKER"

    echo "=== Setup complete ==="
else
    echo "=== Skipping setup (already initialized) ==="
fi

# Start the backend server
echo "Starting FastAPI server..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
