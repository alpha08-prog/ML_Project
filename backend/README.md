# Backend API

FastAPI backend for the EEG Classification project.

## Setup

1. Install dependencies:
```bash
python -m pip install -r ../requirements.txt
```

2. Train models (first time only):
```bash
python train_models.py
```

This will:
- Load and preprocess EEG data
- Train Random Forest (baseline and augmented)
- Train CNN (baseline and augmented)
- Save all models to `models/` directory

3. Start the API server:
```bash
python api.py
```

Or using uvicorn directly:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health & Info
- `GET /` - API info
- `GET /api/health` - Health check
- `GET /api/dataset/info` - Dataset information

### Dashboard
- `GET /api/dashboard` - Dashboard overview metrics

### Model Performance
- `GET /api/performance` - All model performance metrics

### Visualizations
- `GET /api/visualization/umap` - UMAP embedding data
- `GET /api/visualization/channel-importance` - Channel importance analysis
- `GET /api/visualization/eeg-signal/{channel}` - Sample EEG signal for a channel

### Predictions
- `POST /api/predict` - Make prediction on uploaded EDF file
  - Parameters:
    - `file`: EDF file (multipart/form-data)
    - `model_type`: "rf" or "cnn" (query parameter)

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Model Files

After training, the following files will be created in `models/`:

- `rf_baseline.pkl` - Random Forest baseline model
- `rf_augmented.pkl` - Random Forest augmented model
- `cnn_baseline.pt` - CNN baseline model weights
- `cnn_augmented.pt` - CNN augmented model weights
- `cnn_baseline_config.pkl` - CNN baseline configuration
- `cnn_augmented_config.pkl` - CNN augmented configuration
- `*_metrics.pkl` - Performance metrics for each model
- `X_test.npy`, `y_test.npy` - Test dataset
- `X_train.npy`, `y_train.npy` - Training dataset
- `dataset_info.pkl` - Dataset metadata

## Notes

- Models are loaded on API startup
- First request may be slower as models are loaded into memory
- For production, consider using a model serving framework or caching

