# Backend API

FastAPI backend for the EEG classification project.

## Local setup

```bash
cd backend
conda env create -f environment.yml
conda activate mlproj
poetry install
python api.py
```

The API is served on `http://localhost:8000`.

## Training flow

The API expects trained artifacts inside `../models/`. Generate them with:

```bash
python ../download_data.py
python train_models.py
```

If model files are missing, the API still starts, but most endpoints return `503` until training is completed.

## Docker behavior

The container entrypoint supports two modes:

- `AUTO_BOOTSTRAP=true`: download EEG data if needed, train models, then start the API
- `AUTO_BOOTSTRAP=false`: skip training and just start the API

Environment variables:

- `AUTO_BOOTSTRAP` default `false`
- `HOST` default `0.0.0.0`
- `PORT` default `8000`

## Main endpoints

- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/performance`
- `GET /api/visualization/umap`
- `GET /api/visualization/channel-importance`
- `GET /api/visualization/eeg-signal/{channel}`
- `GET /api/dataset/info`
- `POST /api/predict`

Interactive docs are available at `/docs` and `/redoc`.
