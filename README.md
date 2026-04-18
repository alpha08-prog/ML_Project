# EEG Mental Arithmetic Classification

End-to-end EEG classification project for analyzing mental arithmetic performance quality. The repository combines a FastAPI backend, a React frontend, model training utilities, Docker-based local and production workflows, and GitHub Actions for CI/CD, with production deployment running on AWS EC2.

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Running with Docker](#running-with-docker)
- [Data and model pipeline](#data-and-model-pipeline)
- [API overview](#api-overview)
- [Frontend overview](#frontend-overview)
- [Quality checks](#quality-checks)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Overview

This project predicts mental arithmetic task performance quality from EEG recordings. It includes:

- a FastAPI backend for model serving, metrics, dataset summaries, visualizations, and predictions
- a React + Vite frontend for dashboards and interactive exploration
- training scripts for Random Forest and CNN models
- Docker Compose workflows for development and production
- GitHub Actions workflows for validation and deployment
- a production deployment on AWS EC2

The backend can bootstrap the dataset and train artifacts automatically on first Docker startup, which makes the project easier to run on a fresh machine or server.

## Key features

- EEG-based binary classification for mental arithmetic performance quality
- Random Forest baseline and augmented models
- CNN baseline and augmented models when PyTorch is available
- Dataset summary and model performance dashboards
- Visualization endpoints for UMAP embeddings, channel importance, and EEG signal samples
- File upload prediction flow for EEG inputs
- Local development with native tooling or Docker Compose
- Production deployment on AWS EC2 using prebuilt Docker images and SSH-based CD

## Architecture

```text
GitHub Actions CD
        |
        | builds and pushes Docker images
        v
GitHub Container Registry (GHCR)
        |
        | deploys over SSH
        v
AWS EC2 instance
        |
        +-- frontend container (Nginx + React build)
        |       |
        |       +-- proxies /api, /docs, /redoc, /openapi.json
        |
        +-- backend container (FastAPI)
                |
                +-- loads trained artifacts from /models
                +-- reads EEG data from /Data
                +-- serves metrics, visualizations, and predictions
```

In development, the frontend runs through Vite and the backend runs directly or through Docker Compose. In production, the application runs on AWS EC2, where the frontend is built and served by Nginx and proxies API traffic to the backend container.

## Tech stack

**Backend**

- Python 3.12
- FastAPI
- Uvicorn
- scikit-learn
- PyTorch
- NumPy, Pandas, SciPy
- pyEDFlib
- UMAP

**Frontend**

- React 19
- TypeScript
- Vite
- Recharts
- Framer Motion

**DevOps**

- Docker and Docker Compose
- GitHub Actions
- GitHub Container Registry (GHCR)
- AWS EC2
- SSH-based deployment from GitHub Actions to EC2

## Repository structure

```text
ML_Project/
|-- backend/                 FastAPI app, training code, Python setup
|-- frontend/ml/            React frontend
|-- Data/                   EEG metadata and downloaded EDF files at runtime
|-- models/                 Trained model artifacts generated locally or on first deploy
|-- .github/workflows/      CI and CD pipelines
|-- docker-compose.yml      Development Docker Compose stack
|-- docker-compose.prod.yml Production Docker Compose stack
|-- download_data.py        Dataset download helper
|-- README.md               Project documentation
```

## Prerequisites

Choose the setup path that fits how you want to run the project.

**For local native development**

- Python `3.12`
- Conda
- Node.js `20.19.0` or newer
- npm

**For Docker-based development or deployment**

- Docker
- Docker Compose

## Quick start

### Option 1: Run locally without Docker

1. Clone the repository.
2. Set up the backend environment.
3. Install frontend dependencies.
4. Download the dataset and train models.
5. Start backend and frontend.

**Backend**

```bash
cd backend
conda env create -f environment.yml
conda activate mlproj
poetry install
```

**Frontend**

```bash
cd frontend/ml
npm ci
```

**Download data and train models**

```bash
python download_data.py
python backend/train_models.py
```

**Start the backend**

```bash
cd backend
python api.py
```

Backend URL: `http://localhost:8000`

**Start the frontend**

```bash
cd frontend/ml
npm run dev
```

Frontend URL: `http://localhost:5173`

### Option 2: Run with Docker Compose

For a fresh local environment, this is the fastest path:

```bash
docker compose up --build
```

This development stack:

- builds backend and frontend images locally
- exposes the backend on `http://localhost:8000`
- exposes the frontend on `http://localhost:5173`
- mounts persistent volumes for `Data/` and `models/`
- enables backend auto-bootstrap so missing data and artifacts can be created on first run

## Running with Docker

### Development Compose

Use the root `docker-compose.yml` for local development:

```bash
docker compose up --build
```

Services:

- `backend`: built from `backend/Dockerfile`
- `frontend`: built from `frontend/ml/Dockerfile.dev`

### Production Compose

Use `docker-compose.prod.yml` when you already have prebuilt images. This is the production stack used on AWS EC2:

- `BACKEND_IMAGE` required
- `FRONTEND_IMAGE` required
- `AUTO_BOOTSTRAP` optional, defaults to `true`
- `FRONTEND_PORT` optional, defaults to `80`

Example:

```bash
BACKEND_IMAGE=ghcr.io/OWNER/REPO-backend:latest \
FRONTEND_IMAGE=ghcr.io/OWNER/REPO-frontend:latest \
docker compose -f docker-compose.prod.yml up -d
```

The production frontend uses Nginx and proxies:

- `/api`
- `/docs`
- `/redoc`
- `/openapi.json`

## Data and model pipeline

The repository does not commit EEG `.edf` files or trained model artifacts. Instead:

- metadata and record lists live under `Data/`
- EEG files are downloaded by `download_data.py`
- model artifacts are written to `models/`

### Dataset bootstrap

`download_data.py` downloads EEG files from PhysioNet into `Data/eegmat/`.

### Training pipeline

`backend/train_models.py` performs the following high-level steps:

1. loads subject metadata and EEG files
2. windows raw EEG signals
3. downsamples the windows
4. trains Random Forest baseline and augmented models
5. trains CNN baseline and augmented models when PyTorch is available
6. stores trained artifacts, metrics, and dataset summaries in `models/`

Artifacts include:

- Random Forest model pickles
- CNN weights and configs
- metrics files
- dataset summary metadata
- saved train/test arrays used by visualization endpoints

## API overview

The backend is served through FastAPI.

**Base URL**

- local backend: `http://localhost:8000`
- production docs via frontend proxy: `http://<host>/docs`

**Important notes**

- if trained artifacts are missing, the API still starts
- endpoints that require models can return `503` until training is completed
- `GET /api/health` is the quickest readiness check

**Core endpoints**

- `GET /` - basic service status
- `GET /api/health` - backend health and model availability
- `GET /api/dashboard` - summary metrics for the dashboard
- `GET /api/performance` - model performance metrics
- `GET /api/visualization/umap` - UMAP projection data
- `GET /api/visualization/channel-importance` - feature importance aggregated by channel
- `GET /api/visualization/eeg-signal/{channel}` - example EEG waveform data
- `GET /api/dataset/info` - dataset summary information
- `POST /api/predict` - prediction from uploaded EEG file

**Interactive API docs**

- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Frontend overview

The frontend is a React SPA with routes for:

- `/` - dashboard
- `/performance` - model performance views
- `/visualization` - data visualization pages
- `/predictions` - EEG file upload and prediction flow
- `/about` - project and dataset overview

The UI consumes backend data from `/api/*` endpoints. In development, it runs with Vite. In production, it is served by Nginx.

## Quality checks

### Backend

```bash
cd backend
ruff format --check .
ruff check .
mypy . --config-file pyproject.toml
```

### Frontend

```bash
cd frontend/ml
npm run lint
npm run format:check
npm run type-check
npm run build
```

### CI

`.github/workflows/ci.yml` runs on pushes and pull requests and validates:

- backend formatting, linting, and typing
- frontend linting, formatting, type checks, and production build
- Docker Compose configuration for development and production stacks

## Deployment

`.github/workflows/cd.yml` supports automated container build and deployment to AWS EC2.

### What the CD workflow does

- builds backend and frontend images
- pushes them to GitHub Container Registry
- connects to the AWS EC2 instance over SSH
- uploads `docker-compose.prod.yml`
- writes deployment environment values on the host
- runs `docker compose pull`, `down`, and `up -d`

### Trigger behavior

The CD workflow runs on:

- pushes to `main`
- manual `workflow_dispatch`

### Required secrets

- `DEPLOY_HOST`
- `DEPLOY_USER`
- `DEPLOY_SSH_KEY`
- `REGISTRY_USERNAME`
- `REGISTRY_TOKEN`

### Optional secrets and variables

- `DEPLOY_PORT` default `22`
- `DEPLOY_PATH` default `ml-project`
- repository variable `FRONTEND_PORT` default `80`
- repository variable `AUTO_BOOTSTRAP` default `true`

### Production notes

- first production startup can take time if `AUTO_BOOTSTRAP=true`
- named Docker volumes preserve downloaded data and trained artifacts between restarts
- after the first successful bootstrap, setting `AUTO_BOOTSTRAP=false` can speed up future deploys
- the deployed EC2 host must already have Docker and Docker Compose available

## Troubleshooting

### API starts but endpoints fail with `503`

The models are not trained yet. Run:

```bash
python download_data.py
python backend/train_models.py
```

Or allow the backend container to bootstrap automatically on first Docker startup.

### CNN endpoints or predictions are unavailable

PyTorch or CNN artifacts may be missing. The backend can still serve Random Forest functionality even when CNN models are unavailable.

### Docker deployment is slow on first run

This is expected when `AUTO_BOOTSTRAP=true`, because the server may download EEG files and train models before the system is fully ready.

### Frontend loads but API calls fail

Check that:

- the backend is running on port `8000`
- the frontend is using the correct API URL in development
- the production reverse proxy is forwarding `/api` correctly

### CD succeeds but the EC2 app is not reachable

Verify on the target host:

```bash
cd ml-project
docker compose --env-file .env -f docker-compose.prod.yml ps
docker compose --env-file .env -f docker-compose.prod.yml logs --tail=100
```

Also test the live health endpoint:

```bash
curl http://<host>/api/health
```
