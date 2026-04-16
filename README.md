# EEG Mental Arithmetic Classification

This project classifies EEG signals for mental arithmetic performance quality with a FastAPI backend and a React frontend.

## Project layout

```text
ML_Project/
|-- backend/           FastAPI API, training script, Python dependencies
|-- frontend/ml/       React + Vite frontend
|-- Data/              EEG metadata plus downloaded EDF files at runtime
|-- models/            Trained model artifacts generated locally or on first deploy
|-- .github/workflows/ CI and CD GitHub Actions workflows
|-- docker-compose.yml Development Docker Compose stack
|-- docker-compose.prod.yml Production Docker Compose stack
```

## Local development

### Backend

```bash
cd backend
conda env create -f environment.yml
conda activate mlproj
poetry install
python api.py
```

The API runs on `http://localhost:8000`.

### Frontend

```bash
cd frontend/ml
npm ci
npm run dev
```

The frontend runs on `http://localhost:5173`.

### Data and model training

The repository does not store EEG `.edf` files or trained model artifacts.

```bash
python download_data.py
python backend/train_models.py
```

Artifacts are written to `Data/eegmat/` and `models/`.

## Docker workflows

### Development Compose

```bash
docker compose up --build
```

This stack keeps the frontend in Vite dev mode and enables backend auto-bootstrap so a fresh environment can download data and train models on first run.

### Production Compose

`docker-compose.prod.yml` expects prebuilt images and persistent volumes:

- `BACKEND_IMAGE`
- `FRONTEND_IMAGE`
- Optional: `AUTO_BOOTSTRAP` (defaults to `true`)
- Optional: `FRONTEND_PORT` (defaults to `80`)

Example:

```bash
BACKEND_IMAGE=ghcr.io/OWNER/REPO-backend:latest \
FRONTEND_IMAGE=ghcr.io/OWNER/REPO-frontend:latest \
docker compose -f docker-compose.prod.yml up -d
```

The production frontend is served by Nginx and proxies `/api`, `/docs`, `/redoc`, and `/openapi.json` to the backend container.

## CI pipeline

`.github/workflows/ci.yml` runs on every push and pull request and checks:

- Backend formatting, linting, and mypy
- Frontend lint, formatting, type checking, and production build
- Docker Compose configuration validation for development and production stacks

## CD pipeline

`.github/workflows/cd.yml` runs after the `CI` workflow succeeds on `main`, and it can also be started manually with `workflow_dispatch`.

It does two things:

1. Builds and pushes backend and frontend images to GitHub Container Registry.
2. Optionally deploys them to a Linux host over SSH with Docker Compose.

The deploy job is enabled when these GitHub repository secrets exist:

- `DEPLOY_HOST`
- `DEPLOY_USER`
- `DEPLOY_SSH_KEY`
- `REGISTRY_USERNAME`
- `REGISTRY_TOKEN`

Optional secrets and variables:

- `DEPLOY_PORT` default `22`
- `DEPLOY_PATH` default `ml-project`
- Repository variable `FRONTEND_PORT` default `80`
- Repository variable `AUTO_BOOTSTRAP` default `true`

## Notes

- First production startup can take time if `AUTO_BOOTSTRAP=true`, because the backend may download the EEG dataset and train models.
- For faster deploys, keep the named volumes and switch `AUTO_BOOTSTRAP=false` after the first successful bootstrap.
- If you want a different deployment target later, the current CD workflow is easy to adapt because the image publish stage is already separated from the deploy stage.
