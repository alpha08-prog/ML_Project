# EEG Mental Arithmetic Classification - ML Project

A complete machine learning project for classifying EEG signals to determine mental arithmetic task performance quality, featuring a modern React frontend and FastAPI backend.

## 🚀 Quick Start

### 1. Conda Environment

Create the conda environment from the provided `environment.yml` (includes Python 3.12, NumPy, SciPy, PyTorch, and Poetry):
```bash
cd backend
conda env create -f environment.yml
conda activate mlproj
```

**For GPU users (NVIDIA CUDA 12.4):**
```bash
cd backend
conda env create -f environment-gpu.yml
conda activate mlproj
```

### 2. Install Python Packages (Poetry)

With the conda environment active, install the remaining Python dependencies via Poetry:
```bash
cd backend
poetry install
```

> **Note:** `poetry.toml` is configured with `virtualenvs.create = false`, so Poetry installs directly into the active conda environment.

### 3. Download EEG Data

The EEG dataset is not included in the repository. Download it from PhysioNet:
```bash
python download_data.py
```

This downloads ~170 MB of EDF files (72 files for 36 subjects) into `Data/eegmat/`.

### 4. Frontend Setup

Requires Node.js >= 20.19.0.
```bash
cd frontend/ml
npm install
```

### 5. Pre-commit Hooks

Pre-commit runs formatting, linting, and type checks automatically on every `git commit`.

```bash
brew install pre-commit       # install (once per machine)
pre-commit install            # activate hooks for this repo (once per clone)
```

To run all checks manually without committing:
```bash
pre-commit run --all-files
```

### 6. Train Models

```bash
python backend/train_models.py
```

This will train all models and save them to the `models/` directory (~5-10 minutes).

### 7. Start Backend

**Windows:**
```bash
start_backend.bat
```

**Linux/Mac:**
```bash
./start_backend.sh
```

Or manually:
```bash
cd backend
python api.py
```

Backend runs on `http://localhost:8000`

### 8. Start Frontend

```bash
cd frontend/ml
npm run dev
```

Frontend runs on `http://localhost:5173`

## 📁 Project Structure

```
ML_Project/
├── backend/
│   ├── api.py              # FastAPI backend server
│   ├── train_models.py     # Model training script
│   ├── pyproject.toml      # Poetry dependencies
│   ├── poetry.lock         # Poetry lock file
│   ├── poetry.toml         # Poetry config (no virtualenv)
│   ├── environment.yml     # Conda environment (CPU)
│   └── environment-gpu.yml # Conda environment (GPU/CUDA)
├── frontend/ml/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts      # API client service
│   │   ├── pages/          # React pages
│   │   └── components/     # React components
│   └── package.json
├── models/                 # Trained models (generated, gitignored)
├── Data/                   # EEG dataset (gitignored)
├── download_data.py        # Script to fetch EEG data from PhysioNet
├── .hooks/                 # Custom hook scripts (used by pre-commit)
│   └── check-branch-up-to-date.sh
├── .pre-commit-config.yaml # Pre-commit hook definitions
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI pipeline
├── .gitignore              # Git ignore rules
├── .gitattributes          # Git attributes
└── README.md
```

## 🎯 Features

### Frontend
- **Dashboard**: Overview metrics and visualizations
- **Model Performance**: Detailed comparison of baseline vs augmented models
- **Data Visualization**: Interactive EEG signals, UMAP embeddings, channel analysis
- **Predictions**: Upload EDF files and get real-time predictions
- **About**: Project documentation and methodology

### Backend
- **RESTful API**: FastAPI with automatic documentation
- **Model Serving**: Load and serve trained models
- **File Upload**: Process EDF files for predictions
- **Visualizations**: Generate UMAP embeddings and channel importance

## 📊 Models

The project trains and compares:

1. **Random Forest (Baseline)**: Traditional ML approach
2. **Random Forest (Augmented)**: With synthetic data
3. **CNN (Baseline)**: Deep learning approach
4. **CNN (Augmented)**: With synthetic data

All models are saved after training and can be used for predictions.

## 🔧 API Endpoints

- `GET /api/dashboard` - Dashboard metrics
- `GET /api/performance` - Model performance metrics
- `GET /api/visualization/umap` - UMAP embedding data
- `GET /api/visualization/channel-importance` - Channel analysis
- `GET /api/visualization/eeg-signal/{channel}` - EEG signal data
- `POST /api/predict` - Make predictions on uploaded files

Full API documentation: `http://localhost:8000/docs`

## 📚 Documentation

All key instructions are consolidated here. The FastAPI docs are available at runtime at `http://localhost:8000/docs`.

## 🛠️ Technologies

**Backend:**
- Python 3.12 (recommended)
- FastAPI
- PyTorch
- scikit-learn
- NumPy, Pandas, SciPy

**Frontend:**
- React 19
- TypeScript
- Vite
- Recharts
- React Router

## 📝 Notes

- Models are trained on EEG data from 36 subjects
- Class imbalance handled with synthetic data generation
- CNN features require a working PyTorch install; Random Forest works without PyTorch

## 🤝 Contributing

1. Install pre-commit hooks: `brew install pre-commit && pre-commit install`
2. Train models: `python backend/train_models.py`
3. Start backend: `python backend/api.py`
4. Start frontend: `cd frontend/ml && npm run dev`
5. Make changes and test — hooks run automatically on `git commit`

## 📄 License

This project is for educational/research purposes.
