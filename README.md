# EEG Mental Arithmetic Classification - ML Project

A complete machine learning project for classifying EEG signals to determine mental arithmetic task performance quality, featuring a modern React frontend and FastAPI backend.

## 🚀 Quick Start

### 1. Environment & Dependencies

**Recommended (Conda, Python 3.12):**
```bash
# Install Miniconda (once), then in a new shell:
conda create -n mlproj python=3.12 -y
conda activate mlproj
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

PyTorch is optional but required for CNN features. If you have an NVIDIA GPU, install via the official channel; otherwise install CPU-only. Example (choose one):
```bash
# CPU-only
conda install -y -c pytorch pytorch torchvision torchaudio cpuonly

# Or CUDA 12.1
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Node.js (Frontend):**
Requires Node.js >= 20.19.0.
```bash
cd frontend/ml
npm install
```

### 2. Train Models

```bash
python backend/train_models.py
```

This will train all models and save them to the `models/` directory (~5-10 minutes).

### 3. Start Backend

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

### 4. Start Frontend

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
│   └── train_models.py     # Model training script
├── frontend/ml/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts      # API client service
│   │   ├── pages/          # React pages
│   │   └── components/     # React components
│   └── package.json
├── models/                 # Trained models (generated, gitignored)
├── Data/                   # EEG dataset (gitignored)
├── requirements.txt        # Python dependencies
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

1. Train models: `python backend/train_models.py`
2. Start backend: `python backend/api.py`
3. Start frontend: `cd frontend/ml && npm run dev`
4. Make changes and test

## 📄 License

This project is for educational/research purposes.
