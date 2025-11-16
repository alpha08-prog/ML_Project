# Quick Start Guide

## Step 1: Train Models (Required First!)

Before starting the API, you **must** train the models:

```bash
python backend/train_models.py
```
 
This will:
- Load EEG data from `Data/eegmat/`
- Preprocess and create windows
- Train Random Forest models (baseline + augmented)
- Train CNN models (baseline + augmented)
- Save all models to `models/` directory

**Time:** ~5-10 minutes depending on your system

**Output:** Creates `models/` directory with:
- `dataset_info.pkl` - Dataset metadata
- `rf_baseline.pkl`, `rf_augmented.pkl` - Random Forest models
- `cnn_baseline.pt`, `cnn_augmented.pt` - CNN models
- `*_metrics.pkl` - Performance metrics
- `X_train.npy`, `X_test.npy`, etc. - Training/test data

## Step 2: Start Backend API

After models are trained:

```bash
python backend/api.py
```

Or use the convenience scripts:
- **Windows:** `start_backend.bat`
- **Linux/Mac:** `./start_backend.sh`

The API will be available at `http://localhost:8000`

## Step 3: Start Frontend

In a new terminal:

```bash
cd frontend/ml
npm install  # First time only
npm run dev
```

Frontend will be at `http://localhost:5173`

## Troubleshooting

### "Models not trained yet" error

**Solution:** Run the training script:
```bash
python backend/train_models.py
```

### "No such file or directory: models/dataset_info.pkl"

**Solution:** This means models haven't been trained. Run:
```bash
python backend/train_models.py
```

### PyTorch DLL error

**Solution:** The API will work with Random Forest models even if PyTorch fails.
- Use `model_type=rf` for predictions
- See `backend/fix_pytorch.md` for fixing PyTorch

## Verify Installation

1. Check models exist:
   ```bash
   ls models/  # Should show .pkl and .pt files
   ```

2. Test API health:
   ```bash
   curl http://localhost:8000/api/health
   ```

3. Check API docs:
   - Open: `http://localhost:8000/docs`

## What Each Script Does

- `backend/train_models.py` - Trains and saves all models
- `backend/api.py` - Starts FastAPI server
- `frontend/ml/` - React frontend application

