# Frontend-Backend Integration Guide

This guide explains how to run the complete application with the frontend connected to the FastAPI backend.

## Architecture

```
┌─────────────┐         HTTP/REST         ┌─────────────┐
│   Frontend  │ ◄───────────────────────► │   Backend   │
│  (React)    │                            │  (FastAPI)  │
│ Port: 5173  │                            │ Port: 8000 │
└─────────────┘                            └─────────────┘
                                                   │
                                                   ▼
                                            ┌─────────────┐
                                            │   Models    │
                                            │  Directory  │
                                            └─────────────┘
```

## Quick Start

### 1. Train Models (First Time Only)

```bash
python backend/train_models.py
```

This will:
- Process all EEG data
- Train Random Forest and CNN models (baseline and augmented)
- Save models to `models/` directory
- Takes ~5-10 minutes depending on your system

### 2. Start Backend API

**Windows:**
```bash
start_backend.bat
```

**Linux/Mac:**
```bash
chmod +x start_backend.sh
./start_backend.sh
```

**Or manually:**
```bash
cd backend
python api.py
```

The API will be available at `http://localhost:8000`

### 3. Start Frontend

```bash
cd frontend/ml
npm install  # First time only
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

### Dashboard
- `GET /api/dashboard` - Overview metrics

### Model Performance
- `GET /api/performance` - All model metrics

### Visualizations
- `GET /api/visualization/umap` - UMAP embedding
- `GET /api/visualization/channel-importance` - Channel analysis
- `GET /api/visualization/eeg-signal/{channel}` - EEG signal data

### Predictions
- `POST /api/predict?model_type={rf|cnn}` - Upload EDF file and get prediction

## Frontend Configuration

The frontend is configured to connect to `http://localhost:8000` by default.

To change the API URL, create a `.env` file in `frontend/ml/`:

```env
VITE_API_URL=http://localhost:8000
```

## Troubleshooting

### Frontend can't connect to backend

1. **Check backend is running:**
   ```bash
   curl http://localhost:8000/api/health
   ```

2. **Check CORS settings** in `backend/api.py` - make sure your frontend URL is in `allow_origins`

3. **Check browser console** for CORS errors

### Models not found

1. **Train models first:**
   ```bash
   python backend/train_models.py
   ```

2. **Check models directory:**
   ```bash
   ls models/
   ```

### Prediction fails

1. **Check file format** - must be `.edf` format
2. **Check file size** - very large files may timeout
3. **Check backend logs** for error messages

## Development

### Backend Development

- API auto-reloads on code changes (if using `--reload` flag)
- Check logs in terminal for errors
- API docs available at `http://localhost:8000/docs`

### Frontend Development

- Hot module replacement enabled
- Check browser console for errors
- Network tab shows API calls

## Production Deployment

### Backend

1. Use a production ASGI server:
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app
   ```

2. Set up reverse proxy (nginx) for HTTPS

3. Configure environment variables

### Frontend

1. Build for production:
   ```bash
   npm run build
   ```

2. Serve static files with nginx or similar

3. Update API URL in production environment

## File Structure

```
ML_Project/
├── backend/
│   ├── api.py              # FastAPI application
│   ├── train_models.py     # Model training script
│   └── README.md
├── frontend/ml/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts      # API service client
│   │   └── pages/          # React pages
│   └── ...
├── models/                 # Trained models (generated)
├── Data/                  # EEG data
└── requirements.txt       # Python dependencies
```

## Next Steps

- Add authentication if needed
- Implement model versioning
- Add caching for expensive operations
- Set up monitoring and logging
- Add unit tests

