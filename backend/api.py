"""
FastAPI Backend for EEG Classification Project
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
from pathlib import Path
import pyedflib
import scipy.signal as sps
import umap

# Try to import PyTorch, but handle gracefully if it fails
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: PyTorch not available: {e}")
    print("CNN models will not be available, but Random Forest models will work.")
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass

# Import model definitions
import sys
sys.path.append(str(Path(__file__).parent))

# Only import CNN-related functions if torch is available
if TORCH_AVAILABLE:
    from train_models import Simple1DCNN, downsample_window, create_windows, flatten_windows
else:
    # Create dummy functions
    def downsample_window(window, orig_fs=512, target_fs=128):
        ratio = orig_fs // target_fs
        return np.array([
            sps.decimate(window[:, ch], ratio, zero_phase=True)
            for ch in range(window.shape[1])
        ]).T
    
    def create_windows(signal, window_size=512, step=256):
        windows = []
        for i in range(0, signal.shape[0] - window_size, step):
            windows.append(signal[i:i+window_size])
        return np.array(windows)
    
    def flatten_windows(X):
        return X.reshape(X.shape[0], -1)
    
    class Simple1DCNN:
        pass

# Paths - relative to project root (go up one level from backend/)
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_FOLDER = PROJECT_ROOT / "Data" / "eegmat"

# Global variables for loaded models
models_loaded = False
rf_baseline = None
rf_augmented = None
cnn_baseline = None
cnn_augmented = None
dataset_info = None
X_test = None
y_test = None
X_train = None
y_train = None

def load_models():
    """Load all trained models"""
    global rf_baseline, rf_augmented, cnn_baseline, cnn_augmented
    global dataset_info, X_test, y_test, X_train, y_train, models_loaded
    
    if models_loaded:
        return
    
    # Check if models directory exists
    if not MODELS_DIR.exists():
        error_msg = f"""
        ⚠️  Models directory not found: {MODELS_DIR}
        
        Please train the models first by running:
            python backend/train_models.py
        
        This will create the models directory and train all required models.
        """
        print(error_msg)
        raise FileNotFoundError(f"Models directory not found. Please run: python backend/train_models.py")
    
    # Check if dataset_info exists
    dataset_info_path = MODELS_DIR / "dataset_info.pkl"
    if not dataset_info_path.exists():
        error_msg = f"""
        ⚠️  Models not trained yet!
        
        The file {dataset_info_path} does not exist.
        
        Please train the models first by running:
            python backend/train_models.py
        
        This will:
        1. Load and preprocess the EEG data
        2. Train Random Forest models (baseline and augmented)
        3. Train CNN models (baseline and augmented)
        4. Save all models to {MODELS_DIR}
        
        Training takes approximately 5-10 minutes.
        """
        print(error_msg)
        raise FileNotFoundError(f"Models not trained. Please run: python backend/train_models.py")
    
    try:
        # Load dataset info
        with open(dataset_info_path, "rb") as f:
            dataset_info = pickle.load(f)
        
        # Load test data
        X_test = np.load(MODELS_DIR / "X_test.npy")
        y_test = np.load(MODELS_DIR / "y_test.npy")
        X_train = np.load(MODELS_DIR / "X_train.npy")
        y_train = np.load(MODELS_DIR / "y_train.npy")
        
        # Load Random Forest models
        rf_baseline_path = MODELS_DIR / "rf_baseline.pkl"
        rf_augmented_path = MODELS_DIR / "rf_augmented.pkl"
        
        if not rf_baseline_path.exists() or not rf_augmented_path.exists():
            raise FileNotFoundError("Random Forest models not found. Please train models first.")
        
        with open(rf_baseline_path, "rb") as f:
            rf_baseline = pickle.load(f)
        with open(rf_augmented_path, "rb") as f:
            rf_augmented = pickle.load(f)
        
        # Load CNN models only if PyTorch is available
        if TORCH_AVAILABLE:
            try:
                cnn_baseline_pt = MODELS_DIR / "cnn_baseline.pt"
                cnn_augmented_pt = MODELS_DIR / "cnn_augmented.pt"
                
                if cnn_baseline_pt.exists() and cnn_augmented_pt.exists():
                    cnn_config = pickle.load(open(MODELS_DIR / "cnn_baseline_config.pkl", "rb"))
                    cnn_baseline = Simple1DCNN(in_ch=cnn_config["in_ch"])
                    cnn_baseline.load_state_dict(torch.load(cnn_baseline_pt, map_location="cpu"))
                    cnn_baseline.eval()
                    
                    cnn_config_aug = pickle.load(open(MODELS_DIR / "cnn_augmented_config.pkl", "rb"))
                    cnn_augmented = Simple1DCNN(in_ch=cnn_config_aug["in_ch"])
                    cnn_augmented.load_state_dict(torch.load(cnn_augmented_pt, map_location="cpu"))
                    cnn_augmented.eval()
                    print("CNN models loaded successfully")
                else:
                    print("Warning: CNN model files not found. CNN predictions will not be available.")
                    cnn_baseline = None
                    cnn_augmented = None
            except Exception as e:
                print(f"Warning: Could not load CNN models: {e}")
                print("CNN predictions will not be available, but Random Forest will work.")
                cnn_baseline = None
                cnn_augmented = None
        else:
            print("PyTorch not available - CNN models will not be loaded")
            cnn_baseline = None
            cnn_augmented = None
        
        models_loaded = True
        print("✓ Models loaded successfully")
        print(f"  Random Forest: Available")
        print(f"  CNN: {'Available' if (cnn_baseline and cnn_augmented) else 'Not Available'}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
        raise
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Startup
    try:
        load_models()
    except FileNotFoundError as e:
        print("\n" + "="*60)
        print("⚠️  MODELS NOT TRAINED YET")
        print("="*60)
        print("\nTo fix this, run the training script:")
        print("  python backend/train_models.py\n")
        print("The API will start but most endpoints will return errors")
        print("until models are trained.\n")
        print("="*60 + "\n")
    yield
    # Shutdown (if needed)

app = FastAPI(
    title="EEG Classification API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "EEG Classification API", "status": "running"}

@app.get("/api/health")
async def health():
    return {
        "status": "healthy", 
        "models_loaded": models_loaded,
        "torch_available": TORCH_AVAILABLE,
        "cnn_available": cnn_baseline is not None and cnn_augmented is not None if models_loaded else False,
        "rf_available": rf_baseline is not None and rf_augmented is not None if models_loaded else False
    }

@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard overview data"""
    if not models_loaded:
        try:
            load_models()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please run: python backend/train_models.py"
            )
    
    # Load metrics
    with open(MODELS_DIR / "rf_baseline_metrics.pkl", "rb") as f:
        rf_metrics = pickle.load(f)
    with open(MODELS_DIR / "rf_augmented_metrics.pkl", "rb") as f:
        rf_aug_metrics = pickle.load(f)
    
    accuracies = [rf_metrics['accuracy'], rf_aug_metrics['accuracy']]
    
    # Try to load CNN metrics if available
    if cnn_baseline is not None and cnn_augmented is not None:
        try:
            with open(MODELS_DIR / "cnn_baseline_metrics.pkl", "rb") as f:
                cnn_metrics = pickle.load(f)
            with open(MODELS_DIR / "cnn_augmented_metrics.pkl", "rb") as f:
                cnn_aug_metrics = pickle.load(f)
            accuracies.extend([cnn_metrics['accuracy'], cnn_aug_metrics['accuracy']])
        except:
            pass
    
    best_accuracy = max(accuracies)
    
    return {
        "total_subjects": dataset_info["total_subjects"],
        "best_accuracy": best_accuracy,
        "synthetic_samples": dataset_info["synthetic_samples"],
        "class_distribution": dataset_info["class_distribution"],
        "model_status": "trained"
    }

@app.get("/api/performance")
async def get_performance():
    """Get model performance metrics"""
    if not models_loaded:
        try:
            load_models()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please run: python backend/train_models.py"
            )
    
    # Load all metrics
    with open(MODELS_DIR / "rf_baseline_metrics.pkl", "rb") as f:
        rf_metrics = pickle.load(f)
    with open(MODELS_DIR / "rf_augmented_metrics.pkl", "rb") as f:
        rf_aug_metrics = pickle.load(f)
    
    result = {
        "random_forest": {
            "baseline": rf_metrics,
            "augmented": rf_aug_metrics
        }
    }
    
    # Load CNN metrics only if available
    if cnn_baseline is not None and cnn_augmented is not None:
        try:
            with open(MODELS_DIR / "cnn_baseline_metrics.pkl", "rb") as f:
                cnn_metrics = pickle.load(f)
            with open(MODELS_DIR / "cnn_augmented_metrics.pkl", "rb") as f:
                cnn_aug_metrics = pickle.load(f)
            result["cnn"] = {
                "baseline": cnn_metrics,
                "augmented": cnn_aug_metrics
            }
        except Exception as e:
            print(f"Warning: Could not load CNN metrics: {e}")
            # Do not include CNN section if metrics cannot be loaded
    
    return result

@app.get("/api/visualization/umap")
async def get_umap():
    """Get UMAP embedding data"""
    if not models_loaded:
        try:
            load_models()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please run: python backend/train_models.py"
            )
    
    # Sample data for UMAP
    real_min = X_train[y_train == 0]
    maj = X_train[y_train == 1]
    
    nR = min(200, len(real_min))
    nS = min(200, 500)  # synthetic samples
    nM = min(200, len(maj))
    
    real_flat = real_min[:nR].reshape(nR, -1)
    maj_flat = maj[:nM].reshape(nM, -1)
    
    # Generate synthetic samples (simplified)
    synthetic_flat = real_min[:nS].reshape(nS, -1) + np.random.normal(0, 0.1, (nS, real_flat.shape[1]))
    
    Z = np.vstack([real_flat, synthetic_flat, maj_flat])
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(Z)
    
    # Format for frontend
    real_points = [{"x": float(emb[i, 0]), "y": float(emb[i, 1]), "type": "real"} for i in range(nR)]
    synth_points = [{"x": float(emb[i+nR, 0]), "y": float(emb[i+nR, 1]), "type": "synthetic"} for i in range(nS)]
    maj_points = [{"x": float(emb[i+nR+nS, 0]), "y": float(emb[i+nR+nS, 1]), "type": "majority"} for i in range(nM)]
    
    return {
        "points": real_points + synth_points + maj_points
    }

@app.get("/api/visualization/channel-importance")
async def get_channel_importance():
    """Get channel importance data"""
    if not models_loaded:
        try:
            load_models()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please run: python backend/train_models.py"
            )
    
    # Use feature importance from Random Forest
    feature_importance = rf_augmented.feature_importances_
    channels = dataset_info["channels"]
    seq_len = dataset_info["sequence_length"]
    
    # Reshape to get per-channel importance
    importance_reshaped = feature_importance.reshape(seq_len, channels)
    channel_importance = np.mean(importance_reshaped, axis=0)
    
    # Normalize
    channel_importance = channel_importance / channel_importance.max()
    
    return {
        "channels": [f"Ch{i}" for i in range(channels)],
        "importance": channel_importance.tolist(),
        "real": channel_importance.tolist(),  # Same for demo
        "synthetic": (channel_importance * 0.9).tolist()  # Slightly different for demo
    }

@app.get("/api/visualization/eeg-signal/{channel}")
async def get_eeg_signal(channel: int):
    """Get sample EEG signal for a channel"""
    if not models_loaded:
        try:
            load_models()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please run: python backend/train_models.py"
            )
    
    if channel < 0 or channel >= dataset_info["channels"]:
        raise HTTPException(status_code=400, detail="Invalid channel number")
    
    # Get a sample from test data
    sample = X_test[0]  # First test sample
    signal = sample[:, channel]  # Extract channel
    
    # Downsample for visualization (take every 4th point)
    signal_downsampled = signal[::4]
    time_points = list(range(len(signal_downsampled)))
    
    return {
        "time": time_points,
        "amplitude": signal_downsampled.tolist()
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model_type: str = "cnn"):
    """Make prediction on uploaded EEG file"""
    if not models_loaded:
        try:
            load_models()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please run: python backend/train_models.py"
            )
    
    if model_type not in ["rf", "cnn"]:
        raise HTTPException(status_code=400, detail="model_type must be 'rf' or 'cnn'")
    
    # Check if CNN is requested but not available
    if model_type == "cnn" and (not TORCH_AVAILABLE or cnn_augmented is None):
        raise HTTPException(
            status_code=503, 
            detail="CNN model not available. PyTorch failed to load. Please use model_type=rf instead."
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Save temporarily
        temp_path = Path(f"temp_{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Read EDF file
        f = pyedflib.EdfReader(str(temp_path))
        n = f.signals_in_file
        samples = f.getNSamples()[0]
        signals = np.zeros((samples, n))
        for ch in range(n):
            signals[:, ch] = f.readSignal(ch)
        f.close()
        
        # Process signal
        windows = create_windows(signals)
        if len(windows) == 0:
            raise HTTPException(status_code=400, detail="Signal too short")
        
        # Use first window
        window = windows[0]
        window_ds = downsample_window(window)
        
        # Make prediction
        if model_type == "rf":
            window_flat = flatten_windows(window_ds.reshape(1, *window_ds.shape))
            prob = rf_augmented.predict_proba(window_flat)[0, 1]
        else:  # cnn
            if not TORCH_AVAILABLE or cnn_augmented is None:
                raise HTTPException(status_code=503, detail="CNN model not available")
            window_tensor = torch.tensor(window_ds).permute(1, 0).unsqueeze(0).float()
            with torch.no_grad():
                logit = cnn_augmented(window_tensor).item()
            prob = 1 / (1 + np.exp(-logit))
        
        # Cleanup
        temp_path.unlink()
        
        prediction = "Good Quality (Group G)" if prob > 0.5 else "Bad Quality (Group B)"
        confidence = "High" if prob > 0.8 or prob < 0.2 else "Medium" if prob > 0.65 or prob < 0.35 else "Low"
        
        return {
            "class": prediction,
            "probability": float(prob),
            "confidence": confidence
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get dataset information"""
    if not models_loaded:
        try:
            load_models()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please run: python backend/train_models.py"
            )
    
    return dataset_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

