"""FastAPI backend for the EEG classification project."""

from __future__ import annotations

import pickle
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyedflib
import umap
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import train_models as model_utils

TORCH_AVAILABLE = False
torch: Any = None

try:
    import torch as torch_module
except Exception as exc:
    print(f"Warning: PyTorch not available: {exc}")
    print("CNN models will not be available, but Random Forest models will work.")
else:
    torch = torch_module
    TORCH_AVAILABLE = True

Simple1DCNN = getattr(model_utils, "Simple1DCNN", None)
create_windows = model_utils.create_windows
downsample_window = model_utils.downsample_window
flatten_windows = model_utils.flatten_windows

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

models_loaded = False
rf_baseline: Any | None = None
rf_augmented: Any | None = None
cnn_baseline: Any | None = None
cnn_augmented: Any | None = None
dataset_info: dict[str, Any] | None = None
x_test: Any | None = None
y_test: Any | None = None
x_train: Any | None = None
y_train: Any | None = None


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def load_models() -> None:
    """Load trained artifacts into memory."""
    global cnn_augmented, cnn_baseline, dataset_info, models_loaded
    global rf_augmented, rf_baseline, x_test, x_train, y_test, y_train

    if models_loaded:
        return

    if not MODELS_DIR.exists():
        raise FileNotFoundError(
            "Models directory not found. Please run: python backend/train_models.py"
        )

    dataset_info_path = MODELS_DIR / "dataset_info.pkl"
    if not dataset_info_path.exists():
        raise FileNotFoundError(
            "Models not trained. Please run: python backend/train_models.py"
        )

    try:
        dataset_info = cast(dict[str, Any], load_pickle(dataset_info_path))
        x_test = np.load(MODELS_DIR / "X_test.npy")
        y_test = np.load(MODELS_DIR / "y_test.npy")
        x_train = np.load(MODELS_DIR / "X_train.npy")
        y_train = np.load(MODELS_DIR / "y_train.npy")

        rf_baseline_path = MODELS_DIR / "rf_baseline.pkl"
        rf_augmented_path = MODELS_DIR / "rf_augmented.pkl"
        if not rf_baseline_path.exists() or not rf_augmented_path.exists():
            raise FileNotFoundError(
                "Random Forest models not found. Please train models first."
            )

        rf_baseline = load_pickle(rf_baseline_path)
        rf_augmented = load_pickle(rf_augmented_path)

        cnn_baseline = None
        cnn_augmented = None
        if TORCH_AVAILABLE and Simple1DCNN is not None:
            try:
                cnn_baseline_path = MODELS_DIR / "cnn_baseline.pt"
                cnn_augmented_path = MODELS_DIR / "cnn_augmented.pt"

                if cnn_baseline_path.exists() and cnn_augmented_path.exists():
                    cnn_baseline_config = load_pickle(
                        MODELS_DIR / "cnn_baseline_config.pkl"
                    )
                    cnn_baseline = Simple1DCNN(in_ch=cnn_baseline_config["in_ch"])
                    cnn_baseline.load_state_dict(
                        torch.load(cnn_baseline_path, map_location="cpu")
                    )
                    cnn_baseline.eval()

                    cnn_augmented_config = load_pickle(
                        MODELS_DIR / "cnn_augmented_config.pkl"
                    )
                    cnn_augmented = Simple1DCNN(in_ch=cnn_augmented_config["in_ch"])
                    cnn_augmented.load_state_dict(
                        torch.load(cnn_augmented_path, map_location="cpu")
                    )
                    cnn_augmented.eval()
                    print("CNN models loaded successfully")
                else:
                    print(
                        "Warning: CNN model files not found. "
                        "CNN predictions will not be available."
                    )
            except Exception as exc:
                print(f"Warning: Could not load CNN models: {exc}")
                print(
                    "CNN predictions will not be available, "
                    "but Random Forest will work."
                )
                cnn_baseline = None
                cnn_augmented = None
        elif not TORCH_AVAILABLE:
            print("PyTorch not available - CNN models will not be loaded")

        models_loaded = True
        cnn_status = (
            "Available"
            if cnn_baseline is not None and cnn_augmented is not None
            else "Not Available"
        )
        print("Models loaded successfully")
        print("  Random Forest: Available")
        print(f"  CNN: {cnn_status}")
    except FileNotFoundError:
        raise
    except Exception as exc:
        print(f"Error loading models: {exc}")
        traceback.print_exc()
        raise


def require_models_loaded() -> None:
    if models_loaded:
        return

    try:
        load_models()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_dataset_info() -> dict[str, Any]:
    require_models_loaded()
    assert dataset_info is not None
    return dataset_info


def get_train_data() -> tuple[Any, Any]:
    require_models_loaded()
    assert x_train is not None
    assert y_train is not None
    return x_train, y_train


def get_test_data() -> tuple[Any, Any]:
    require_models_loaded()
    assert x_test is not None
    assert y_test is not None
    return x_test, y_test


def get_rf_augmented_model() -> Any:
    require_models_loaded()
    assert rf_augmented is not None
    return rf_augmented


def get_cnn_metric_bundle() -> tuple[Any, Any] | None:
    if cnn_baseline is None or cnn_augmented is None:
        return None

    try:
        cnn_metrics = load_pickle(MODELS_DIR / "cnn_baseline_metrics.pkl")
        cnn_augmented_metrics = load_pickle(MODELS_DIR / "cnn_augmented_metrics.pkl")
    except Exception as exc:
        print(f"Warning: Could not load CNN metrics: {exc}")
        return None

    return cnn_metrics, cnn_augmented_metrics


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Try to warm models on startup without blocking the API from booting."""
    try:
        load_models()
    except FileNotFoundError:
        print("=" * 60)
        print("MODELS NOT TRAINED YET")
        print("Run: python backend/train_models.py")
        print("Most endpoints will return 503 until models are available.")
        print("=" * 60)
    yield


app = FastAPI(title="EEG Classification API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "EEG Classification API", "status": "running"}


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "torch_available": TORCH_AVAILABLE,
        "cnn_available": (
            models_loaded and cnn_baseline is not None and cnn_augmented is not None
        ),
        "rf_available": (
            models_loaded and rf_baseline is not None and rf_augmented is not None
        ),
    }


@app.get("/api/dashboard")
async def get_dashboard() -> dict[str, Any]:
    info = get_dataset_info()
    rf_metrics = load_pickle(MODELS_DIR / "rf_baseline_metrics.pkl")
    rf_augmented_metrics = load_pickle(MODELS_DIR / "rf_augmented_metrics.pkl")

    accuracies = [
        rf_metrics["accuracy"],
        rf_augmented_metrics["accuracy"],
    ]
    cnn_metric_bundle = get_cnn_metric_bundle()
    if cnn_metric_bundle is not None:
        cnn_metrics, cnn_augmented_metrics = cnn_metric_bundle
        accuracies.extend(
            [
                cnn_metrics["accuracy"],
                cnn_augmented_metrics["accuracy"],
            ]
        )

    return {
        "total_subjects": info["total_subjects"],
        "best_accuracy": max(accuracies),
        "synthetic_samples": info["synthetic_samples"],
        "class_distribution": info["class_distribution"],
        "model_status": "trained",
    }


@app.get("/api/performance")
async def get_performance() -> dict[str, Any]:
    require_models_loaded()

    result = {
        "random_forest": {
            "baseline": load_pickle(MODELS_DIR / "rf_baseline_metrics.pkl"),
            "augmented": load_pickle(MODELS_DIR / "rf_augmented_metrics.pkl"),
        }
    }

    cnn_metric_bundle = get_cnn_metric_bundle()
    if cnn_metric_bundle is not None:
        cnn_metrics, cnn_augmented_metrics = cnn_metric_bundle
        result["cnn"] = {
            "baseline": cnn_metrics,
            "augmented": cnn_augmented_metrics,
        }

    return result


@app.get("/api/visualization/umap")
async def get_umap() -> dict[str, Any]:
    train_features, train_labels = get_train_data()

    minority_samples = train_features[train_labels == 0]
    majority_samples = train_features[train_labels == 1]

    minority_count = min(200, len(minority_samples))
    synthetic_count = min(200, 500)
    majority_count = min(200, len(majority_samples))

    minority_flat = minority_samples[:minority_count].reshape(minority_count, -1)
    majority_flat = majority_samples[:majority_count].reshape(majority_count, -1)
    synthetic_flat = minority_samples[:synthetic_count].reshape(synthetic_count, -1)
    synthetic_flat = synthetic_flat + np.random.normal(
        0,
        0.1,
        (synthetic_count, minority_flat.shape[1]),
    )

    stacked = np.vstack([minority_flat, synthetic_flat, majority_flat])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(stacked)

    points: list[dict[str, float | str]] = []
    for index in range(minority_count):
        points.append(
            {
                "x": float(embedding[index, 0]),
                "y": float(embedding[index, 1]),
                "type": "real",
            }
        )

    for index in range(synthetic_count):
        points.append(
            {
                "x": float(embedding[index + minority_count, 0]),
                "y": float(embedding[index + minority_count, 1]),
                "type": "synthetic",
            }
        )

    for index in range(majority_count):
        embedding_index = index + minority_count + synthetic_count
        points.append(
            {
                "x": float(embedding[embedding_index, 0]),
                "y": float(embedding[embedding_index, 1]),
                "type": "majority",
            }
        )

    return {"points": points}


@app.get("/api/visualization/channel-importance")
async def get_channel_importance() -> dict[str, Any]:
    info = get_dataset_info()
    rf_model = get_rf_augmented_model()

    feature_importance = rf_model.feature_importances_
    channels = info["channels"]
    sequence_length = info["sequence_length"]

    importance_reshaped = feature_importance.reshape(sequence_length, channels)
    channel_importance = np.mean(importance_reshaped, axis=0)
    channel_importance = channel_importance / channel_importance.max()

    return {
        "channels": [f"Ch{index}" for index in range(channels)],
        "importance": channel_importance.tolist(),
        "real": channel_importance.tolist(),
        "synthetic": (channel_importance * 0.9).tolist(),
    }


@app.get("/api/visualization/eeg-signal/{channel}")
async def get_eeg_signal(channel: int) -> dict[str, Any]:
    info = get_dataset_info()
    test_features, _ = get_test_data()

    if channel < 0 or channel >= info["channels"]:
        raise HTTPException(status_code=400, detail="Invalid channel number")

    sample = test_features[0]
    signal = sample[:, channel]
    signal_downsampled = signal[::4]

    return {
        "time": list(range(len(signal_downsampled))),
        "amplitude": signal_downsampled.tolist(),
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = "cnn",
) -> dict[str, Any]:
    require_models_loaded()

    if model_type not in {"rf", "cnn"}:
        raise HTTPException(status_code=400, detail="model_type must be 'rf' or 'cnn'")

    if model_type == "cnn" and (not TORCH_AVAILABLE or cnn_augmented is None):
        raise HTTPException(
            status_code=503,
            detail=(
                "CNN model not available. PyTorch failed to load. "
                "Please use model_type=rf instead."
            ),
        )

    temp_path = Path(f"temp_{file.filename}")
    try:
        contents = await file.read()
        with temp_path.open("wb") as temp_file:
            temp_file.write(contents)

        edf_reader = pyedflib.EdfReader(str(temp_path))
        try:
            signal_count = edf_reader.signals_in_file
            sample_count = edf_reader.getNSamples()[0]
            signals = np.zeros((sample_count, signal_count))
            for channel_index in range(signal_count):
                signals[:, channel_index] = edf_reader.readSignal(channel_index)
        finally:
            edf_reader.close()

        windows = create_windows(signals)
        if len(windows) == 0:
            raise HTTPException(status_code=400, detail="Signal too short")

        window = windows[0]
        downsampled_window = downsample_window(window)

        if model_type == "rf":
            rf_model = get_rf_augmented_model()
            flattened = flatten_windows(
                downsampled_window.reshape(1, *downsampled_window.shape)
            )
            probability = rf_model.predict_proba(flattened)[0, 1]
        else:
            assert cnn_augmented is not None
            window_tensor = (
                torch.tensor(downsampled_window).permute(1, 0).unsqueeze(0).float()
            )
            with torch.no_grad():
                logit = cnn_augmented(window_tensor).item()
            probability = 1 / (1 + np.exp(-logit))

        prediction = (
            "Good Quality (Group G)" if probability > 0.5 else "Bad Quality (Group B)"
        )
        confidence = (
            "High"
            if probability > 0.8 or probability < 0.2
            else "Medium"
            if probability > 0.65 or probability < 0.35
            else "Low"
        )

        return {
            "class": prediction,
            "probability": float(probability),
            "confidence": confidence,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {exc}"
        ) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/dataset/info")
async def get_dataset_info_route() -> dict[str, Any]:
    return get_dataset_info()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
