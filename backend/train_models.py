"""
Model Training Script
Trains and saves all models for the EEG classification project
"""
import pickle
import pyedflib
import numpy as np
import pandas as pd
import scipy.signal as sps
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_score, recall_score, f1_score, confusion_matrix
)
# Try to import PyTorch, but handle gracefully if it fails
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: PyTorch not available: {e}")
    print("CNN models will not be trained, but Random Forest models will work fine.")
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass
    class DataLoader:
        pass
    class TensorDataset:
        pass

# Paths - relative to project root
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "Data" / "eegmat"
INFO_CSV = PROJECT_ROOT / "Data" / "eegmat" / "subject-info.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def read_edf(file_path):
    """Read EDF file and return signals"""
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    samples = f.getNSamples()[0]
    signals = np.zeros((samples, n))
    for ch in range(n):
        signals[:, ch] = f.readSignal(ch)
    f.close()
    return signals  # (T, C)

def create_windows(signal, window_size=512, step=256):
    """Create sliding windows from signal"""
    windows = []
    for i in range(0, signal.shape[0] - window_size, step):
        windows.append(signal[i:i+window_size])
    return np.array(windows)

def downsample_window(window, orig_fs=512, target_fs=128):
    """Downsample window to target frequency"""
    ratio = orig_fs // target_fs
    return np.array([
        sps.decimate(window[:, ch], ratio, zero_phase=True)
        for ch in range(window.shape[1])
    ]).T

def flatten_windows(X):
    """Flatten windows for RandomForest"""
    return X.reshape(X.shape[0], -1)

def metrics(y_true, probs, preds):
    """Calculate classification metrics"""
    return {
        'accuracy': float((y_true == preds).mean()),
        'roc_auc': float(roc_auc_score(y_true, probs)),
        'pr_auc': float(average_precision_score(y_true, probs)),
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'f1': float(f1_score(y_true, preds, zero_division=0)),
        'cm': confusion_matrix(y_true, preds).tolist()
    }

# CNN Model Definition (only if PyTorch is available)
if TORCH_AVAILABLE:
    class Simple1DCNN(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_model(model, train_loader, epochs=10, device="cpu"):
        """Train CNN model"""
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model

    def eval_torch(model, loader, device="cpu"):
        """Evaluate CNN model"""
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                probs = 1/(1+np.exp(-logits))
                ys.append(yb.numpy())
                ps.append(probs)
        ys = np.concatenate(ys)
        ps = np.concatenate(ps)
        preds = (ps>=0.5).astype(int)
        return metrics(ys, ps, preds)

    def prepare_loader(X, y, batch=64, shuffle=True):
        """Prepare PyTorch DataLoader"""
        X_t = torch.tensor(X).permute(0,2,1).float()
        y_t = torch.tensor(y).float()
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)
else:
    # Dummy functions if PyTorch not available
    class Simple1DCNN:
        pass
    def train_model(*args, **kwargs):
        pass
    def eval_torch(*args, **kwargs):
        pass
    def prepare_loader(*args, **kwargs):
        pass

def main():
    print("=" * 60)
    print("EEG Classification Model Training")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading data...")
    info = pd.read_csv(str(INFO_CSV))
    X, y = [], []
    
    for _, row in tqdm(info.iterrows(), total=len(info)):
        subj = str(row["Subject"]).zfill(2)
        label = row["Count quality"]
        f1 = DATA_FOLDER / f"Subject{subj}_1.edf"
        f2 = DATA_FOLDER / f"Subject{subj}_2.edf"
        
        if not f1.exists() or not f2.exists():
            continue
        
        baseline = read_edf(str(f1))
        task = read_edf(str(f2))
        combined = np.concatenate([baseline, task], axis=0)
        X.append(combined)
        y.append(label)
    
    X = np.array(X, dtype=object)
    y = np.array(y)
    print(f"Loaded {len(X)} subjects")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create windows
    print("\n[2/6] Creating windows...")
    all_windows, all_labels = [], []
    for signal, label in zip(X, y):
        wins = create_windows(signal)
        for w in wins:
            all_windows.append(w)
            all_labels.append(label)
    
    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    print(f"Total windows: {all_windows.shape}")
    
    # Downsample
    print("\n[3/6] Downsampling...")
    X_ds = np.array([downsample_window(w) for w in tqdm(all_windows)])
    print(f"Downsampled shape: {X_ds.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_ds, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    
    # Save test data for API
    np.save(MODELS_DIR / "X_test.npy", X_test)
    np.save(MODELS_DIR / "y_test.npy", y_test)
    np.save(MODELS_DIR / "X_train.npy", X_train)
    np.save(MODELS_DIR / "y_train.npy", y_train)
    
    # Train Random Forest Baseline
    print("\n[4/6] Training Random Forest (Baseline)...")
    X_train_flat = flatten_windows(X_train)
    X_test_flat = flatten_windows(X_test)
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train)
    probs_rf = rf.predict_proba(X_test_flat)[:,1]
    preds_rf = rf.predict(X_test_flat)
    metrics_rf = metrics(y_test, probs_rf, preds_rf)
    
    # Save RF baseline
    with open(MODELS_DIR / "rf_baseline.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(MODELS_DIR / "rf_baseline_metrics.pkl", "wb") as f:
        pickle.dump(metrics_rf, f)
    print(f"RF Baseline - Accuracy: {metrics_rf['accuracy']:.4f}, F1: {metrics_rf['f1']:.4f}")
    
    # Generate synthetic data (simplified - using SMOTE-like approach for demo)
    # In production, you'd train a GAN here
    print("\n[5/6] Generating synthetic data...")
    minor_idx = np.where(y_train == 0)[0]
    X_min = X_train[minor_idx]
    # Simple augmentation: add noise to minority samples
    synthetic = X_min.copy()
    for i in range(500 - len(X_min)):
        idx = i % len(X_min)
        noise = np.random.normal(0, 0.1, X_min[idx].shape)
        synthetic = np.vstack([synthetic, X_min[idx:idx+1] + noise])
    synthetic = synthetic[:500]
    print(f"Generated {len(synthetic)} synthetic samples")
    
    # Train Random Forest Augmented
    X_train_aug = np.concatenate([X_train, synthetic], axis=0)
    y_train_aug = np.concatenate([y_train, np.zeros(len(synthetic), dtype=int)], axis=0)
    X_train_flat_aug = flatten_windows(X_train_aug)
    
    rf_aug = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_aug.fit(X_train_flat_aug, y_train_aug)
    probs_rf_aug = rf_aug.predict_proba(X_test_flat)[:,1]
    preds_rf_aug = rf_aug.predict(X_test_flat)
    metrics_rf_aug = metrics(y_test, probs_rf_aug, preds_rf_aug)
    
    # Save RF augmented
    with open(MODELS_DIR / "rf_augmented.pkl", "wb") as f:
        pickle.dump(rf_aug, f)
    with open(MODELS_DIR / "rf_augmented_metrics.pkl", "wb") as f:
        pickle.dump(metrics_rf_aug, f)
    print(f"RF Augmented - Accuracy: {metrics_rf_aug['accuracy']:.4f}, F1: {metrics_rf_aug['f1']:.4f}")
    
    # Train CNN models (only if PyTorch is available)
    if TORCH_AVAILABLE:
        print("\n[6/6] Training CNN models...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            train_loader_orig = prepare_loader(X_train, y_train, batch=64)
            train_loader_aug = prepare_loader(X_train_aug, y_train_aug, batch=64)
            test_loader = prepare_loader(X_test, y_test, batch=128, shuffle=False)
            
            # CNN Baseline
            cnn = Simple1DCNN(in_ch=X_train.shape[2])
            cnn = train_model(cnn, train_loader_orig, epochs=8, device=device)
            metrics_cnn = eval_torch(cnn, test_loader, device=device)
            torch.save(cnn.state_dict(), str(MODELS_DIR / "cnn_baseline.pt"))
            with open(MODELS_DIR / "cnn_baseline_metrics.pkl", "wb") as f:
                pickle.dump(metrics_cnn, f)
            with open(MODELS_DIR / "cnn_baseline_config.pkl", "wb") as f:
                pickle.dump({"in_ch": X_train.shape[2]}, f)
            print(f"CNN Baseline - Accuracy: {metrics_cnn['accuracy']:.4f}, F1: {metrics_cnn['f1']:.4f}")
            
            # CNN Augmented
            cnn_aug = Simple1DCNN(in_ch=X_train.shape[2])
            cnn_aug = train_model(cnn_aug, train_loader_aug, epochs=8, device=device)
            metrics_cnn_aug = eval_torch(cnn_aug, test_loader, device=device)
            torch.save(cnn_aug.state_dict(), str(MODELS_DIR / "cnn_augmented.pt"))
            with open(MODELS_DIR / "cnn_augmented_metrics.pkl", "wb") as f:
                pickle.dump(metrics_cnn_aug, f)
            with open(MODELS_DIR / "cnn_augmented_config.pkl", "wb") as f:
                pickle.dump({"in_ch": X_train.shape[2]}, f)
            print(f"CNN Augmented - Accuracy: {metrics_cnn_aug['accuracy']:.4f}, F1: {metrics_cnn_aug['f1']:.4f}")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not train CNN models: {e}")
            print("Random Forest models have been trained successfully.")
            print("CNN models will not be available, but RF models will work.")
    else:
        print("\n[6/6] Skipping CNN models (PyTorch not available)")
        print("Random Forest models have been trained successfully.")
        print("To train CNN models, fix PyTorch installation (see backend/fix_pytorch.md)")
    
    # Save dataset info
    dataset_info = {
        "total_subjects": len(X),
        "class_distribution": {"good": int(np.sum(y == 1)), "bad": int(np.sum(y == 0))},
        "total_windows": len(all_windows),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "synthetic_samples": len(synthetic),
        "channels": X_train.shape[2],
        "sequence_length": X_train.shape[1]
    }
    with open(MODELS_DIR / "dataset_info.pkl", "wb") as f:
        pickle.dump(dataset_info, f)
    
    print("\n" + "=" * 60)
    print("Training complete! Models saved to models/ directory")
    print("=" * 60)

if __name__ == "__main__":
    main()

