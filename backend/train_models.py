"""
Model Training Script
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
    precision_score, recall_score, f1_score
)

# -----------------------------
# PyTorch (optional)
# -----------------------------
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: PyTorch not available: {e}")

# -----------------------------
# Paths
# -----------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "Data" / "eegmat"
INFO_CSV = DATA_FOLDER / "subject-info.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utils
# -----------------------------
def read_edf(file_path):
    f = pyedflib.EdfReader(str(file_path))
    n = f.signals_in_file
    samples = f.getNSamples()[0]
    signals = np.zeros((samples, n))
    for ch in range(n):
        signals[:, ch] = f.readSignal(ch)
    f.close()
    return signals

def create_windows(signal, window_size=512, step=256):
    return np.array([
        signal[i:i + window_size]
        for i in range(0, signal.shape[0] - window_size, step)
    ])

def downsample_window(window, orig_fs=512, target_fs=128):
    ratio = orig_fs // target_fs
    return np.array([
        sps.decimate(window[:, ch], ratio, zero_phase=True)
        for ch in range(window.shape[1])
    ]).T

def flatten_windows(X):
    return X.reshape(X.shape[0], -1)

def metrics(y_true, probs, preds):
    return {
        'accuracy': float((y_true == preds).mean()),
        'roc_auc': float(roc_auc_score(y_true, probs)),
        'pr_auc': float(average_precision_score(y_true, probs)),
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'f1': float(f1_score(y_true, preds, zero_division=0)),
    }

# -----------------------------
# CNN
# -----------------------------
if TORCH_AVAILABLE:
    class Simple1DCNN(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 32, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def prepare_loader(X, y, batch=64, shuffle=True):
        X_t = torch.tensor(X).permute(0, 2, 1).float()
        y_t = torch.tensor(y).float()
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch, shuffle=shuffle)

    def train_model(model, loader, epochs=5, device="cpu"):
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model

    def eval_model(model, loader, device="cpu"):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb.to(device)).cpu().numpy()
                probs = 1 / (1 + np.exp(-logits))
                ys.append(yb.numpy())
                ps.append(probs)

        ys = np.concatenate(ys)
        ps = np.concatenate(ps)
        preds = (ps >= 0.5).astype(int)
        return metrics(ys, ps, preds)

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("\n[1/6] Loading data...")
    info = pd.read_csv(INFO_CSV)

    X, y = [], []
    for _, row in tqdm(info.iterrows(), total=len(info)):
        subj = str(row["Subject"]).zfill(2)
        label = int(row["Count quality"])

        f1 = DATA_FOLDER / f"Subject{subj}_1.edf"
        f2 = DATA_FOLDER / f"Subject{subj}_2.edf"

        if not f1.exists() or not f2.exists():
            continue

        sig = np.concatenate([read_edf(f1), read_edf(f2)], axis=0)
        X.append(sig)
        y.append(label)

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)

    print("Loaded:", len(X))

    print("\n[2/6] Windowing...")
    windows, labels = [], []
    for s, l in zip(X, y):
        for w in create_windows(s):
            windows.append(w)
            labels.append(l)

    windows = np.array(windows)
    labels = np.array(labels)

    print("\n[3/6] Downsampling...")
    X_ds = np.array([downsample_window(w) for w in tqdm(windows)])

    print("\n[4/6] Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_ds, labels, test_size=0.2, stratify=labels, random_state=42
    )

    X_train_f = flatten_windows(X_train)
    X_test_f = flatten_windows(X_test)

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    rf.fit(X_train_f, y_train)
    probs_rf = rf.predict_proba(X_test_f)[:, 1]
    preds_rf = rf.predict(X_test_f)

    metrics_rf = metrics(y_test, probs_rf, preds_rf)

    with open(MODELS_DIR / "rf_baseline_metrics.pkl", "wb") as f:
        pickle.dump(metrics_rf, f)

    probs = rf.predict_proba(X_test_f)[:, 1]
    preds = rf.predict(X_test_f)

    with open(MODELS_DIR / "rf_baseline.pkl", "wb") as f:
        pickle.dump(rf, f)

    print("RF Accuracy:", (preds == y_test).mean())

    print("\n[5/6] Augmentation...")
    X_min = X_train[y_train == 0]

    synthetic = []
    for i in range(200):
        idx = i % len(X_min)
        noise = np.random.normal(0, 0.1, X_min[idx].shape)
        synthetic.append(X_min[idx] + noise)

    synthetic = np.array(synthetic)

    X_aug = np.concatenate([X_train, synthetic])
    y_aug = np.concatenate([y_train, np.zeros(len(synthetic))])

    rf_aug = RandomForestClassifier(n_estimators=200)
    rf_aug.fit(flatten_windows(X_aug), y_aug)

    probs_rf_aug = rf_aug.predict_proba(X_test_f)[:, 1]
    preds_rf_aug = rf_aug.predict(X_test_f)

    metrics_rf_aug = metrics(y_test, probs_rf_aug, preds_rf_aug)

    with open(MODELS_DIR / "rf_augmented_metrics.pkl", "wb") as f:
        pickle.dump(metrics_rf_aug, f)
    

    with open(MODELS_DIR / "rf_augmented.pkl", "wb") as f:
        pickle.dump(rf_aug, f)

    print("\n[6/6] CNN...")
    if TORCH_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using:", device)

        train_loader = prepare_loader(X_train, y_train)
        test_loader = prepare_loader(X_test, y_test, shuffle=False)

        cnn = Simple1DCNN(X_train.shape[2])
        cnn = train_model(cnn, train_loader, device=device)

        print("CNN:", eval_model(cnn, test_loader, device))
    else:
        print("Skipping CNN")

    # -------------------------
    # SAVE FOR API (CRITICAL)
    # -------------------------
    dataset_info = {
        "total_subjects": len(X),
        "class_distribution": {
            "good": int(np.sum(y == 1)),
            "bad": int(np.sum(y == 0))
        },
        "total_windows": len(windows),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "synthetic_samples": len(synthetic),
        "channels": X_train.shape[2],
        "sequence_length": X_train.shape[1]
    }

    with open(MODELS_DIR / "dataset_info.pkl", "wb") as f:
        pickle.dump(dataset_info, f)

    np.save(MODELS_DIR / "X_test.npy", X_test)
    np.save(MODELS_DIR / "y_test.npy", y_test)
    np.save(MODELS_DIR / "X_train.npy", X_train)
    np.save(MODELS_DIR / "y_train.npy", y_train)

    print("\n✅ Training complete")

if __name__ == "__main__":
    main()