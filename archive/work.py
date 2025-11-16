


# ============================================================
# STEP 1 — IMPORTS
# ============================================================
import os
import pyedflib
import numpy as np
import pandas as pd
import scipy.signal as sps
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# ============================================================


DATA_FOLDER = "Data/eegmat/"
INFO_CSV = "Data/eegmat/subject-info.csv"

info = pd.read_csv(INFO_CSV)

def read_edf(file_path):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    samples = f.getNSamples()[0]
    signals = np.zeros((samples, n))

    for ch in range(n):
        signals[:, ch] = f.readSignal(ch)

    f.close()
    return signals   # (T, C)


X = []
y = []

for _, row in tqdm(info.iterrows(), total=len(info)):
    subj = str(row["Subject"]).zfill(2)
    label = row["Count quality"]      # 0 or 1

    f1 = os.path.join(DATA_FOLDER, f"Subject{subj}_1.edf")
    f2 = os.path.join(DATA_FOLDER, f"Subject{subj}_2.edf")

    if not os.path.exists(f1) or not os.path.exists(f2):
        print("Missing:", f1, f2)
        continue

    baseline = read_edf(f1)         # (T, C)
    task     = read_edf(f2)

    combined = np.concatenate([baseline, task], axis=0)
    X.append(combined)
    y.append(label)

X = np.array(X, dtype=object)
y = np.array(y)

print("Subjects loaded:", len(X))
print("Class distribution:", np.bincount(y))


def create_windows(signal, window_size=512, step=256):
    windows = []
    for i in range(0, signal.shape[0] - window_size, step):
        windows.append(signal[i:i+window_size])
    return np.array(windows)

all_windows = []
all_labels = []

for signal, label in zip(X, y):
    wins = create_windows(signal)
    for w in wins:
        all_windows.append(w)
        all_labels.append(label)

all_windows = np.array(all_windows)     # (N, T, C)
all_labels = np.array(all_labels)

print("Total windows:", all_windows.shape)
print("Labels:", np.bincount(all_labels))


def downsample_window(window, orig_fs=512, target_fs=128):
    ratio = orig_fs // target_fs
    return np.array([
        sps.decimate(window[:, ch], ratio, zero_phase=True)
        for ch in range(window.shape[1])
    ]).T

X_ds = np.array([downsample_window(w) for w in all_windows])

print("Downsampled:", X_ds.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X_ds, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)


device = "cuda" if torch.cuda.is_available() else "cpu"

minor_idx = np.where(y_train == 0)[0]
X_min = X_train[minor_idx]      # (N_minority, T, C)

# For PyTorch: (N, C, T)
X_min_torch = torch.tensor(X_min).permute(0,2,1).float().to(device)

loader = DataLoader(
    TensorDataset(X_min_torch),
    batch_size=32,
    shuffle=True,
    drop_last=True
)

channels = X_min_torch.shape[1]
seq_len  = X_min_torch.shape[2]


latent_dim = 32

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, channels * seq_len),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, channels, seq_len)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def lipschitz_penalty(C, real):
    real = real.requires_grad_(True)
    pred = C(real)
    grads = torch.autograd.grad(pred.sum(), real, create_graph=True)[0]
    return (grads.norm(2, dim=(1,2)) ** 2).mean()


G = Generator().to(device)
C = Critic().to(device)

opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.9))
opt_C = optim.Adam(C.parameters(), lr=1e-4, betas=(0.5,0.9))

n_epochs = 50
lambda_lp = 0.5
n_critic = 1

for epoch in range(n_epochs):
    for real_batch, in loader:

        # Train Critic
        for _ in range(n_critic):
            z = torch.randn(real_batch.size(0), latent_dim, device=device)
            fake = G(z).detach()

            loss_C = -(C(real_batch).mean() - C(fake).mean())
            lp = lipschitz_penalty(C, real_batch)

            loss_C_total = loss_C + lambda_lp * lp

            opt_C.zero_grad()
            loss_C_total.backward()
            opt_C.step()

        # Train Generator
        z = torch.randn(real_batch.size(0), latent_dim, device=device)
        fake = G(z)
        loss_G = -C(fake).mean()

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{n_epochs}  |  Critic: {loss_C_total:.4f}  |  Gen: {loss_G:.4f}")


N_new = 500   # how many synthetic minority samples you want

z = torch.randn(N_new, latent_dim, device=device)
synthetic = G(z).detach().cpu().numpy()
synthetic = np.transpose(synthetic, (0,2,1))   # (N, T, C)

print("Synthetic samples:", synthetic.shape)


# === A.1 Prepare datasets ===
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix

# X_train, X_test shape: (N, T, C)
# y_train, y_test: (N,)
# synthetic: (N_synth, T, C) produced by GAN (from earlier cell)

# Flatten for RandomForest (feature-based)
def flatten_windows(X):
    return X.reshape(X.shape[0], -1)

X_train_flat = flatten_windows(X_train)        # original
X_test_flat  = flatten_windows(X_test)

# Augmented train sets
X_train_aug = np.concatenate([X_train, synthetic], axis=0)
y_train_aug = np.concatenate([y_train, np.zeros(len(synthetic), dtype=int)], axis=0)

X_train_flat_aug = flatten_windows(X_train_aug)

print("Train sizes: original", X_train.shape[0], "augmented", X_train_aug.shape[0])


# === A.2 Random Forest on flattened windows ===
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train)
probs_rf = rf.predict_proba(X_test_flat)[:,1]
preds_rf = rf.predict(X_test_flat)

def metrics(y_true, probs, preds):
    return {
        'accuracy': (y_true == preds).mean(),
        'roc_auc': roc_auc_score(y_true, probs),
        'pr_auc' : average_precision_score(y_true, probs),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
        'cm': confusion_matrix(y_true, preds)
    }

metrics_rf = metrics(y_test, probs_rf, preds_rf)
print("RF baseline:", metrics_rf)


# RF on augmented
rf_aug = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_aug.fit(X_train_flat_aug, y_train_aug)
probs_rf_aug = rf_aug.predict_proba(X_test_flat)[:,1]
preds_rf_aug = rf_aug.predict(X_test_flat)
metrics_rf_aug = metrics(y_test, probs_rf_aug, preds_rf_aug)
print("RF augmented:", metrics_rf_aug)


# === A.3 CNN classifier (PyTorch) ===
# Reuse Simple1DCNN from earlier if defined; otherwise define again
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert arrays -> tensors (N, C, T)
def prepare_loader(X, y, batch=64, shuffle=True):
    X_t = torch.tensor(X).permute(0,2,1).float()   # (N, C, T)
    y_t = torch.tensor(y).float()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

train_loader_orig = prepare_loader(X_train, y_train, batch=64)
train_loader_aug  = prepare_loader(X_train_aug, y_train_aug, batch=64)
test_loader = prepare_loader(X_test, y_test, batch=128, shuffle=False)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
def train_model(model, train_loader, epochs=10):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        tot = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(xb)
        # quick eval
    return model

def eval_torch(model, loader):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            probs = 1/(1+np.exp(-logits))
            ys.append(yb.numpy()); ps.append(probs)
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    preds = (ps>=0.5).astype(int)
    return metrics(ys, ps, preds)

# Train baseline CNN
cnn = Simple1DCNN(in_ch=X_train.shape[2])
cnn = train_model(cnn, train_loader_orig, epochs=8)
metrics_cnn = eval_torch(cnn, test_loader)
print("CNN baseline:", metrics_cnn)

# Train augmented CNN
cnn_aug = Simple1DCNN(in_ch=X_train.shape[2])
cnn_aug = train_model(cnn_aug, train_loader_aug, epochs=8)
metrics_cnn_aug = eval_torch(cnn_aug, test_loader)
print("CNN augmented:", metrics_cnn_aug)


# === B.1 KS tests per-channel (real minority vs synthetic) ===
from scipy.stats import ks_2samp
# real minority windows from training
real_min = X_train[y_train==0]    # (n_real, T, C)
synth = synthetic                  # (n_synth, T, C) from GAN

ks_results = {}
for ch in range(real_min.shape[2]):
    r = real_min[:,:,ch].ravel()
    s = synth[:,:,ch].ravel()
    stat, p = ks_2samp(r, s)
    ks_results[f"ch_{ch}"] = (stat, p)
ks_sorted = sorted(ks_results.items(), key=lambda x: x[1][0], reverse=True)
print("Top KS (largest divergence):", ks_sorted[:6])


# === B.2 UMAP visualization ===
import umap, matplotlib.pyplot as plt
nR = min(200, real_min.shape[0])
nS = min(200, synth.shape[0])
maj = X_train[y_train==1]
nM = min(200, maj.shape[0])

real_flat = real_min[:nR].reshape(nR, -1)
synth_flat = synth[:nS].reshape(nS, -1)
maj_flat = maj[:nM].reshape(nM, -1)

Z = np.vstack([real_flat, synth_flat, maj_flat])
labels_vis = (['real']*nR) + (['synth']*nS) + (['major']*nM)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
emb = reducer.fit_transform(Z)

plt.figure(figsize=(8,6))
for lbl in ['real','synth','major']:
    idxs = [i for i,l in enumerate(labels_vis) if l==lbl]
    plt.scatter(emb[idxs,0], emb[idxs,1], label=lbl, s=20, alpha=0.7)
plt.legend(); plt.title("UMAP: real minority vs synthetic vs majority"); plt.show()


# === B.3 Classifier score distributions (real vs synthetic) ===
# Using CNN augmented model
import numpy as np
cnn_aug.eval()
with torch.no_grad():
    real_t = torch.tensor(real_min).permute(0,2,1).float().to(device)
    synth_t = torch.tensor(synth).permute(0,2,1).float().to(device)
    pr_real = torch.sigmoid(cnn_aug(real_t)).cpu().numpy()
    pr_synth = torch.sigmoid(cnn_aug(synth_t)).cpu().numpy()

print("Mean prob real:", pr_real.mean(), "synth:", pr_synth.mean())
from scipy.stats import ks_2samp
ks, p = ks_2samp(pr_real.flatten(), pr_synth.flatten())
print("KS on predicted probs:", ks, p)


# === B.4 Integrated Gradients (captum) for CNN explanations ===
from captum.attr import IntegratedGradients
ig = IntegratedGradients(cnn_aug)

# pick examples: one real and one synthetic
idx_real=0; idx_synth=0
inp_real = torch.tensor(real_min[idx_real].T).unsqueeze(0).float().to(device)   # (1, C, T)
inp_synth = torch.tensor(synth[idx_synth].T).unsqueeze(0).float().to(device)

attr_real = ig.attribute(inp_real, target=None, n_steps=50).cpu().numpy().squeeze()  # (C,T)
attr_synth = ig.attribute(inp_synth, target=None, n_steps=50).cpu().numpy().squeeze()

# plot mean abs importance per channel
imp_real = np.mean(np.abs(attr_real), axis=1)
imp_synth = np.mean(np.abs(attr_synth), axis=1)

plt.figure(figsize=(8,4))
plt.bar(np.arange(len(imp_real))-0.15, imp_real, width=0.3, label='real')
plt.bar(np.arange(len(imp_synth))+0.15, imp_synth, width=0.3, label='synth')
plt.xlabel('Channel'); plt.ylabel('Mean |IG|'); plt.legend(); plt.show()


import shap
import xgboost as xgb

# Use flattened windows and labels for XGBoost
model = xgb.XGBClassifier().fit(X_train_flat, y_train)

explainer = shap.TreeExplainer(model)
sample_X = X_train_flat[:50]  # Use a small subset to avoid memory issues
interaction_values = explainer.shap_interaction_values(sample_X)

# Plot interaction between feature 0 and feature 1
shap.plots.scatter(
    interaction_values[:, 0, 1],
    color=sample_X[:, 0]
)