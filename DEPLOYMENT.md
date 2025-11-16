# Deployment Guide

## Pre-Deployment Checklist

Before pushing to GitHub, ensure:

- ✅ Models are trained locally (not committed)
- ✅ Large data files are excluded (.gitignore configured)
- ✅ No sensitive data in code
- ✅ Documentation is up to date

## What Gets Pushed to GitHub

### ✅ Included (Small files)
- Source code (`.py`, `.ts`, `.tsx`)
- Configuration files
- Documentation (`.md`)
- `subject-info.csv` (metadata only)
- Project structure

### ❌ Excluded (Large files)
- `models/*` - All trained models (users train locally)
- `Data/eegmat/*.edf` - EEG data files
- `node_modules/` - Frontend dependencies
- `__pycache__/` - Python cache
- Build artifacts

## Repository Size

Expected repository size: **~5-10 MB** (without large files)

If you include:
- Models: +50-100 MB
- EEG data: +500 MB - 2 GB
- node_modules: +200-300 MB

**Total with everything: 750 MB - 2.5 GB** (too large for GitHub free tier)

## Setup Instructions for New Users

1. **Clone repository:**
   ```bash
   git clone <repo-url>
   cd ML_Project
   ```

2. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   cd frontend/ml && npm install
   ```

3. **Add EEG data:**
   - Place `.edf` files in `Data/eegmat/`
   - Ensure `subject-info.csv` is present

4. **Train models:**
   ```bash
   python backend/train_models.py
   ```

5. **Start application:**
   ```bash
   # Terminal 1: Backend
   python backend/api.py
   
   # Terminal 2: Frontend
   cd frontend/ml && npm run dev
   ```

## GitHub Repository Setup

### Recommended Repository Structure

```
ML_Project/
├── .gitignore          ✅ Commit
├── .gitattributes      ✅ Commit
├── README.md           ✅ Commit
├── requirements.txt    ✅ Commit
├── backend/            ✅ Commit (code only)
├── frontend/ml/        ✅ Commit (code only, no node_modules)
├── archive/            ✅ Commit (optional)
└── Data/eegmat/        ⚠️  Commit metadata only (.csv, .txt)
```

### Files to Add Manually

Users need to add:
- EEG `.edf` files to `Data/eegmat/`
- Train models (creates `models/` directory)

## Large File Alternatives

If you need to share large files:

1. **Git LFS (Git Large File Storage):**
   ```bash
   git lfs install
   git lfs track "*.edf"
   git lfs track "models/*.pt"
   ```

2. **External Storage:**
   - Google Drive
   - Dropbox
   - AWS S3
   - Zenodo (for research data)

3. **Data Download Script:**
   Create a script that downloads data from external source

## Verification

Check what will be committed:
```bash
git status
git ls-files
```

Verify large files are ignored:
```bash
git check-ignore -v models/*.pt
git check-ignore -v Data/eegmat/*.edf
```

