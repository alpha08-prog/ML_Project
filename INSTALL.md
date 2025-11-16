# Installation Guide

## Python Version Recommendation

**IMPORTANT**: This project works best with **Python 3.11** or **Python 3.12**. 

Python 3.13 is very new and many packages (especially numpy, scipy) don't have pre-built wheels yet, which can cause compilation errors.

## Installation Steps

### Option 1: Using Python 3.11 or 3.12 (Recommended)

1. Install Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/)
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```
3. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```
4. Install dependencies:
   ```bash
   # IMPORTANT: Always use 'python -m pip' instead of just 'pip' to ensure
   # you're using the correct Python interpreter
   python -m pip install -r requirements.txt
   ```

### Option 1b: If you have multiple Python installations (Anaconda + system Python)

**IMPORTANT**: If you have multiple Python installations, always use `python -m pip` instead of just `pip` to ensure you're installing to the correct Python environment.

```bash
# Check which Python you're using
python --version

# Use python -m pip to install (ensures correct Python interpreter)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# If you encounter llvmlite conflicts (common with Anaconda):
python -m pip install -r requirements.txt --ignore-installed llvmlite
```

### Option 2: If Using Python 3.13

If you must use Python 3.13, try these workarounds:

1. **Install numpy first with pre-built wheels**:
   ```bash
   pip install numpy --only-binary :all:
   pip install -r requirements.txt
   ```

2. **Or install build tools** (requires Visual Studio Build Tools on Windows):
   ```bash
   # Install Microsoft C++ Build Tools from:
   # https://visualstudio.microsoft.com/visual-cpp-build-tools/
   # Then try installing again
   pip install -r requirements.txt
   ```

3. **Use conda instead** (often has better Python 3.13 support):
   ```bash
   conda create -n ml_project python=3.13
   conda activate ml_project
   conda install numpy pandas scipy scikit-learn matplotlib seaborn
   pip install -r requirements.txt
   ```

### PyTorch Installation

PyTorch installation depends on your system:

**For CPU-only:**
```bash
pip install torch torchaudio torchvision
```

**For GPU support (CUDA 11.8):**
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For GPU support (CUDA 12.1):**
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific setup.

## Troubleshooting

### Error: "Python dependency not found" or architecture mismatch

- **Solution**: 
  - Use Python 3.11 or 3.12 instead of 3.13
  - Always use `python -m pip` instead of just `pip` to ensure correct Python interpreter
  - Or install Microsoft Visual C++ Build Tools on Windows

### Error: "Cannot uninstall llvmlite" or "distutils installed project"

- **Solution**: This is common with Anaconda. Use:
  ```bash
  python -m pip install -r requirements.txt --ignore-installed llvmlite
  ```

### Error: "Failed building wheel for numpy"

- **Solution**: Install numpy separately first:
  ```bash
  python -m pip install numpy --only-binary :all:
  python -m pip install -r requirements.txt
  ```

### Error: "Failed building wheel for pyedflib"

- **Solution**: pyedflib requires pre-built wheels on Windows. Try:
  ```bash
  # Option 1: Install from pre-built wheel (recommended)
  python -m pip install --only-binary :all: pyedflib
  
  # Option 2: If that fails, try upgrading pip first
  python -m pip install --upgrade pip
  python -m pip install pyedflib
  
  # Option 3: Install Visual C++ Build Tools if compilation is needed
  # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  ```

### Error: "No module named 'pyedflib'" or other import errors

- **Solution**: Make sure you're using the correct Python and all dependencies are installed:
  ```bash
  python --version  # Verify Python version
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

### Multiple Python installations detected

- **Problem**: You have both Anaconda Python and system Python installed
- **Solution**: Always use `python -m pip` instead of `pip` to ensure you're installing to the correct Python environment
- Check which Python is active: `python --version` and `python -m pip --version`

