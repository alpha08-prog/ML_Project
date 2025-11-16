# Archive

This directory contains archived files that are kept for reference but are no longer actively used.

## work.py

The original `work.py` file contains the Jupyter notebook code that was converted to Python script format. This has been refactored into:

- `backend/train_models.py` - Clean, production-ready training script

The original `work.py` is kept here for reference but is not needed for the project to function.

## Migration Notes

If you need to reference the original notebook code:
- See `work.py` for the original implementation
- See `backend/train_models.py` for the refactored version

The refactored version includes:
- Better error handling
- Path resolution fixes
- PyTorch graceful degradation
- Model saving functionality

