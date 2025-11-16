#!/bin/bash
echo "Starting FastAPI Backend..."
echo ""
echo "Make sure you've trained the models first with: python backend/train_models.py"
echo ""
cd backend
python api.py

