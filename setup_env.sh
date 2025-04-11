#!/bin/bash

echo "[*] Creating Conda environment: lmfed_venv"
conda create -n lmfed_venv python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lmfed_venv

echo "[*] Installing PyTorch with CUDA 12 support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[*] Installing Transformers & other dependencies..."
pip install transformers datasets scikit-learn tqdm tensorboard

echo "[*] Setup complete ✅"
echo "👉 To start: conda activate lmfed_venv && python main.py"
