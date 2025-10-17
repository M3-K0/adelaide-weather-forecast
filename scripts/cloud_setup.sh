#!/bin/bash
# Cloud GPU Setup Script for Weather Forecasting Training

set -e

echo "Setting up Adelaide Weather Forecasting on Cloud GPU..."

# Update system
sudo apt-get update
sudo apt-get install -y wget bzip2 ca-certificates curl

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash
source ~/.bashrc

# Create environment
conda create -n weather python=3.10 -y
conda activate weather

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other packages
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

echo "Environment setup complete!"
echo "Next steps:"
echo "1. Upload your data to the cloud instance"
echo "2. Run: conda activate weather"
echo "3. Run: python scripts/train_embeddings.py"