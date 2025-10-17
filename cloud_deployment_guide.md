# Cloud GPU Deployment Guide

## Quick Start

### 1. Data Upload Strategy
Your Zarr datasets are large (~5GB total). Options:
- **Upload to cloud storage**: AWS S3, Google Cloud Storage 
- **Direct transfer**: rsync/scp to cloud instance
- **Compress first**: `tar -czf data.tar.gz data/era5/zarr/`

### 2. Instance Requirements
- **GPU**: V100, A100, or RTX 4090/3090
- **RAM**: 32GB+ (for data loading)
- **Storage**: 50GB+ for data + models
- **Providers**: Vast.ai, RunPod, AWS p3/g4, Google Cloud

### 3. Setup Commands
```bash
# On cloud instance:
git clone <your-repo> weather-forecast
cd weather-forecast
bash scripts/cloud_setup.sh

# Upload data (from local):
rsync -avz --progress data/ instance-ip:~/weather-forecast/data/
```

### 4. Training
```bash
conda activate weather
python scripts/train_embeddings.py
```

## Performance Expectations
- **A100**: ~2 hours for 100 epochs
- **V100**: ~3-4 hours for 100 epochs  
- **RTX 4090**: ~4-5 hours for 100 epochs

## Configuration
Training optimized for cloud:
- Batch size: 64 (adjust based on GPU memory)
- Mixed precision: Enabled
- Workers: 4 (adjust based on CPU cores)
- Checkpoints: Every 10 epochs

## Monitoring
Progress logged to:
- Console with tqdm progress bars
- `outputs/training_*/training.log`
- `outputs/training_*/metrics.json`

## After Training
Download models:
```bash
# From cloud instance
rsync -avz --progress outputs/ local-machine:~/weather-forecast/outputs/
```