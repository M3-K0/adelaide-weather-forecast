# Adelaide Weather Forecasting System

An embedding-based analog retrieval system for 72-hour weather forecasting in Adelaide, Australia, using CNN encoders and FAISS for vector similarity search.

## 🎯 Project Goals

- **Target**: 72-hour forecasts beating persistence baseline by 20%+ RMSE
- **Region**: Adelaide area (25°×25° around -34.9°S, 138.6°E)
- **Architecture**: CNN encoder → 256D embeddings → FAISS retrieval → analog ensemble
- **Data**: ERA5 reanalysis (2010-2020) + GFS real-time

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ ERA5/GFS Data   │ -> │ CNN Encoder      │ -> │ 256D Embeddings │
│ (25°×25° grid)  │    │ + FiLM Condition │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Weather Forecast│ <- │ Analog Ensemble  │ <- │ FAISS Index     │
│ (6,12,24,48,72h)│    │ (30 best analogs)│    │ (similarity)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
weather-forecast/
├── data/
│   ├── era5/           # ERA5 reanalysis downloads
│   ├── gfs/            # GFS real-time data  
│   └── processed/      # Zarr datasets
├── models/
│   ├── embeddings/     # Trained CNN encoders
│   ├── baselines/      # Simple analog models
│   └── checkpoints/    # Training checkpoints
├── configs/
│   ├── data.yaml       # Data configuration
│   ├── model.yaml      # Model hyperparameters
│   └── training.yaml   # Training settings
├── scripts/
│   ├── download_era5.py    # ERA5 data downloader
│   ├── train_embeddings.py # Model training
│   ├── run_forecast.py     # Operational forecasting
│   └── evaluate.py         # Model evaluation
├── src/
│   ├── data/           # Data pipeline modules
│   ├── models/         # Model architectures
│   ├── training/       # Training loops
│   └── evaluation/     # Metrics and validation
└── notebooks/          # Analysis and visualization
```

## 🚀 Quick Start

### Phase 0: Environment Setup ✅

1. **Install miniforge3** (completed)
2. **Create conda environment** (completed)
3. **Install packages** (in progress)
4. **Verify setup**

```bash
cd ~/weather-forecast
python test_environment.py
```

### Phase 1: Data Pipeline

1. **Setup CDS API credentials**
```bash
# Create ~/.cdsapirc with your ECMWF credentials
echo "url: https://cds.climate.copernicus.eu/api/v2" > ~/.cdsapirc
echo "key: {uid}:{api-key}" >> ~/.cdsapirc
```

2. **Download ERA5 data**
```bash
cd scripts
python download_era5.py
```

3. **Implement GFS fetcher** (for real-time forecasts)

### Phase 2: Baseline System

1. **Simple analog ensemble**: L2 distance on normalized fields
2. **Find 50 best analogs** per forecast lead (6, 12, 24, 48, 72h)
3. **Rank weighting**: w_i = 1/i^0.5
4. **Evaluate vs persistence**

### Phase 3: Learned Embeddings

1. **CNN encoder**: 4 conv stages (32→64→128→256), ASPP module
2. **FiLM conditioning**: lead time + seasonal indicators
3. **InfoNCE contrastive loss** (temp=0.07)
4. **FAISS indexing**: 256D embeddings, retrieve 30 analogs

### Phase 4: Hybrid System

1. **Flow-dependent bias correction** on GFS
2. **Blend analog ensemble** with corrected GFS
3. **Uncertainty quantification** from analog spread

## ⚙️ System Requirements

- **Hardware**: RTX 3060 (12GB), 64GB RAM, Gen4 NVMe
- **OS**: WSL2 Ubuntu
- **Python**: 3.11+ with conda/miniforge3
- **CUDA**: 12.1+ for GPU acceleration

## 📦 Dependencies

**Core ML/Data:**
- PyTorch (CUDA 12.1)
- FAISS-GPU 
- Xarray, Dask, Zarr
- NetCDF4, h5netcdf

**Weather Data:**
- CDS API (ERA5)
- OpenDAP (GFS)

**Utilities:**
- NumPy, Pandas
- PyYAML, logging

## 🎯 Target Performance

- **Baseline**: Persistence forecast
- **Goal**: 20%+ RMSE improvement over persistence
- **Variables**: 2m temperature, precipitation
- **Leads**: 6, 12, 24, 48, 72 hours
- **Domain**: Adelaide metropolitan area

## 📊 Evaluation Metrics

- **Primary**: RMSE, MAE, Bias
- **Secondary**: Correlation, NSE, KGE
- **Probabilistic**: CRPS, Reliability, Resolution

## 🔄 Workflow

1. **Training**: Learn embeddings on 2010-2018 ERA5 data
2. **Validation**: Tune hyperparameters on 2019 data
3. **Testing**: Final evaluation on 2020 data
4. **Operational**: Real-time forecasts with GFS + analog retrieval

## 📈 Progress Tracking

- [x] Phase 0: Environment setup
- [x] Project structure creation
- [x] Configuration files
- [ ] Phase 1: Data pipeline
- [ ] Phase 2: Baseline system
- [ ] Phase 3: CNN embeddings
- [ ] Phase 4: Hybrid forecasting

## 🔍 Key Features

- **Multi-scale CNN**: Captures both local and synoptic patterns
- **Contrastive learning**: InfoNCE loss for similarity learning
- **FiLM conditioning**: Lead time and seasonal awareness
- **GPU acceleration**: FAISS + PyTorch for fast retrieval
- **Zarr storage**: Efficient chunked data access
- **Modular design**: Easy to extend and modify

## 📚 References

- ERA5 reanalysis: ECMWF
- FAISS: Facebook AI Similarity Search
- InfoNCE: Contrastive learning framework
- FiLM: Feature-wise Linear Modulation

---

**Status**: CNN Training Optimized - Ready for H200 Cloud Deployment

## 🚀 Cloud GPU Deployment

### Quick Deploy to H200/A100:
```bash
git clone https://github.com/M3-K0/adelaide-weather-forecast.git
cd adelaide-weather-forecast
bash scripts/cloud_setup.sh
conda activate weather
python scripts/test_data_quality.py  # Validate data
python scripts/train_embeddings.py  # Train model
```

### Performance Expectations:
- **H200 80GB**: ~45-60 minutes, $2-3 total cost
- **A100 80GB**: ~1.5-2 hours, $2-4 total cost
- **RTX 4090**: ~4-5 hours, $1-2 total cost

### Fixed Issues:
- ✅ NaN loss prevention
- ✅ Gradient clipping 
- ✅ GPU utilization optimization
- ✅ Data validation pipeline