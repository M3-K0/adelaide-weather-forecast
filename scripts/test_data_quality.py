#!/usr/bin/env python3
"""Quick data quality check before training."""

import numpy as np
import torch
import zarr
from pathlib import Path

def check_data_quality():
    """Check for NaN/Inf values in dataset."""
    
    print("🔍 Checking data quality...")
    
    # Check surface data
    surface_path = Path('data/era5/zarr/era5_surface_2010_2020.zarr')
    if surface_path.exists():
        print(f"📁 Loading surface data: {surface_path}")
        surface_ds = zarr.open(surface_path, mode='r')
        
        for var in ['msl', 't2m', 'sp', 'tp']:
            if var in surface_ds:
                data = surface_ds[var]
                print(f"  {var}: shape={data.shape}")
                
                # Sample a chunk to check
                sample = data[0:10, :, :]
                
                nan_count = np.isnan(sample).sum()
                inf_count = np.isinf(sample).sum()
                
                print(f"    NaN: {nan_count}, Inf: {inf_count}")
                print(f"    Range: [{np.nanmin(sample):.3f}, {np.nanmax(sample):.3f}]")
    
    # Check pressure data  
    pressure_path = Path('data/era5/zarr/era5_pressure_2010_2019.zarr')
    if pressure_path.exists():
        print(f"📁 Loading pressure data: {pressure_path}")
        pressure_ds = zarr.open(pressure_path, mode='r')
        
        for level in [1000, 850, 500]:
            for var in ['t', 'q', 'u', 'v']:
                key = f'{var}_{level}'
                if key in pressure_ds:
                    data = pressure_ds[key]
                    print(f"  {key}: shape={data.shape}")
                    
                    # Sample a chunk
                    sample = data[0:10, :, :]
                    
                    nan_count = np.isnan(sample).sum()
                    inf_count = np.isinf(sample).sum()
                    
                    print(f"    NaN: {nan_count}, Inf: {inf_count}")
                    print(f"    Range: [{np.nanmin(sample):.3f}, {np.nanmax(sample):.3f}]")

def test_gpu():
    """Test GPU functionality."""
    print("\n🚀 Testing GPU...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test tensor operations
        print("Testing tensor operations...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Warm up
        for _ in range(10):
            z = torch.matmul(x, y)
        
        print("✓ GPU operations working")
        
        # Test memory
        try:
            big_tensor = torch.randn(5000, 5000, device=device)
            print("✓ GPU memory allocation working")
            del big_tensor
        except Exception as e:
            print(f"⚠️ GPU memory issue: {e}")
        
    else:
        print("❌ CUDA not available")

if __name__ == "__main__":
    check_data_quality()
    test_gpu()
    print("\n✅ Data quality check complete!")