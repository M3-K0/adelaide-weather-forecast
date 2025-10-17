#!/usr/bin/env python3
"""
Environment verification script for Adelaide Weather Forecasting System
Run this after all packages are installed to verify the setup.
"""

def test_imports():
    """Test if all required packages can be imported."""
    
    print("Testing package imports...")
    
    # Core scientific packages
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        
    # Data handling
    try:
        import xarray as xr
        print(f"✓ Xarray {xr.__version__}")
    except ImportError as e:
        print(f"✗ Xarray: {e}")
        
    try:
        import dask
        print(f"✓ Dask {dask.__version__}")
    except ImportError as e:
        print(f"✗ Dask: {e}")
        
    try:
        import zarr
        print(f"✓ Zarr {zarr.__version__}")
    except ImportError as e:
        print(f"✗ Zarr: {e}")
        
    try:
        import netCDF4
        print(f"✓ NetCDF4 {netCDF4.__version__}")
    except ImportError as e:
        print(f"✗ NetCDF4: {e}")
        
    try:
        import h5netcdf
        print(f"✓ h5netcdf {h5netcdf.__version__}")
    except ImportError as e:
        print(f"✗ h5netcdf: {e}")
        
    # Machine learning
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        # Test CUDA
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  - CUDA available: {torch.cuda.is_available()}")
            print(f"  - GPU: {device_name}")
            print(f"  - CUDA version: {torch.version.cuda}")
        else:
            print("  - CUDA not available")
            
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        
    try:
        import faiss
        print(f"✓ FAISS {faiss.__version__}")
        
        # Test FAISS GPU
        try:
            ngpus = faiss.get_num_gpus()
            print(f"  - FAISS GPUs available: {ngpus}")
        except:
            print("  - FAISS GPU not available")
            
    except ImportError as e:
        print(f"✗ FAISS: {e}")
        
    # Data access
    try:
        import cdsapi
        print("✓ CDS API")
    except ImportError as e:
        print(f"✗ CDS API: {e}")


def test_cuda_tensor_operations():
    """Test basic CUDA tensor operations."""
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("✗ CUDA not available for tensor operations")
            return
            
        print("\nTesting CUDA tensor operations...")
        
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Matrix multiplication
        z = torch.mm(x, y)
        
        # Move back to CPU
        z_cpu = z.cpu()
        
        print("✓ CUDA tensor operations successful")
        print(f"  - Created {x.shape} tensors on GPU")
        print(f"  - Matrix multiplication: {z.shape}")
        print(f"  - GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        
        # Clean up
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ CUDA tensor operations failed: {e}")


def test_faiss_gpu():
    """Test FAISS GPU functionality."""
    
    try:
        import faiss
        import numpy as np
        
        if faiss.get_num_gpus() == 0:
            print("✗ No FAISS GPUs available")
            return
            
        print("\nTesting FAISS GPU operations...")
        
        # Create test data
        d = 256  # embedding dimension
        n = 1000  # number of vectors
        
        # Generate random embeddings
        embeddings = np.random.randn(n, d).astype(np.float32)
        
        # Create GPU index
        res = faiss.StandardGpuResources()
        index_cpu = faiss.IndexFlatIP(d)  # Inner product index
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        
        # Add embeddings to index
        index_gpu.add(embeddings)
        
        # Search for similar vectors
        k = 5  # number of nearest neighbors
        query = embeddings[:10]  # use first 10 as queries
        
        distances, indices = index_gpu.search(query, k)
        
        print("✓ FAISS GPU operations successful")
        print(f"  - Index dimension: {d}")
        print(f"  - Vectors in index: {index_gpu.ntotal}")
        print(f"  - Search k={k} neighbors")
        print(f"  - Top distances shape: {distances.shape}")
        
    except Exception as e:
        print(f"✗ FAISS GPU operations failed: {e}")


def test_data_loading():
    """Test basic data loading capabilities."""
    
    try:
        import xarray as xr
        import numpy as np
        
        print("\nTesting data handling...")
        
        # Create synthetic weather data
        lat = np.linspace(-37.4, -32.4, 21)  # Adelaide region
        lon = np.linspace(136.1, 141.1, 21)
        time = np.arange('2020-01-01', '2020-01-10', dtype='datetime64[h]')
        
        # Create test dataset
        data = xr.Dataset({
            'temperature': (('time', 'lat', 'lon'), 
                           np.random.randn(len(time), len(lat), len(lon)) + 20),
            'pressure': (('time', 'lat', 'lon'),
                        np.random.randn(len(time), len(lat), len(lon)) + 1013.25)
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })
        
        # Test basic operations
        temp_mean = data.temperature.mean()
        subset = data.sel(lat=slice(-35, -34), lon=slice(138, 139))
        
        print("✓ Xarray data handling successful")
        print(f"  - Dataset shape: {data.dims}")
        print(f"  - Temperature mean: {temp_mean.values:.2f}")
        print(f"  - Subset shape: {subset.dims}")
        
    except Exception as e:
        print(f"✗ Data handling failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Adelaide Weather Forecasting - Environment Verification")
    print("=" * 60)
    
    test_imports()
    test_cuda_tensor_operations()
    test_faiss_gpu()
    test_data_loading()
    
    print("\n" + "=" * 60)
    print("Environment verification complete!")
    print("=" * 60)