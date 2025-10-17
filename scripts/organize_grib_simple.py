#!/usr/bin/env python3
"""
Simple GRIB to Zarr Converter for ERA5 Data
Just processes the working GRIB files and creates unified datasets.
"""

import os
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def load_surface_grib(grib_path, description):
    """Load surface GRIB file with main variables only."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {description}: {grib_path}")
    
    try:
        # Load just the main surface variables (skip precipitation for now)
        ds = xr.open_dataset(grib_path, engine='cfgrib', 
                            filter_by_keys={'shortName': ['u10', 'v10', 't2m', 'msl']})
        
        logger.info(f"  Variables: {list(ds.data_vars.keys())}")
        logger.info(f"  Time: {ds.time.min().values} to {ds.time.max().values}")
        logger.info(f"  Shape: {dict(ds.sizes)}")
        
        return ds
    except Exception as e:
        logger.error(f"Failed to load {grib_path}: {e}")
        return None


def load_pressure_grib(grib_path, description):
    """Load pressure level GRIB file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {description}: {grib_path}")
    
    try:
        ds = xr.open_dataset(grib_path, engine='cfgrib')
        
        logger.info(f"  Variables: {list(ds.data_vars.keys())}")
        logger.info(f"  Levels: {ds.isobaricInhPa.values}")
        logger.info(f"  Time: {ds.time.min().values} to {ds.time.max().values}")
        logger.info(f"  Shape: {dict(ds.sizes)}")
        
        return ds
    except Exception as e:
        logger.error(f"Failed to load {grib_path}: {e}")
        return None


def main():
    logger = setup_logging()
    logger.info("Starting simple GRIB organization")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Create output directory
    zarr_dir = Path("data/era5/zarr")
    zarr_dir.mkdir(parents=True, exist_ok=True)
    
    # Process surface data
    logger.info("\n=== SURFACE DATA ===")
    surface_datasets = []
    
    # Load 2010-2019 surface data
    surface_2010_2019 = load_surface_grib(
        'data/e88d1a9e3f212150503ffccf17bdf35a single levels 2010 to 2019/data.grib',
        'Surface 2010-2019'
    )
    if surface_2010_2019:
        surface_datasets.append(surface_2010_2019)
    
    # Load December 2020 surface data
    surface_dec_2020 = load_surface_grib(
        'data/194b9ab1760177368a142890bde3da14 single level december 2020/data.grib',
        'Surface December 2020'
    )
    if surface_dec_2020:
        surface_datasets.append(surface_dec_2020)
    
    # Combine surface data
    if surface_datasets:
        logger.info("Combining surface datasets...")
        surface_combined = xr.concat(surface_datasets, dim='time')
        surface_combined = surface_combined.sortby('time')
        
        # Remove duplicates
        _, unique_indices = np.unique(surface_combined.time.values, return_index=True)
        surface_combined = surface_combined.isel(time=sorted(unique_indices))
        
        logger.info(f"Combined surface: {dict(surface_combined.sizes)}")
        logger.info(f"Variables: {list(surface_combined.data_vars.keys())}")
        
        # Save to Zarr
        output_path = zarr_dir / "era5_surface_2010_2020.zarr"
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        
        logger.info(f"Saving to {output_path}")
        surface_combined.to_zarr(output_path)
        logger.info("✓ Surface data saved")
    
    # Process pressure data
    logger.info("\n=== PRESSURE DATA ===")
    pressure_datasets = []
    
    # Load all pressure level GRIB files
    pressure_files = [
        ('data/7cb3b111cbee78570a4e308bd98653c4 pressure leveles 2010 2011/data.grib', 'Pressure 2010-2011'),
        ('data/e18e13d0e22d7c50169e0e788387a3d7 pressure levels 2012 13 14 15/data.grib', 'Pressure 2012-2015'),
        ('data/202192f8adb5c8f12260cd01f9a763ba pressure levels 2016 17 18 19/data.grib', 'Pressure 2016-2019')
    ]
    
    for grib_path, description in pressure_files:
        if Path(grib_path).exists():
            ds = load_pressure_grib(grib_path, description)
            if ds:
                pressure_datasets.append(ds)
    
    # Combine pressure data
    if pressure_datasets:
        logger.info("Combining pressure datasets...")
        pressure_combined = xr.concat(pressure_datasets, dim='time')
        pressure_combined = pressure_combined.sortby('time')
        
        # Remove duplicates
        _, unique_indices = np.unique(pressure_combined.time.values, return_index=True)
        pressure_combined = pressure_combined.isel(time=sorted(unique_indices))
        
        logger.info(f"Combined pressure: {dict(pressure_combined.sizes)}")
        logger.info(f"Variables: {list(pressure_combined.data_vars.keys())}")
        logger.info(f"Levels: {pressure_combined.isobaricInhPa.values}")
        
        # Save to Zarr
        output_path = zarr_dir / "era5_pressure_2010_2019.zarr"
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        
        logger.info(f"Saving to {output_path}")
        pressure_combined.to_zarr(output_path)
        logger.info("✓ Pressure data saved")
    
    logger.info("\n=== SUMMARY ===")
    logger.info("GRIB organization completed!")
    logger.info("Note: 2020 pressure data from NetCDF files needs separate processing")


if __name__ == "__main__":
    main()