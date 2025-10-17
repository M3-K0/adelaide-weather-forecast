#!/usr/bin/env python3
"""
Organize and Convert ERA5 GRIB Data to Unified Zarr Datasets
Combines all GRIB files into properly structured datasets for training.
"""

import os
import yaml
import logging
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import glob
from tqdm import tqdm


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('organize_grib_data.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_grib_dataset(grib_path, description=""):
    """Load a GRIB file and return xarray dataset with metadata."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {description}: {grib_path}")
    
    try:
        # Load main variables (excluding precipitation to avoid time conflicts)
        ds = xr.open_dataset(grib_path, engine='cfgrib', 
                            filter_by_keys={'shortName': ['u10', 'v10', 't2m', 'msl', 'z', 'q', 't', 'u', 'v']})
        
        logger.info(f"  Main variables: {list(ds.data_vars.keys())}")
        
        # Try to load precipitation separately if this is surface data
        try:
            ds_tp = xr.open_dataset(grib_path, engine='cfgrib', 
                                   filter_by_keys={'shortName': 'tp'})
            
            if 'tp' in ds_tp.data_vars:
                logger.info("  Loading precipitation separately...")
                
                # Interpolate precipitation to main time grid
                ds_tp_interp = ds_tp.interp(time=ds.time, method='linear', 
                                          kwargs={'fill_value': 'extrapolate'})
                
                # Merge with main dataset
                ds = xr.merge([ds, ds_tp_interp])
                logger.info("  Added precipitation with time interpolation")
                
        except Exception as e:
            logger.info(f"  No precipitation data or failed to load: {e}")
        
        logger.info(f"  Final variables: {list(ds.data_vars.keys())}")
        logger.info(f"  Time range: {ds.time.min().values} to {ds.time.max().values}")
        logger.info(f"  Shape: {dict(ds.sizes)}")
        
        return ds
        
    except Exception as e:
        logger.error(f"Failed to load {grib_path}: {e}")
        return None


def process_surface_data():
    """Process and combine all surface level data."""
    
    logger = logging.getLogger(__name__)
    logger.info("Processing surface level data...")
    
    # Define GRIB files for surface data
    grib_files = [
        {
            'path': 'data/e88d1a9e3f212150503ffccf17bdf35a single levels 2010 to 2019/data.grib',
            'description': 'Surface levels 2010-2019',
            'years': (2010, 2019)
        },
        {
            'path': 'data/194b9ab1760177368a142890bde3da14 single level december 2020/data.grib', 
            'description': 'Surface December 2020',
            'years': (2020, 2020)
        }
    ]
    
    # Add existing NetCDF files from test script
    netcdf_files = glob.glob('data/era5/surface/era5_surface_2020_*.nc')
    
    datasets = []
    
    # Load GRIB datasets
    for grib_info in grib_files:
        grib_path = Path(grib_info['path'])
        if grib_path.exists():
            ds = load_grib_dataset(grib_path, grib_info['description'])
            if ds is not None:
                # Add metadata
                ds.attrs.update({
                    'source_file': str(grib_path),
                    'description': grib_info['description'],
                    'years': f"{grib_info['years'][0]}-{grib_info['years'][1]}"
                })
                datasets.append(ds)
        else:
            logger.warning(f"GRIB file not found: {grib_path}")
    
    # Load NetCDF files from test script (Jan-Nov 2020)
    for nc_file in sorted(netcdf_files):
        nc_path = Path(nc_file)
        if nc_path.exists():
            logger.info(f"Loading NetCDF: {nc_path.name}")
            try:
                ds = xr.open_dataset(nc_path, engine='h5netcdf')
                
                # Standardize variable names to match GRIB
                var_mapping = {
                    '10m_u_component_of_wind': 'u10',
                    '10m_v_component_of_wind': 'v10', 
                    '2m_temperature': 't2m',
                    'mean_sea_level_pressure': 'msl',
                    'total_precipitation': 'tp'
                }
                
                for old_name, new_name in var_mapping.items():
                    if old_name in ds.data_vars:
                        ds = ds.rename({old_name: new_name})
                
                ds.attrs.update({
                    'source_file': str(nc_path),
                    'description': f'Surface NetCDF {nc_path.stem}',
                })
                datasets.append(ds)
                logger.info(f"  Variables: {list(ds.data_vars.keys())}")
                
            except Exception as e:
                logger.warning(f"Failed to load {nc_path}: {e}")
    
    if not datasets:
        logger.error("No surface datasets found!")
        return None
    
    # Combine all datasets
    logger.info("Combining surface datasets...")
    combined = xr.concat(datasets, dim='time', combine_attrs='override')
    
    # Sort by time and remove duplicates
    combined = combined.sortby('time')
    _, unique_indices = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=sorted(unique_indices))
    
    # Add final metadata
    combined.attrs.update({
        'title': 'ERA5 Surface Variables for Adelaide Region',
        'description': 'Combined surface level data from multiple sources',
        'variables': 'u10, v10, t2m, msl, tp',
        'region': 'Adelaide, Australia (32-38S, 136-142E)',
        'processing_date': datetime.now().isoformat(),
        'temporal_resolution': '6-hourly',
        'spatial_resolution': '0.25 degrees'
    })
    
    logger.info(f"Combined surface dataset: {dict(combined.sizes)}")
    logger.info(f"Time range: {combined.time.min().values} to {combined.time.max().values}")
    logger.info(f"Variables: {list(combined.data_vars.keys())}")
    
    return combined


def process_pressure_data():
    """Process and combine all pressure level data."""
    
    logger = logging.getLogger(__name__)
    logger.info("Processing pressure level data...")
    
    # Define GRIB files for pressure data
    grib_files = [
        {
            'path': 'data/7cb3b111cbee78570a4e308bd98653c4 pressure leveles 2010 2011/data.grib',
            'description': 'Pressure levels 2010-2011',
            'years': (2010, 2011)
        },
        {
            'path': 'data/e18e13d0e22d7c50169e0e788387a3d7 pressure levels 2012 13 14 15/data.grib',
            'description': 'Pressure levels 2012-2015', 
            'years': (2012, 2015)
        },
        {
            'path': 'data/202192f8adb5c8f12260cd01f9a763ba pressure levels 2016 17 18 19/data.grib',
            'description': 'Pressure levels 2016-2019',
            'years': (2016, 2019)
        }
    ]
    
    # Add existing NetCDF files from test script (2020)
    netcdf_files = glob.glob('data/era5/pressure_levels/era5_pressure_2020_*.nc')
    
    datasets = []
    
    # Load GRIB datasets
    for grib_info in grib_files:
        grib_path = Path(grib_info['path'])
        if grib_path.exists():
            ds = load_grib_dataset(grib_path, grib_info['description'])
            if ds is not None:
                # Add metadata
                ds.attrs.update({
                    'source_file': str(grib_path),
                    'description': grib_info['description'],
                    'years': f"{grib_info['years'][0]}-{grib_info['years'][1]}"
                })
                datasets.append(ds)
        else:
            logger.warning(f"GRIB file not found: {grib_path}")
    
    # Load NetCDF files from test script (2020)
    for nc_file in sorted(netcdf_files):
        nc_path = Path(nc_file)
        if nc_path.exists():
            logger.info(f"Loading NetCDF: {nc_path.name}")
            try:
                ds = xr.open_dataset(nc_path, engine='h5netcdf')
                
                # Standardize variable names to match GRIB
                var_mapping = {
                    'geopotential': 'z',
                    'temperature': 't',
                    'specific_humidity': 'q', 
                    'u_component_of_wind': 'u',
                    'v_component_of_wind': 'v'
                }
                
                for old_name, new_name in var_mapping.items():
                    if old_name in ds.data_vars:
                        ds = ds.rename({old_name: new_name})
                
                # Rename pressure coordinate to match GRIB
                if 'level' in ds.coords:
                    ds = ds.rename({'level': 'isobaricInhPa'})
                
                ds.attrs.update({
                    'source_file': str(nc_path),
                    'description': f'Pressure NetCDF {nc_path.stem}',
                })
                datasets.append(ds)
                logger.info(f"  Variables: {list(ds.data_vars.keys())}")
                
            except Exception as e:
                logger.warning(f"Failed to load {nc_path}: {e}")
    
    if not datasets:
        logger.error("No pressure datasets found!")
        return None
    
    # Combine all datasets
    logger.info("Combining pressure datasets...")
    combined = xr.concat(datasets, dim='time', combine_attrs='override')
    
    # Sort by time and remove duplicates
    combined = combined.sortby('time')
    _, unique_indices = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=sorted(unique_indices))
    
    # Add final metadata
    combined.attrs.update({
        'title': 'ERA5 Pressure Level Variables for Adelaide Region',
        'description': 'Combined pressure level data from multiple sources',
        'variables': 'z, t, q, u, v',
        'levels': '500 hPa, 850 hPa',
        'region': 'Adelaide, Australia (32-38S, 136-142E)',
        'processing_date': datetime.now().isoformat(),
        'temporal_resolution': '6-hourly',
        'spatial_resolution': '0.25 degrees'
    })
    
    logger.info(f"Combined pressure dataset: {dict(combined.sizes)}")
    logger.info(f"Time range: {combined.time.min().values} to {combined.time.max().values}")
    logger.info(f"Variables: {list(combined.data_vars.keys())}")
    logger.info(f"Pressure levels: {combined.isobaricInhPa.values}")
    
    return combined


def save_to_zarr(dataset, output_path, dataset_type):
    """Save dataset to Zarr format with optimal chunking."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saving {dataset_type} to Zarr: {output_path}")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing if present
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
        logger.info(f"Removed existing {output_path}")
    
    # Set up optimal chunking
    encoding = {}
    
    # Time chunks: ~1 year of 6-hourly data (365*4 = 1460)
    time_chunk = min(1460, len(dataset.time))
    
    for var in dataset.data_vars:
        chunks = {'time': time_chunk}
        
        # Spatial chunks
        if 'latitude' in dataset[var].dims:
            chunks['latitude'] = min(21, len(dataset.latitude))
        if 'longitude' in dataset[var].dims:
            chunks['longitude'] = min(21, len(dataset.longitude))
            
        # Pressure level chunks (keep together)
        if 'isobaricInhPa' in dataset[var].dims:
            chunks['isobaricInhPa'] = len(dataset.isobaricInhPa)
        
        # Convert chunk dict to tuple of ints
        chunk_tuple = tuple(chunks[dim] for dim in dataset[var].dims)
        
        encoding[var] = {
            'chunks': chunk_tuple
        }
        
        logger.info(f"  {var}: chunks = {chunks}")
    
    # Save to Zarr
    logger.info("Writing to disk...")
    dataset.to_zarr(output_path, encoding=encoding)
    
    # Verify the saved dataset
    logger.info("Verifying saved dataset...")
    test_ds = xr.open_zarr(output_path)
    logger.info(f"✓ Saved successfully: {dict(test_ds.sizes)}")
    test_ds.close()
    
    return output_path


def main():
    """Main organization function."""
    
    logger = setup_logging()
    logger.info("Starting ERA5 GRIB data organization")
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    logger.info(f"Working directory: {project_dir}")
    
    # Create output directories
    zarr_dir = Path("data/era5/zarr")
    zarr_dir.mkdir(parents=True, exist_ok=True)
    
    # Process surface data
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SURFACE DATA")
    logger.info("="*60)
    
    surface_ds = process_surface_data()
    if surface_ds is not None:
        surface_zarr = save_to_zarr(
            surface_ds, 
            zarr_dir / "era5_surface_2010_2020.zarr",
            "surface"
        )
        logger.info(f"✓ Surface data saved to: {surface_zarr}")
    else:
        logger.error("✗ Failed to process surface data")
    
    # Process pressure data  
    logger.info("\n" + "="*60)
    logger.info("PROCESSING PRESSURE DATA")
    logger.info("="*60)
    
    pressure_ds = process_pressure_data()
    if pressure_ds is not None:
        pressure_zarr = save_to_zarr(
            pressure_ds,
            zarr_dir / "era5_pressure_2010_2020.zarr", 
            "pressure"
        )
        logger.info(f"✓ Pressure data saved to: {pressure_zarr}")
    else:
        logger.error("✗ Failed to process pressure data")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DATA ORGANIZATION SUMMARY")
    logger.info("="*60)
    
    if surface_ds is not None:
        logger.info(f"Surface dataset: {dict(surface_ds.sizes)}")
        logger.info(f"  Time: {surface_ds.time.min().values} to {surface_ds.time.max().values}")
        logger.info(f"  Variables: {list(surface_ds.data_vars.keys())}")
    
    if pressure_ds is not None:
        logger.info(f"Pressure dataset: {dict(pressure_ds.sizes)}")
        logger.info(f"  Time: {pressure_ds.time.min().values} to {pressure_ds.time.max().values}")
        logger.info(f"  Variables: {list(pressure_ds.data_vars.keys())}")
        logger.info(f"  Levels: {pressure_ds.isobaricInhPa.values}")
    
    logger.info("ERA5 data organization completed!")


if __name__ == "__main__":
    main()