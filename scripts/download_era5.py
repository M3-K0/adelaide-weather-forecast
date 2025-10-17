#!/usr/bin/env python3
"""
ERA5 Data Download Script for Adelaide Weather Forecasting
Downloads ERA5 reanalysis data for the Adelaide region (25°×25° box).
"""

import os
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
import cdsapi
import xarray as xr
import zarr


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('era5_download.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path=None):
    if config_path is None:
        # Try different possible paths
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir / '../configs/data.yaml',
            script_dir.parent / 'configs/data.yaml',
            Path('configs/data.yaml'),
            Path('../configs/data.yaml')
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError("Could not find data.yaml config file")
            
    config_path = Path(config_path)
    """Load data configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def download_era5_pressure_levels(c, config, year, output_dir):
    """Download ERA5 pressure level data."""
    
    logger = logging.getLogger(__name__)
    
    # Pressure level variables
    pressure_vars = [
        'geopotential',      # z500
        'temperature',       # t850
        'specific_humidity', # q850
        'u_component_of_wind',  # u850
        'v_component_of_wind',  # v850
    ]
    
    pressure_levels = ['500', '850']
    area = config['era5']['area']  # [N, W, S, E]
    
    # Generate all months for the year
    months = [f"{month:02d}" for month in range(1, 13)]
    
    output_file = output_dir / f"era5_pressure_levels_{year}.nc"
    
    if output_file.exists():
        logger.info(f"File {output_file} already exists, skipping...")
        return output_file
        
    logger.info(f"Downloading ERA5 pressure levels for {year}...")
    
    try:
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': pressure_vars,
                'pressure_level': pressure_levels,
                'year': str(year),
                'month': months,
                'day': [f"{day:02d}" for day in range(1, 32)],
                'time': [f"{hour:02d}:00" for hour in range(24)],
                'area': area,
                'grid': config['era5']['grid'],
                'format': 'netcdf',
            },
            str(output_file)
        )
        
        logger.info(f"Successfully downloaded {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to download pressure levels for {year}: {e}")
        if output_file.exists():
            output_file.unlink()  # Remove partial file
        raise


def download_era5_surface(c, config, year, output_dir):
    """Download ERA5 surface data."""
    
    logger = logging.getLogger(__name__)
    
    # Surface variables
    surface_vars = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        'total_precipitation',
        'convective_available_potential_energy',
        'mean_sea_level_pressure',
    ]
    
    area = config['era5']['area']
    months = [f"{month:02d}" for month in range(1, 13)]
    
    output_file = output_dir / f"era5_surface_{year}.nc"
    
    if output_file.exists():
        logger.info(f"File {output_file} already exists, skipping...")
        return output_file
        
    logger.info(f"Downloading ERA5 surface variables for {year}...")
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': surface_vars,
                'year': str(year),
                'month': months,
                'day': [f"{day:02d}" for day in range(1, 32)],
                'time': [f"{hour:02d}:00" for hour in range(24)],
                'area': area,
                'grid': config['era5']['grid'],
                'format': 'netcdf',
            },
            str(output_file)
        )
        
        logger.info(f"Successfully downloaded {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to download surface variables for {year}: {e}")
        if output_file.exists():
            output_file.unlink()
        raise


def combine_and_convert_to_zarr(pressure_file, surface_file, year, zarr_dir):
    """Combine pressure and surface data and convert to Zarr."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Converting {year} data to Zarr format...")
    
    try:
        # Open both datasets
        ds_pressure = xr.open_dataset(pressure_file)
        ds_surface = xr.open_dataset(surface_file)
        
        # Combine datasets
        ds_combined = xr.merge([ds_pressure, ds_surface])
        
        # Add metadata
        ds_combined.attrs.update({
            'title': f'ERA5 Reanalysis Data for Adelaide Region - {year}',
            'description': 'Pressure level and surface variables for weather forecasting',
            'region': 'Adelaide, Australia (25°×25° box)',
            'source': 'ERA5 reanalysis from ECMWF',
            'download_date': datetime.now().isoformat(),
            'grid_resolution': '0.25 degrees',
        })
        
        # Create zarr output path
        zarr_path = zarr_dir / f"era5_{year}.zarr"
        
        if zarr_path.exists():
            logger.info(f"Zarr file {zarr_path} already exists, skipping conversion...")
            return zarr_path
        
        # Save to Zarr with chunking for efficient access
        encoding = {}
        for var in ds_combined.data_vars:
            encoding[var] = {
                'chunks': {'time': 24, 'latitude': 50, 'longitude': 50},
                'compressor': zarr.Blosc(cname='lz4', clevel=5)
            }
        
        ds_combined.to_zarr(zarr_path, encoding=encoding)
        logger.info(f"Successfully created Zarr dataset: {zarr_path}")
        
        # Clean up original NetCDF files to save space
        pressure_file.unlink()
        surface_file.unlink()
        logger.info("Removed temporary NetCDF files")
        
        return zarr_path
        
    except Exception as e:
        logger.error(f"Failed to convert {year} to Zarr: {e}")
        raise


def main():
    """Main download function."""
    
    logger = setup_logging()
    logger.info("Starting ERA5 data download for Adelaide weather forecasting")
    
    # Load configuration
    config = load_config()
    
    # Set up directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "era5"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zarr_dir = base_dir / "data" / "processed"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize CDS API client
    try:
        c = cdsapi.Client()
        logger.info("CDS API client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CDS API client: {e}")
        logger.error("Please check your CDS API credentials in ~/.cdsapirc")
        return
    
    # Download data for each year
    years = range(config['era5']['years'][0], config['era5']['years'][1] + 1)
    
    successful_downloads = []
    failed_downloads = []
    
    for year in years:
        try:
            logger.info(f"Processing year {year}...")
            
            # Download pressure level data
            pressure_file = download_era5_pressure_levels(c, config, year, data_dir)
            
            # Download surface data
            surface_file = download_era5_surface(c, config, year, data_dir)
            
            # Convert to Zarr
            zarr_path = combine_and_convert_to_zarr(
                pressure_file, surface_file, year, zarr_dir
            )
            
            successful_downloads.append((year, zarr_path))
            logger.info(f"✓ Successfully processed {year}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {year}: {e}")
            failed_downloads.append((year, str(e)))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ERA5 Download Summary")
    logger.info("="*60)
    logger.info(f"Successful downloads: {len(successful_downloads)}")
    for year, path in successful_downloads:
        logger.info(f"  {year}: {path}")
        
    if failed_downloads:
        logger.info(f"Failed downloads: {len(failed_downloads)}")
        for year, error in failed_downloads:
            logger.info(f"  {year}: {error}")
    
    logger.info("ERA5 download process completed")


if __name__ == "__main__":
    main()