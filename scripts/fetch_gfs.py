#!/usr/bin/env python3
"""
GFS Real-time Data Fetcher for Adelaide Weather Forecasting
Downloads GFS operational forecasts for the Adelaide region.
"""

import os
import yaml
import logging
import requests
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gfs_fetch.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path='../configs/data.yaml'):
    """Load data configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_latest_gfs_cycle():
    """Get the latest available GFS cycle."""
    now = datetime.utcnow()
    
    # GFS runs at 00, 06, 12, 18 UTC with ~3-4 hour delay
    cycles = [0, 6, 12, 18]
    
    # Find the most recent cycle that should be available
    current_hour = now.hour
    for cycle in reversed(cycles):
        if current_hour >= cycle + 4:  # Account for processing delay
            cycle_time = now.replace(hour=cycle, minute=0, second=0, microsecond=0)
            break
    else:
        # If no cycle available today, use last cycle from yesterday
        yesterday = now - timedelta(days=1)
        cycle_time = yesterday.replace(hour=18, minute=0, second=0, microsecond=0)
    
    return cycle_time


def build_gfs_url(cycle_time, forecast_hour, variable_set='pgrb2'):
    """Build GFS download URL."""
    
    # NOMADS GFS 0.25 degree
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    
    # Format cycle time
    cycle_str = cycle_time.strftime('%Y%m%d')
    cycle_hour = cycle_time.strftime('%H')
    
    # Variable selection for weather forecasting
    variables = [
        'HGT:500',           # 500mb geopotential height
        'TMP:850',           # 850mb temperature  
        'UGRD:850',          # 850mb U wind
        'VGRD:850',          # 850mb V wind
        'SPFH:850',          # 850mb specific humidity
        'PRMSL',             # Mean sea level pressure
        'TMP:2_m',           # 2m temperature
        'UGRD:10_m',         # 10m U wind
        'VGRD:10_m',         # 10m V wind
        'APCP',              # Accumulated precipitation
        'CAPE',              # CAPE
    ]
    
    # Adelaide region: roughly 32-38S, 136-142E
    # GFS grid coordinates (0-360 longitude)
    area_params = {
        'leftlon': 136,
        'rightlon': 142,
        'toplat': -32,
        'bottomlat': -38
    }
    
    # Build URL parameters
    params = {
        'file': f'gfs.t{cycle_hour}z.pgrb2.0p25.f{forecast_hour:03d}',
        'var_' + var: 'on' for var in variables
    }
    params.update(area_params)
    params['dir'] = f'/gfs.{cycle_str}/{cycle_hour}/atmos'
    
    # Construct full URL
    param_str = '&'.join([f'{k}={v}' for k, v in params.items()])
    url = f"{base_url}?{param_str}"
    
    return url


def download_gfs_file(url, output_path, timeout=300):
    """Download a single GFS file."""
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Downloading: {url}")
        
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Download to temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
            tmp_path = tmp_file.name
        
        # Move to final location
        shutil.move(tmp_path, output_path)
        
        logger.info(f"Downloaded: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def process_gfs_forecast(cycle_time, max_forecast_hours=72):
    """Download and process a complete GFS forecast."""
    
    logger = logging.getLogger(__name__)
    
    # Set up directories
    base_dir = Path(__file__).parent.parent
    gfs_dir = base_dir / "data" / "gfs"
    gfs_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = base_dir / "data" / "processed" 
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    cycle_str = cycle_time.strftime('%Y%m%d_%H')
    
    logger.info(f"Processing GFS forecast for cycle: {cycle_str}")
    
    downloaded_files = []
    forecast_hours = list(range(0, max_forecast_hours + 1, 3))  # Every 3 hours
    
    # Download all forecast hours
    for fhour in forecast_hours:
        try:
            url = build_gfs_url(cycle_time, fhour)
            filename = f"gfs_{cycle_str}_f{fhour:03d}.grib2"
            filepath = gfs_dir / filename
            
            if not filepath.exists():
                download_gfs_file(url, filepath)
            else:
                logger.info(f"File already exists: {filepath}")
                
            downloaded_files.append(filepath)
            
        except Exception as e:
            logger.error(f"Failed to download forecast hour {fhour}: {e}")
            continue
    
    if not downloaded_files:
        raise ValueError("No GFS files downloaded successfully")
    
    # Convert to combined dataset
    zarr_path = processed_dir / f"gfs_{cycle_str}.zarr"
    
    if not zarr_path.exists():
        logger.info("Converting GFS files to Zarr format...")
        combine_gfs_to_zarr(downloaded_files, zarr_path)
        
        # Clean up GRIB files to save space
        for filepath in downloaded_files:
            filepath.unlink()
        logger.info("Removed temporary GRIB files")
    
    return zarr_path


def combine_gfs_to_zarr(grib_files, output_path):
    """Combine GFS GRIB files into a single Zarr dataset."""
    
    logger = logging.getLogger(__name__)
    
    datasets = []
    
    for grib_file in grib_files:
        try:
            # Open with xarray and cfgrib
            ds = xr.open_dataset(grib_file, engine='cfgrib')
            
            # Add forecast hour coordinate
            forecast_hour = int(grib_file.stem.split('_f')[-1])
            ds = ds.expand_dims('forecast_hour')
            ds['forecast_hour'] = [forecast_hour]
            
            datasets.append(ds)
            
        except Exception as e:
            logger.warning(f"Failed to process {grib_file}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No GRIB files could be processed")
    
    # Combine along forecast dimension
    combined = xr.concat(datasets, dim='forecast_hour')
    
    # Add metadata
    combined.attrs.update({
        'title': 'GFS Operational Forecast for Adelaide Region',
        'description': 'Global Forecast System operational weather forecast',
        'source': 'NOAA/NCEP GFS',
        'region': 'Adelaide, Australia (32-38S, 136-142E)',
        'download_date': datetime.now().isoformat(),
        'grid_resolution': '0.25 degrees',
    })
    
    # Save to Zarr with compression
    encoding = {}
    for var in combined.data_vars:
        encoding[var] = {
            'compressor': 'lz4',
            'chunks': {'forecast_hour': 25, 'latitude': 25, 'longitude': 25}
        }
    
    combined.to_zarr(output_path, encoding=encoding)
    logger.info(f"Saved combined GFS forecast: {output_path}")


def fetch_latest_gfs():
    """Fetch the latest available GFS forecast."""
    
    logger = setup_logging()
    logger.info("Starting GFS real-time fetch...")
    
    try:
        # Get latest cycle
        cycle_time = get_latest_gfs_cycle()
        logger.info(f"Latest GFS cycle: {cycle_time}")
        
        # Process forecast
        zarr_path = process_gfs_forecast(cycle_time)
        
        logger.info(f"✓ Successfully fetched GFS forecast: {zarr_path}")
        return zarr_path
        
    except Exception as e:
        logger.error(f"✗ GFS fetch failed: {e}")
        raise


def main():
    """Main function for command line usage."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch GFS operational forecasts')
    parser.add_argument('--max-hours', type=int, default=72,
                       help='Maximum forecast hours to download (default: 72)')
    parser.add_argument('--cycle', type=str,
                       help='Specific cycle to download (format: YYYYMMDDHH)')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        if args.cycle:
            # Parse specific cycle
            cycle_time = datetime.strptime(args.cycle, '%Y%m%d%H')
        else:
            # Get latest cycle
            cycle_time = get_latest_gfs_cycle()
        
        logger.info(f"Fetching GFS cycle: {cycle_time}")
        
        # Process forecast
        zarr_path = process_gfs_forecast(cycle_time, args.max_hours)
        
        logger.info(f"✓ GFS forecast saved: {zarr_path}")
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())