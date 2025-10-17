#!/usr/bin/env python3
"""
ERA5 Quarterly Data Downloader for Adelaide Weather Forecasting
Downloads ERA5 data in quarterly chunks - sweet spot between queue waits and size limits.
"""

import os
import yaml
import logging
import cdsapi
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import calendar
import sys
from tqdm import tqdm


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('era5_quarterly_download.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path=None):
    """Load data configuration."""
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
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def download_era5_quarter(client, year, quarter, variables, area, output_dir, dataset_type='pressure_levels'):
    """Download ERA5 data for a quarter (3 months) with progress monitoring."""
    
    logger = logging.getLogger(__name__)
    
    # Define quarters
    quarters = {
        1: [1, 2, 3],    # Q1: Jan-Mar
        2: [4, 5, 6],    # Q2: Apr-Jun  
        3: [7, 8, 9],    # Q3: Jul-Sep
        4: [10, 11, 12]  # Q4: Oct-Dec
    }
    
    months = quarters[quarter]
    quarter_name = f"Q{quarter}"
    
    # Create filename
    if dataset_type == 'pressure_levels':
        filename = f"era5_pressure_{year}_{quarter_name}.nc"
        dataset_name = 'reanalysis-era5-pressure-levels'
    else:
        filename = f"era5_surface_{year}_{quarter_name}.nc"
        dataset_name = 'reanalysis-era5-single-levels'
    
    output_path = output_dir / filename
    
    # Skip if already exists
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"⏭️  Skipping {filename} ({file_size:.1f} MB - already exists)")
        return output_path
    
    # Prepare request parameters for quarter
    request_params = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'year': str(year),
        'month': [f"{month:02d}" for month in months],  # 3 months
        'day': [f"{day:02d}" for day in range(1, 32)],  # All days
        'time': ['00:00', '06:00', '12:00', '18:00'],    # 6-hourly
        'area': area,  # [North, West, South, East]
    }
    
    # Add variables based on dataset type
    if dataset_type == 'pressure_levels':
        request_params.update({
            'pressure_level': ['500', '850'],
            'variable': variables['pressure_levels']
        })
    else:
        request_params.update({
            'variable': variables['surface']
        })
    
    print(f"\n🌍 Requesting {filename}")
    print(f"   📊 Variables: {len(request_params['variable'])}")
    print(f"   📅 Quarter: {months} (~90 days)")
    print(f"   📍 Area: Adelaide region")
    
    try:
        # Submit request with progress monitoring
        start_time = time.time()
        
        print(f"📥 Downloading {filename}")
        
        pbar = tqdm(
            total=100,
            desc="Progress",
            unit="%",
            bar_format="{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Submit the request
        client.retrieve(
            dataset_name,
            request_params,
            str(output_path)
        )
        
        # Complete progress
        pbar.n = 100
        pbar.refresh()
        pbar.close()
        
        download_time = time.time() - start_time
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        download_speed = file_size / download_time if download_time > 0 else 0
        
        print(f"✓ Completed: {file_size:.1f} MB in {download_time:.1f}s ({download_speed:.2f} MB/s)")
        
        # Log summary
        logger.info(f"✓ Downloaded {filename}: {file_size:.1f} MB in {download_time:.1f}s ({download_speed:.2f} MB/s)")
        
        return output_path
        
    except Exception as e:
        if 'pbar' in locals():
            pbar.close()
        
        logger.error(f"✗ Failed to download {filename}: {e}")
        
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        
        raise


def convert_to_zarr(netcdf_files, output_zarr, dataset_type='pressure_levels'):
    """Convert quarterly NetCDF files to a single Zarr dataset."""
    
    logger = logging.getLogger(__name__)
    
    if output_zarr.exists():
        logger.info(f"Zarr dataset already exists: {output_zarr}")
        return output_zarr
    
    logger.info(f"Converting {len(netcdf_files)} NetCDF files to Zarr...")
    
    # Sort files chronologically
    netcdf_files = sorted(netcdf_files)
    
    # Open and combine datasets
    datasets = []
    
    for nc_file in netcdf_files:
        try:
            logger.info(f"Processing {nc_file.name}...")
            
            # Open with xarray
            ds = xr.open_dataset(nc_file)
            
            # Ensure proper time coordinate
            if 'time' in ds.coords:
                ds = ds.sortby('time')
            
            datasets.append(ds)
            
        except Exception as e:
            logger.warning(f"Failed to process {nc_file}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid NetCDF files to process")
    
    # Combine all datasets
    logger.info("Combining datasets...")
    combined = xr.concat(datasets, dim='time', combine_attrs='override')
    
    # Sort by time
    combined = combined.sortby('time')
    
    # Add metadata
    combined.attrs.update({
        'title': f'ERA5 {dataset_type.replace("_", " ").title()} for Adelaide Region',
        'description': 'ERA5 reanalysis data downloaded from Copernicus CDS',
        'source': 'ERA5 reanalysis from ECMWF',
        'region': 'Adelaide, Australia (32-38S, 136-142E)',
        'download_date': datetime.now().isoformat(),
        'grid_resolution': '0.25 degrees',
        'temporal_resolution': '6-hourly',
    })
    
    # Set up chunking for efficient access
    encoding = {}
    chunk_time = min(365 * 4, len(combined.time))  # ~1 year of 6-hourly data
    
    for var in combined.data_vars:
        chunks = {'time': chunk_time}
        
        # Add spatial chunking if present
        if 'latitude' in combined[var].dims:
            chunks['latitude'] = min(25, len(combined.latitude))
        if 'longitude' in combined[var].dims:
            chunks['longitude'] = min(25, len(combined.longitude))
        if 'level' in combined[var].dims:
            chunks['level'] = len(combined.level)
        
        encoding[var] = {
            'compressor': 'lz4',
            'chunks': chunks
        }
    
    # Save to Zarr
    logger.info(f"Saving to Zarr: {output_zarr}")
    combined.to_zarr(output_zarr, encoding=encoding)
    
    # Close datasets to free memory
    for ds in datasets:
        ds.close()
    combined.close()
    
    logger.info(f"✓ Zarr conversion complete: {output_zarr}")
    
    return output_zarr


def download_era5_dataset(start_year=2010, end_year=2020):
    """Download complete ERA5 dataset in quarterly chunks."""
    
    logger = setup_logging()
    logger.info("Starting ERA5 quarterly download for Adelaide weather forecasting")
    
    # Load configuration
    try:
        config = load_config()
        era5_config = config['era5']
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Initialize CDS API client
    try:
        client = cdsapi.Client()
        logger.info("CDS API client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CDS API client: {e}")
        return
    
    # Create output directories
    base_dir = Path(__file__).parent.parent
    era5_dir = base_dir / "data" / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)
    
    pressure_dir = era5_dir / "pressure_levels"
    surface_dir = era5_dir / "surface"
    zarr_dir = era5_dir / "zarr"
    
    pressure_dir.mkdir(exist_ok=True)
    surface_dir.mkdir(exist_ok=True)
    zarr_dir.mkdir(exist_ok=True)
    
    # Extract configuration
    area = era5_config['area']  # [North, West, South, East]
    
    # Map config variables to CDS API names
    variables = {
        'pressure_levels': [
            'geopotential',
            'temperature',
            'specific_humidity',
            'u_component_of_wind',
            'v_component_of_wind'
        ],
        'surface': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
            'total_precipitation',
            'mean_sea_level_pressure'
        ]
    }
    
    logger.info(f"Download configuration:")
    logger.info(f"  Area: {area}")
    logger.info(f"  Years: {start_year}-{end_year}")
    logger.info(f"  Pressure variables: {variables['pressure_levels']}")
    logger.info(f"  Surface variables: {variables['surface']}")
    
    # Download data quarter by quarter
    pressure_files = []
    surface_files = []
    failed_downloads = []
    
    total_years = end_year - start_year + 1
    total_quarters = total_years * 4
    total_files = total_quarters * 2  # pressure + surface
    
    # Overall progress tracking
    print(f"\n🚀 Starting ERA5 Quarterly Download Campaign")
    print(f"📅 Period: {start_year}-{end_year} ({total_years} years, {total_quarters} quarters)")
    print(f"📍 Region: Adelaide, Australia")
    print(f"💾 Output: {zarr_dir}")
    print(f"⚡ Strategy: Quarterly chunks (3 months each)")
    print(f"📊 Total requests: {total_files} (vs 264 monthly)")
    print(f"🕐 Current CDS queue: https://cds.climate.copernicus.eu/live/queue")
    print("="*60)
    
    campaign_start = time.time()
    
    # Create overall progress bar
    overall_pbar = tqdm(
        total=total_files,
        desc="Overall Progress",
        unit="files",
        position=0,
        leave=True
    )
    
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):  # Q1, Q2, Q3, Q4
            quarter_name = f"Q{quarter}"
            current_quarter = (year - start_year) * 4 + quarter
            
            print(f"\n📆 {year}-{quarter_name} - Quarter {current_quarter}/{total_quarters}")
            print("-" * 50)
            
            try:
                # Download pressure levels
                print("🔄 Pressure levels dataset:")
                pressure_file = download_era5_quarter(
                    client, year, quarter, variables, area, 
                    pressure_dir, 'pressure_levels'
                )
                pressure_files.append(pressure_file)
                overall_pbar.update(1)
                
                # Wait between requests to be nice to CDS
                time.sleep(3)
                
                # Download surface data
                print("\n🔄 Surface dataset:")
                surface_file = download_era5_quarter(
                    client, year, quarter, variables, area,
                    surface_dir, 'surface'
                )
                surface_files.append(surface_file)
                overall_pbar.update(1)
                
                # Calculate progress statistics
                elapsed_time = time.time() - campaign_start
                completed_files = len(pressure_files) + len(surface_files)
                remaining_files = total_files - completed_files
                
                if completed_files > 0:
                    avg_time_per_file = elapsed_time / completed_files
                    eta = remaining_files * avg_time_per_file
                    eta_hours = eta / 3600
                    
                    print(f"\n📈 Campaign Statistics:")
                    print(f"   ✅ Completed: {completed_files}/{total_files} files")
                    print(f"   ⏱️  Elapsed: {elapsed_time/3600:.1f} hours")
                    print(f"   🕐 ETA: {eta_hours:.1f} hours remaining")
                    print(f"   📊 Progress: {(completed_files/total_files*100):.1f}%")
                
                # Wait between quarters
                time.sleep(3)
                
            except Exception as e:
                error_msg = f"{year}-{quarter_name}: {str(e)}"
                failed_downloads.append(error_msg)
                print(f"\n❌ Failed {year}-{quarter_name}: {e}")
                
                # Wait longer after failure
                time.sleep(10)
                continue
    
    overall_pbar.close()
    
    # Convert to Zarr format
    logger.info("\n=== Converting to Zarr format ===")
    
    try:
        if pressure_files:
            pressure_zarr = convert_to_zarr(
                pressure_files, 
                zarr_dir / "era5_pressure_levels.zarr",
                'pressure_levels'
            )
        
        if surface_files:
            surface_zarr = convert_to_zarr(
                surface_files,
                zarr_dir / "era5_surface.zarr", 
                'surface'
            )
        
        logger.info("✓ Zarr conversion completed")
        
    except Exception as e:
        logger.error(f"✗ Zarr conversion failed: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ERA5 Quarterly Download Summary")
    logger.info("="*60)
    logger.info(f"Successful pressure downloads: {len(pressure_files)}")
    logger.info(f"Successful surface downloads: {len(surface_files)}")
    logger.info(f"Failed downloads: {len(failed_downloads)}")
    
    if failed_downloads:
        logger.info("Failed downloads:")
        for failure in failed_downloads:
            logger.info(f"  {failure}")
    
    logger.info("ERA5 quarterly download process completed")


def main():
    """Main function for command line usage."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Download ERA5 data in quarterly chunks')
    parser.add_argument('--start-year', type=int, default=2010,
                       help='Start year (default: 2010)')
    parser.add_argument('--end-year', type=int, default=2020,
                       help='End year (default: 2020)')
    parser.add_argument('--test-quarter', action='store_true',
                       help='Download only 2020-Q1 for testing')
    
    args = parser.parse_args()
    
    if args.test_quarter:
        # Test with just Q1 2020
        logger = setup_logging()
        logger.info("Test mode: downloading 2020-Q1 only")
        download_era5_dataset(2020, 2020)
    else:
        download_era5_dataset(args.start_year, args.end_year)


if __name__ == "__main__":
    main()