#!/usr/bin/env python3
"""
Simple Analog Ensemble Baseline for Adelaide Weather Forecasting
Implements basic analog forecast using L2 distance on meteorological fields.
"""

import numpy as np
import xarray as xr
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


class BaselineAnalogForecaster:
    """Simple analog ensemble forecaster using L2 distance."""
    
    def __init__(self, config_path='../configs/model.yaml'):
        """Initialize the baseline forecaster."""
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.baseline_config = config['baselines']['simple_analog']
        self.num_analogs = self.baseline_config['num_analogs']
        self.distance_metric = self.baseline_config['distance_metric']
        self.variables = self.baseline_config['variables']
        
        # Model components
        self.scaler = StandardScaler()
        self.analog_database = None
        self.analog_targets = None
        self.is_fitted = False
        
        # Adelaide grid points (approximate)
        self.adelaide_lat = -34.9285
        self.adelaide_lon = 138.6007
        
    def extract_features(self, dataset, target_time):
        """Extract meteorological features for analog matching."""
        
        # Get data at target time
        data_slice = dataset.sel(time=target_time, method='nearest')
        
        features = []
        
        for var in self.variables:
            if var == 'z500':
                # 500mb geopotential height
                var_data = data_slice['geopotential'].sel(plev=500)
            elif var == 'mslp':
                # Mean sea level pressure
                var_data = data_slice['mean_sea_level_pressure']
            elif var == 't850':
                # 850mb temperature
                var_data = data_slice['temperature'].sel(plev=850)
            elif var == 'u850':
                # 850mb U wind
                var_data = data_slice['u_component_of_wind'].sel(plev=850)
            elif var == 'v850':
                # 850mb V wind
                var_data = data_slice['v_component_of_wind'].sel(plev=850)
            elif var == 'q850':
                # 850mb specific humidity
                var_data = data_slice['specific_humidity'].sel(plev=850)
            else:
                self.logger.warning(f"Unknown variable: {var}")
                continue
            
            # Flatten spatial dimensions
            var_flat = var_data.values.flatten()
            
            # Remove NaN values
            var_flat = var_flat[~np.isnan(var_flat)]
            
            features.extend(var_flat)
        
        return np.array(features)
    
    def extract_targets(self, dataset, target_time, lead_hours):
        """Extract forecast targets (Adelaide surface conditions)."""
        
        forecast_time = target_time + timedelta(hours=lead_hours)
        
        try:
            # Get data at forecast time
            forecast_data = dataset.sel(time=forecast_time, method='nearest')
            
            # Extract Adelaide area values
            adelaide_data = forecast_data.sel(
                latitude=slice(self.adelaide_lat - 0.5, self.adelaide_lat + 0.5),
                longitude=slice(self.adelaide_lon - 0.5, self.adelaide_lon + 0.5)
            )
            
            targets = {}
            
            # 2m temperature
            if '2m_temperature' in forecast_data:
                targets['2m_temperature'] = adelaide_data['2m_temperature'].mean().values
            
            # Total precipitation (accumulated over lead time)
            if 'total_precipitation' in forecast_data:
                targets['total_precipitation'] = adelaide_data['total_precipitation'].mean().values
            
            # Mean sea level pressure
            if 'mean_sea_level_pressure' in forecast_data:
                targets['mean_sea_level_pressure'] = adelaide_data['mean_sea_level_pressure'].mean().values
            
            return targets
            
        except KeyError as e:
            self.logger.warning(f"Target time {forecast_time} not found: {e}")
            return None
    
    def build_analog_database(self, dataset, train_years, lead_hours=24):
        """Build database of analog patterns and their outcomes."""
        
        self.logger.info(f"Building analog database for {train_years} with {lead_hours}h lead time")
        
        # Filter to training years
        train_data = dataset.sel(time=slice(f'{train_years[0]}', f'{train_years[1]}'))
        
        # Get all valid times (excluding those too close to end)
        all_times = train_data.time.values
        valid_times = all_times[:-int(lead_hours/6)]  # Assuming 6-hourly data
        
        features_list = []
        targets_list = []
        
        self.logger.info(f"Processing {len(valid_times)} analog cases...")
        
        for i, valid_time in enumerate(valid_times):
            if i % 100 == 0:
                self.logger.info(f"Processed {i}/{len(valid_times)} cases")
            
            try:
                # Extract features at analysis time
                features = self.extract_features(train_data, valid_time)
                
                # Extract targets at forecast time
                targets = self.extract_targets(train_data, valid_time, lead_hours)
                
                if features is not None and targets is not None:
                    features_list.append(features)
                    targets_list.append(targets)
                    
            except Exception as e:
                self.logger.warning(f"Failed to process time {valid_time}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid analog cases found")
        
        # Convert to arrays
        self.analog_database = np.array(features_list)
        self.analog_targets = targets_list
        
        # Fit scaler
        self.scaler.fit(self.analog_database)
        self.analog_database = self.scaler.transform(self.analog_database)
        
        self.is_fitted = True
        
        self.logger.info(f"Analog database built: {len(self.analog_database)} cases")
        self.logger.info(f"Feature dimensions: {self.analog_database.shape[1]}")
    
    def find_analogs(self, query_features, k=None):
        """Find k most similar analog cases."""
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call build_analog_database first.")
        
        if k is None:
            k = self.num_analogs
        
        # Normalize query features
        query_normalized = self.scaler.transform(query_features.reshape(1, -1))
        
        # Compute distances
        if self.distance_metric == 'l2':
            distances = euclidean_distances(query_normalized, self.analog_database)[0]
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Find k nearest neighbors
        analog_indices = np.argsort(distances)[:k]
        analog_distances = distances[analog_indices]
        
        return analog_indices, analog_distances
    
    def predict(self, query_features, lead_hours=24, k=None):
        """Make analog ensemble forecast."""
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call build_analog_database first.")
        
        # Find analogs
        analog_indices, analog_distances = self.find_analogs(query_features, k)
        
        # Retrieve analog outcomes
        analog_outcomes = [self.analog_targets[i] for i in analog_indices]
        
        # Compute ensemble forecast
        forecast = {}
        
        # Get all available target variables
        if analog_outcomes:
            target_vars = analog_outcomes[0].keys()
            
            for var in target_vars:
                values = [outcome[var] for outcome in analog_outcomes if var in outcome]
                
                if values:
                    # Simple ensemble mean
                    forecast[var] = np.mean(values)
                    
                    # Add ensemble spread
                    forecast[f'{var}_std'] = np.std(values)
                    forecast[f'{var}_min'] = np.min(values)
                    forecast[f'{var}_max'] = np.max(values)
        
        # Add metadata
        forecast['num_analogs'] = len(analog_indices)
        forecast['mean_distance'] = np.mean(analog_distances)
        forecast['lead_hours'] = lead_hours
        
        return forecast
    
    def evaluate_persistence(self, dataset, test_years, lead_hours=24):
        """Evaluate persistence baseline."""
        
        self.logger.info("Evaluating persistence baseline...")
        
        # Filter to test years
        test_data = dataset.sel(time=slice(f'{test_years[0]}', f'{test_years[1]}'))
        
        all_times = test_data.time.values
        valid_times = all_times[:-int(lead_hours/6)]
        
        predictions = []
        observations = []
        
        for valid_time in valid_times[:100]:  # Limit for speed
            try:
                # Current conditions (persistence forecast)
                current = self.extract_targets(test_data, valid_time, 0)
                
                # Actual future conditions
                future = self.extract_targets(test_data, valid_time, lead_hours)
                
                if current and future:
                    predictions.append(current)
                    observations.append(future)
                    
            except Exception as e:
                continue
        
        # Compute metrics
        metrics = {}
        if predictions and observations:
            pred_temp = [p.get('2m_temperature', np.nan) for p in predictions]
            obs_temp = [o.get('2m_temperature', np.nan) for o in observations]
            
            valid_pairs = [(p, o) for p, o in zip(pred_temp, obs_temp) 
                          if not (np.isnan(p) or np.isnan(o))]
            
            if valid_pairs:
                pred_array = np.array([p for p, o in valid_pairs])
                obs_array = np.array([o for p, o in valid_pairs])
                
                metrics['persistence_rmse'] = np.sqrt(np.mean((pred_array - obs_array)**2))
                metrics['persistence_mae'] = np.mean(np.abs(pred_array - obs_array))
                metrics['persistence_bias'] = np.mean(pred_array - obs_array)
                metrics['n_samples'] = len(valid_pairs)
        
        return metrics
    
    def save_model(self, filepath):
        """Save trained model."""
        
        model_data = {
            'analog_database': self.analog_database,
            'analog_targets': self.analog_targets,
            'scaler': self.scaler,
            'config': {
                'num_analogs': self.num_analogs,
                'distance_metric': self.distance_metric,
                'variables': self.variables
            },
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model."""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.analog_database = model_data['analog_database']
        self.analog_targets = model_data['analog_targets']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        
        # Update config
        config = model_data['config']
        self.num_analogs = config['num_analogs']
        self.distance_metric = config['distance_metric']
        self.variables = config['variables']
        
        self.logger.info(f"Model loaded: {filepath}")


def demo_baseline_analog():
    """Demonstrate baseline analog forecasting."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Adelaide Weather Forecasting - Baseline Analog Demo")
    
    # This is a demo - in practice you'd load real ERA5 data
    logger.info("Note: This demo requires ERA5 data to be downloaded first")
    logger.info("Use scripts/download_era5.py to download training data")
    
    # Show configuration
    forecaster = BaselineAnalogForecaster()
    logger.info(f"Analog forecaster configuration:")
    logger.info(f"  Variables: {forecaster.variables}")
    logger.info(f"  Number of analogs: {forecaster.num_analogs}")
    logger.info(f"  Distance metric: {forecaster.distance_metric}")
    
    return forecaster


if __name__ == "__main__":
    demo_baseline_analog()