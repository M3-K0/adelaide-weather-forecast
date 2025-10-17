#!/usr/bin/env python3
"""
Train CNN Embeddings with InfoNCE Loss for Weather Forecasting
Optimized for RTX 3060 with comprehensive progress tracking.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
from tqdm import tqdm
from typing import Dict, Tuple, List
import json

# Optional wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.cnn_encoder import WeatherCNNEncoder


def setup_logging(output_dir: Path):
    """Set up logging with file and console output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class WeatherDataset(Dataset):
    """Dataset for loading weather states from Zarr."""
    
    def __init__(self, surface_path: str, pressure_path: str, 
                 normalize: bool = False, cache_size: int = 100):
        """Initialize dataset with Zarr files."""
        
        # Load datasets
        self.surface_ds = xr.open_zarr(surface_path)
        self.pressure_ds = xr.open_zarr(pressure_path)
        
        # Align time coordinates
        common_times = np.intersect1d(
            self.surface_ds.time.values, 
            self.pressure_ds.time.values
        )
        
        self.surface_ds = self.surface_ds.sel(time=common_times)
        self.pressure_ds = self.pressure_ds.sel(time=common_times)
        
        self.n_samples = len(common_times)
        self.times = common_times
        
        # Cache for faster access
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.total_accesses = 0
        
        # Skip normalization for now to avoid hanging
        self.norm_stats = None
            
        print(f"Dataset initialized: {self.n_samples} samples")
        print(f"Time range: {self.times[0]} to {self.times[-1]}")
    
    def compute_norm_stats(self):
        """Compute mean and std for normalization."""
        print("Computing normalization statistics...")
        
        # Sample subset for stats
        sample_idx = np.random.choice(self.n_samples, 
                                    min(1000, self.n_samples), 
                                    replace=False)
        
        samples = []
        for idx in tqdm(sample_idx, desc="Computing stats"):
            sample = self.get_raw_sample(idx)
            samples.append(sample)
        
        samples = np.stack(samples)
        self.norm_stats = {
            'mean': samples.mean(axis=(0, 2, 3)),
            'std': samples.std(axis=(0, 2, 3)) + 1e-6
        }
        print(f"Normalization stats computed for {samples.shape[1]} channels")
    
    def get_raw_sample(self, idx: int) -> np.ndarray:
        """Get raw sample without normalization."""
        
        time = self.times[idx]
        
        # Get surface variables (only msl for now)
        surface_data = self.surface_ds.sel(time=time)
        msl = surface_data['msl'].values  # (lat, lon)
        
        # Get pressure level variables
        pressure_data = self.pressure_ds.sel(time=time)
        
        # Stack all variables
        channels = []
        
        # Surface variables
        channels.append(msl)
        
        # Pressure variables at each level
        for level in [850, 500]:
            level_data = pressure_data.sel(isobaricInhPa=level)
            channels.append(level_data['z'].values)  # geopotential
            channels.append(level_data['t'].values)  # temperature
            channels.append(level_data['q'].values)  # humidity
            channels.append(level_data['u'].values)  # u-wind
            channels.append(level_data['v'].values)  # v-wind
        
        # Stack into (C, H, W) format
        sample = np.stack(channels, axis=0).astype(np.float32)
        
        return sample
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a weather state sample."""
        
        self.total_accesses += 1
        
        # Check cache
        if idx in self.cache:
            sample = self.cache[idx]
            self.cache_hits += 1
        else:
            sample = self.get_raw_sample(idx)
            
            # Normalize if stats available
            if self.norm_stats is not None:
                mean = self.norm_stats['mean'][:, np.newaxis, np.newaxis]
                std = self.norm_stats['std'][:, np.newaxis, np.newaxis]
                # Add epsilon to prevent division by zero
                std = np.maximum(std, 1e-6)
                sample = (sample - mean) / std
                
                # Replace any NaN or Inf values
                sample = np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Update cache
            if len(self.cache) < self.cache_size:
                self.cache[idx] = sample
        
        # Get temporal features (for FiLM conditioning)
        time = self.times[idx]
        try:
            # Python 3.13+ compatible
            from datetime import timezone
            dt = datetime.fromtimestamp(time.astype('datetime64[s]').astype(int), timezone.utc)
        except (AttributeError, ImportError):
            # Fallback for older Python
            dt = datetime.utcfromtimestamp(time.astype('datetime64[s]').astype(int))
        
        day_of_year = dt.timetuple().tm_yday
        temporal_features = np.array([
            dt.hour / 24.0,           # Hour of day
            day_of_year / 365.0,      # Day of year  
            np.sin(2 * np.pi * day_of_year / 365),  # Seasonal sin
            np.cos(2 * np.pi * day_of_year / 365),  # Seasonal cos
        ], dtype=np.float32)
        
        return {
            'data': torch.from_numpy(sample),
            'temporal': torch.from_numpy(temporal_features),
            'time': int(time.astype('datetime64[s]').astype(int)),  # Convert to timestamp
            'idx': idx
        }


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for embedding learning."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings: torch.Tensor, 
                positive_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            embeddings: (B, D) embedding vectors
            positive_mask: (B, B) boolean mask for positive pairs
        """
        
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create default positive mask (consecutive samples are similar)
        if positive_mask is None:
            positive_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
            # Add temporal neighbors as positives
            positive_mask = positive_mask | torch.diag(torch.ones(batch_size-1, dtype=torch.bool, device=embeddings.device), 1)
            positive_mask = positive_mask | torch.diag(torch.ones(batch_size-1, dtype=torch.bool, device=embeddings.device), -1)
        
        # Mask out self-similarity
        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        # Use fp16-safe mask value
        mask_value = -1e4  # Safe for both fp16 and fp32
        sim_matrix = sim_matrix.masked_fill(mask, mask_value)
        
        # InfoNCE loss
        # Get positive similarities (excluding self)
        pos_mask_no_self = positive_mask & ~mask
        
        # Check how many positives we have per sample
        pos_counts = pos_mask_no_self.sum(dim=1)
        
        if pos_counts.sum() > 0:
            # Extract positive similarities
            pos_sim = sim_matrix[pos_mask_no_self]
            
            # Group by original sample index
            pos_indices = torch.nonzero(pos_mask_no_self, as_tuple=True)
            sample_indices = pos_indices[0]
            
            # Compute loss per sample
            losses = []
            for i in range(batch_size):
                # Get positives for this sample
                sample_pos_mask = (sample_indices == i)
                if sample_pos_mask.sum() > 0:
                    sample_pos_sim = pos_sim[sample_pos_mask]
                    sample_logsumexp = torch.logsumexp(sim_matrix[i], dim=0)
                    sample_loss = -torch.mean(sample_pos_sim) + sample_logsumexp
                    losses.append(sample_loss)
                else:
                    # No positives for this sample, use standard InfoNCE
                    sample_logsumexp = torch.logsumexp(sim_matrix[i], dim=0)
                    losses.append(sample_logsumexp)
            
            loss = torch.mean(torch.stack(losses))
        else:
            # Fallback: standard contrastive loss without explicit positives
            # Each sample is its own positive (already masked out)
            loss = torch.mean(torch.logsumexp(sim_matrix, dim=1))
        
        return loss


class Trainer:
    """Training manager with progress tracking."""
    
    def __init__(self, model, train_loader, val_loader, config, output_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize loss and optimizer
        self.criterion = InfoNCELoss(temperature=config['temperature'])
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Store gradient clipping value
        self.gradient_clip = config.get('gradient_clip', 0.0)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['num_epochs']
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Progress tracking
        self.start_time = None
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.logger = setup_logging(self.output_dir)
        
        # Initialize wandb if available
        self.use_wandb = config.get('use_wandb', False) and HAS_WANDB
        if self.use_wandb:
            try:
                wandb.init(
                    project="weather-embeddings",
                    config=config,
                    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                wandb.watch(model)
            except:
                self.use_wandb = False
                self.logger.warning("Failed to initialize wandb")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0
        batch_times = []
        
        pbar = tqdm(self.train_loader, 
                   desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
                   ncols=120)
        
        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()
            
            # Move to device
            data = batch['data'].to(self.device)
            temporal = batch['temporal'].to(self.device)
            
            # Extract model inputs from temporal features
            # temporal: [hour/24, day_of_year/365, sin_seasonal, cos_seasonal]
            hours = (temporal[:, 0] * 24).long()  # Convert back to 0-23
            months = torch.zeros_like(hours)  # Placeholder - we don't have month info
            lead_times = torch.zeros_like(hours)  # Placeholder for lead time
            
            # Add data validation to prevent NaN
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"⚠️ NaN/Inf detected in input data at batch {batch_idx}")
                continue
                
            # Training step with NaN checking
            embeddings = self.model(data, lead_times, months, hours)
            
            # Check for NaN in embeddings
            if torch.isnan(embeddings).any():
                print(f"⚠️ NaN detected in embeddings at batch {batch_idx}")
                continue
                
            loss = self.criterion(embeddings)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"⚠️ NaN loss detected at batch {batch_idx}")
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            if hasattr(self, 'gradient_clip') and self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Track metrics
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            epoch_loss += loss.item()
            
            # Update progress bar
            avg_loss = epoch_loss / (batch_idx + 1)
            avg_time = np.mean(batch_times)
            eta = avg_time * (len(self.train_loader) - batch_idx - 1)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'batch_time': f'{batch_time:.3f}s',
                'eta': f'{eta:.1f}s'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_time': batch_time,
                })
        
        avg_epoch_loss = epoch_loss / len(self.train_loader)
        return avg_epoch_loss
    
    def validate(self) -> float:
        """Validate model."""
        self.model.eval()
        
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                data = batch['data'].to(self.device)
                temporal = batch['temporal'].to(self.device)
                
                # Extract model inputs from temporal features
                hours = (temporal[:, 0] * 24).long()
                months = torch.zeros_like(hours)
                lead_times = torch.zeros_like(hours)
                
                if self.scaler:
                    with autocast():
                        embeddings = self.model(data, lead_times, months, hours)
                        loss = self.criterion(embeddings)
                else:
                    embeddings = self.model(data, lead_times, months, hours)
                    loss = self.criterion(embeddings)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        # Keep only last 3 checkpoints
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
    
    def train(self):
        """Main training loop with progress tracking."""
        
        self.logger.info("="*60)
        self.logger.info("Starting Training")
        self.logger.info("="*60)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
        self.logger.info(f"Batch size: {self.config['batch_size']}")
        self.logger.info(f"Epochs: {self.config['num_epochs']}")
        self.logger.info(f"Mixed precision: {self.scaler is not None}")
        
        self.start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler (after optimizer.step())
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Track metrics
            epoch_time = time.time() - epoch_start
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['epoch_times'].append(epoch_time)
            self.metrics['learning_rates'].append(current_lr)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
            
            # Calculate progress stats
            total_elapsed = time.time() - self.start_time
            avg_epoch_time = total_elapsed / (epoch + 1)
            remaining_epochs = self.config['num_epochs'] - epoch - 1
            eta = remaining_epochs * avg_epoch_time
            
            # Log progress
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f} {'*BEST*' if is_best else ''}")
            self.logger.info(f"  Learning Rate: {current_lr:.6f}")
            self.logger.info(f"  Epoch Time: {epoch_time:.1f}s")
            self.logger.info(f"  Total Time: {total_elapsed/3600:.2f}h")
            self.logger.info(f"  ETA: {eta/3600:.2f}h")
            
            # Report cache stats
            if hasattr(self.train_loader.dataset.dataset, 'cache_hits'):
                dataset = self.train_loader.dataset.dataset
                if dataset.total_accesses > 0:
                    hit_rate = dataset.cache_hits / dataset.total_accesses
                    self.logger.info(f"  Cache hit rate: {hit_rate:.1%}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time,
                })
            
            # Save metrics to file
            with open(self.output_dir / 'metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        # Training complete
        total_time = time.time() - self.start_time
        self.logger.info("="*60)
        self.logger.info("Training Complete!")
        self.logger.info(f"Total Time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best Val Loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Model saved to: {self.output_dir}")
        self.logger.info("="*60)


def main():
    """Main training script."""
    
    # Configuration
    config = {
        # Data
        'surface_zarr': 'data/era5/zarr/era5_surface_2010_2020.zarr',
        'pressure_zarr': 'data/era5/zarr/era5_pressure_2010_2019.zarr',
        'val_split': 0.1,
        
        # Model
        'model_config': 'configs/model.yaml',
        
        # Training - optimized for cloud GPU with NaN fixes
        'batch_size': 64,  # Larger batch for GPU utilization
        'num_epochs': 100,   # Full training
        'learning_rate': 1e-4,  # Much lower to prevent NaN
        'weight_decay': 1e-5,  # Lower weight decay
        'temperature': 0.1,  # Higher temperature for stability
        'mixed_precision': False,  # Disabled until NaN fixed
        'num_workers': 4,  # Better GPU utilization
        'pin_memory': True,
        'gradient_clip': 1.0,  # Add gradient clipping
        'prefetch_factor': 4,  # Better data loading
        
        # Logging
        'save_every': 10,
        'use_wandb': False,  # Set to True if you have wandb
        'output_dir': f'outputs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Loading datasets...")
    dataset = WeatherDataset(
        config['surface_zarr'],
        config['pressure_zarr']
    )
    
    # Split into train/val
    n_samples = len(dataset)
    n_val = int(n_samples * config['val_split'])
    n_train = n_samples - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")
    
    # Create dataloaders with better performance
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True if config['num_workers'] > 0 else False,
        prefetch_factor=config.get('prefetch_factor', 2) if config['num_workers'] > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True if config['num_workers'] > 0 else False,
        prefetch_factor=config.get('prefetch_factor', 2) if config['num_workers'] > 0 else None
    )
    
    # Initialize model
    print("Initializing model...")
    model = WeatherCNNEncoder(config_path=config['model_config'])
    
    # Compile model for speed (PyTorch 2.0+) - DISABLED FOR NOW
    # try:
    #     if hasattr(torch, 'compile'):
    #         print("Compiling model with torch.compile()...")
    #         model = torch.compile(model)
    # except Exception as e:
    #     print(f"Compile failed ({e}), using eager mode")
    print("Using eager mode (torch.compile disabled)")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir
    )
    
    # Start training
    trainer.train()
    
    print("\nTraining complete! Check outputs in:", output_dir)


if __name__ == "__main__":
    main()