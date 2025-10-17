#!/usr/bin/env python3
"""
CNN Encoder with FiLM Conditioning for Adelaide Weather Forecasting
Implements deep learning encoder for weather pattern embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import math
from pathlib import Path


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.scale = nn.Linear(condition_dim, feature_dim)
        self.shift = nn.Linear(condition_dim, feature_dim)
        
    def forward(self, x, condition):
        """
        Apply FiLM conditioning.
        x: (B, C, H, W) feature maps
        condition: (B, condition_dim) conditioning vector
        """
        # Get scale and shift parameters
        gamma = self.scale(condition).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.shift(condition).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        
        # Apply FiLM: x_out = gamma * x + beta
        return gamma * x + beta


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    
    def __init__(self, in_channels, out_channels, dilation_rates):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        # 1x1 convolution
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Dilated convolutions
        for rate in dilation_rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        total_channels = out_channels * (len(dilation_rates) + 2)  # +2 for 1x1 and global pool
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        results = []
        
        # Apply all dilated convolutions
        for conv in self.convs:
            results.append(conv(x))
        
        # Global average pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        results.append(global_feat)
        
        # Concatenate and project
        out = torch.cat(results, dim=1)
        out = self.project(out)
        
        return out


class LeadTimeEmbedding(nn.Module):
    """Sinusoidal position encoding for lead time."""
    
    def __init__(self, embedding_dim, max_lead_time=72):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_lead_time = max_lead_time
        
        # Create sinusoidal embeddings
        position = torch.arange(0, max_lead_time + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-math.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(max_lead_time + 1, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, lead_times):
        """
        lead_times: (B,) tensor of lead times in hours
        """
        # Clamp to valid range
        lead_times = torch.clamp(lead_times, 0, self.max_lead_time).long()
        return self.pe[lead_times]


class SeasonalEmbedding(nn.Module):
    """Learnable embeddings for seasonal information."""
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.month_embed = nn.Embedding(12, embedding_dim // 2)
        self.hour_embed = nn.Embedding(24, embedding_dim // 2)
    
    def forward(self, months, hours):
        """
        months: (B,) tensor of months (0-11)
        hours: (B,) tensor of hours (0-23)
        """
        month_emb = self.month_embed(months)
        hour_emb = self.hour_embed(hours)
        return torch.cat([month_emb, hour_emb], dim=1)


class CNNEncoderStage(nn.Module):
    """Single stage of CNN encoder with optional FiLM conditioning."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, 
                 use_film=False, condition_dim=None):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_film = use_film
        if use_film and condition_dim is not None:
            self.film = FiLMLayer(out_channels, condition_dim)
    
    def forward(self, x, condition=None):
        x = self.conv(x)
        x = self.bn(x)
        
        if self.use_film and condition is not None:
            x = self.film(x, condition)
        
        x = self.relu(x)
        return x


class WeatherCNNEncoder(nn.Module):
    """CNN encoder for weather pattern embeddings with FiLM conditioning."""
    
    def __init__(self, config_path='../configs/model.yaml'):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        encoder_config = config['encoder']
        film_config = config['film']
        
        self.input_shape = encoder_config['input_shape']  # [H, W, C]
        self.embedding_dim = encoder_config['embedding_dim']
        
        # Input dimensions: 9 variables (z500, mslp, t850, u850, v850, q850, u10, v10, t2m)
        input_channels = self.input_shape[2]
        
        # Stage configurations
        stages = encoder_config['architecture']
        
        # Build conditioning embeddings
        lead_embed_dim = film_config['lead_time_embedding']['embedding_dim']
        seasonal_embed_dim = film_config['seasonal_embedding']['embedding_dim']
        
        self.lead_time_embedding = LeadTimeEmbedding(
            lead_embed_dim, 
            film_config['lead_time_embedding']['max_lead']
        )
        
        self.seasonal_embedding = SeasonalEmbedding(seasonal_embed_dim)
        
        # Total conditioning dimension
        self.condition_dim = lead_embed_dim + seasonal_embed_dim
        
        # CNN stages
        self.stages = nn.ModuleList()
        film_layers = film_config['film_layers']
        
        current_channels = input_channels
        
        for i, (stage_name, stage_config) in enumerate(stages.items(), 1):
            out_channels = stage_config['filters']
            kernel_size = stage_config['kernel_size']
            stride = stage_config['stride']
            
            use_film = i in film_layers
            
            stage = CNNEncoderStage(
                current_channels, out_channels, kernel_size, stride,
                use_film=use_film, condition_dim=self.condition_dim if use_film else None
            )
            
            self.stages.append(stage)
            current_channels = out_channels
        
        # ASPP module
        aspp_config = encoder_config['aspp']
        self.aspp = ASPPModule(
            current_channels, 
            aspp_config['filters'],
            aspp_config['dilation_rates']
        )
        
        # Global context and final projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Calculate final feature size after ASPP
        aspp_out_channels = aspp_config['filters']
        
        self.final_projection = nn.Sequential(
            nn.Linear(aspp_out_channels, self.embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )
        
        # L2 normalization for embeddings
        self.normalize = nn.functional.normalize
    
    def forward(self, x, lead_times, months, hours):
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) weather data
            lead_times: (B,) lead times in hours
            months: (B,) months (0-11)
            hours: (B,) hours (0-23)
        
        Returns:
            embeddings: (B, embedding_dim) normalized embeddings
        """
        batch_size = x.size(0)
        
        # Create conditioning vector
        lead_emb = self.lead_time_embedding(lead_times)
        seasonal_emb = self.seasonal_embedding(months, hours)
        condition = torch.cat([lead_emb, seasonal_emb], dim=1)
        
        # Pass through CNN stages
        for i, stage in enumerate(self.stages):
            if hasattr(stage, 'use_film') and stage.use_film:
                x = stage(x, condition)
            else:
                x = stage(x)
        
        # ASPP module
        x = self.aspp(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.view(batch_size, -1)  # (B, C)
        
        # Final projection
        embeddings = self.final_projection(x)
        
        # L2 normalize embeddings
        embeddings = self.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding_dim(self):
        """Get embedding dimension."""
        return self.embedding_dim


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for weather pattern learning."""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, positive_indices=None):
        """
        Compute InfoNCE loss.
        
        Args:
            embeddings: (B, D) normalized embeddings
            positive_indices: (B,) indices of positive pairs (optional)
        
        Returns:
            loss: scalar InfoNCE loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Compute pairwise similarities
        similarities = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create labels - each sample is positive with itself
        if positive_indices is None:
            labels = torch.arange(batch_size, device=device)
        else:
            labels = positive_indices
        
        # Mask out self-similarities
        mask = torch.eye(batch_size, device=device).bool()
        similarities.masked_fill_(mask, float('-inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarities, labels)
        
        return loss


def create_model(config_path='../configs/model.yaml'):
    """Create CNN encoder model from configuration."""
    
    model = WeatherCNNEncoder(config_path)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return model


def demo_cnn_encoder():
    """Demonstrate CNN encoder functionality."""
    
    print("Adelaide Weather Forecasting - CNN Encoder Demo")
    
    # Create model
    model = create_model()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Embedding dimension: {model.get_embedding_dim()}")
    
    # Create dummy input (batch_size=4, channels=9, height=101, width=101)
    batch_size = 4
    x = torch.randn(batch_size, 9, 101, 101)
    lead_times = torch.tensor([6, 12, 24, 48])
    months = torch.tensor([0, 3, 6, 9])  # Jan, Apr, Jul, Oct
    hours = torch.tensor([0, 6, 12, 18])
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(x, lead_times, months, hours)
    
    print(f"Input shape: {x.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Embedding norms: {torch.norm(embeddings, dim=1)}")
    
    # Test InfoNCE loss
    loss_fn = InfoNCELoss()
    loss = loss_fn(embeddings)
    print(f"InfoNCE loss: {loss.item():.4f}")
    
    return model


if __name__ == "__main__":
    demo_cnn_encoder()