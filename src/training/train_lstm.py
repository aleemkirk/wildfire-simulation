"""
Training Script for LSTM Wildfire Spread Prediction

This script demonstrates how to:
1. Load data using WildfireDatasetWithFeatures
2. Create DataLoaders for training and validation
3. Train an LSTM model (WildfireLSTM or BidirectionalWildfireLSTM)
4. Evaluate performance
5. Save model checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Import custom modules
import sys
sys.path.append('..')
from preprocessing.feature_engineering_pytorch import WildfireDatasetWithFeatures
from models.lstm import WildfireLSTM, BidirectionalWildfireLSTM


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Better than BCE for imbalanced datasets (most cells don't burn).
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, T, 1, H, W]
            target: Ground truth [B, T, H, W]

        Returns:
            dice_loss: Scalar loss value
        """
        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred)

        # Add channel dimension to target if needed
        if target.dim() == 3:
            target = target.unsqueeze(2)  # [B, T, 1, H, W]
        elif target.dim() == 4:
            target = target.unsqueeze(2)  # [B, T, 1, H, W]

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute Dice coefficient
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        # Return Dice loss
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss for better training.
    Uses positive class weighting to handle class imbalance.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float = 10.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # Penalize false negatives more by weighting positive class higher
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, T, 1, H, W]
            target: Ground truth [B, T, H, W]
        """
        # Reshape for BCE: [B, T, 1, H, W] -> [B*T*H*W, 1]
        B, T, C, H, W = pred.shape
        pred_flat = pred.view(-1, 1)
        target_flat = target.view(-1, 1)

        # Move pos_weight to same device as pred
        if self.bce.pos_weight.device != pred.device:
            self.bce.pos_weight = self.bce.pos_weight.to(pred.device)

        bce_loss = self.bce(pred_flat, target_flat)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        pred: Predictions [B, T, 1, H, W]
        target: Ground truth [B, T, H, W]
        threshold: Threshold for binary classification

    Returns:
        metrics: Dictionary of metric values
    """
    # Apply sigmoid and threshold
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    # Add channel dimension to target if needed
    if target.dim() == 3:
        target = target.unsqueeze(2)
    elif target.dim() == 4:
        target = target.unsqueeze(2)

    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    # Compute metrics
    tp = ((pred_flat == 1) & (target_flat == 1)).sum().float()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().float()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().float()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().float()

    # Avoid division by zero
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    accuracy = (tp + tn) / (tp + fp + tn + fn + epsilon)

    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + epsilon)

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item()
    }


def train_one_epoch(model: nn.Module,
                     dataloader: DataLoader,
                     criterion: nn.Module,
                     optimizer: optim.Optimizer,
                     device: torch.device,
                     epoch: int) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        metrics: Dictionary with average loss and metrics
    """
    model.train()
    running_loss = 0.0
    all_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'iou': 0.0
    }

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (features, target, metadata) in enumerate(pbar):
        # Move to device
        features = features.to(device)  # [B, T, C, H, W]
        target = target.to(device)      # [B, T, H, W]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(features)  # [B, T, 1, H, W]

        # Compute loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        batch_metrics = compute_metrics(output.detach(), target.detach())
        for key in all_metrics:
            all_metrics[key] += batch_metrics[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{batch_metrics["iou"]:.4f}'
        })

    # Average metrics
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    for key in all_metrics:
        all_metrics[key] /= num_batches

    return {'loss': avg_loss, **all_metrics}


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             epoch: int) -> Dict[str, float]:
    """
    Validate the model.

    Returns:
        metrics: Dictionary with average loss and metrics
    """
    model.eval()
    running_loss = 0.0
    all_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'iou': 0.0
    }

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for features, target, metadata in pbar:
            # Move to device
            features = features.to(device)
            target = target.to(device)

            # Forward pass
            output = model(features)

            # Compute loss
            loss = criterion(output, target)

            # Update metrics
            running_loss += loss.item()
            batch_metrics = compute_metrics(output, target)
            for key in all_metrics:
                all_metrics[key] += batch_metrics[key]

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_metrics["iou"]:.4f}'
            })

    # Average metrics
    num_batches = len(dataloader)
    avg_loss = running_loss / num_batches
    for key in all_metrics:
        all_metrics[key] /= num_batches

    return {'loss': avg_loss, **all_metrics}


def save_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    metrics: Dict[str, float],
                    checkpoint_dir: Path,
                    is_best: bool = False):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")


def plot_training_history(history: Dict[str, List[float]], save_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History - LSTM Model', fontsize=16, fontweight='bold')

    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'iou']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            ax.plot(history[train_key], label='Train', linewidth=2)
        if val_key in history:
            ax.plot(history[val_key], label='Validation', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training plot: {save_path}")
    plt.close()


def main():
    """Main training function."""
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    config = {
        # Data
        'data_dir': Path('../../data/raw/dataset_64_64_all_10days_final/2022/Albania'),
        'stats_path': Path('../../data/processed/normalization_stats.json'),
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,

        # Model
        'model_type': 'lstm',  # Options: 'lstm' or 'bidirectional_lstm'
        'in_channels': 34,  # Will be automatically determined by dataset
        'hidden_dims': [64, 64, 32],  # Hidden dimensions for each LSTM layer
        'kernel_sizes': [3, 3, 3],  # Kernel sizes for each LSTM layer
        'out_channels': 1,
        'dropout': 0.1,

        # Training
        'batch_size': 4,
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_workers': 4,

        # Loss
        'bce_weight': 0.5,
        'dice_weight': 0.5,
        'pos_weight': 60.0,  # Weight for positive class (increase if recall is low)

        # Checkpoints
        'checkpoint_dir': Path('trained_models/lstm'),
        'save_every': 5,

        # Device (supports CUDA, MPS for Apple Silicon, or CPU)
        'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    }

    print("="*80)
    print("WILDFIRE SPREAD PREDICTION - LSTM TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    device = torch.device(config['device'])
    print(f"Using device: {device}\n")

    # ========================================================================
    # PREPARE DATA
    # ========================================================================
    print("Loading dataset...")

    # Get all NetCDF files
    nc_files = sorted(list(config['data_dir'].glob('*.nc')))
    print(f"Found {len(nc_files)} samples")

    if len(nc_files) == 0:
        print("ERROR: No .nc files found!")
        print("Please check the data_dir path in the configuration.")
        return

    # Create full dataset with normalization
    full_dataset = WildfireDatasetWithFeatures(
        file_paths=nc_files,
        apply_feature_engineering=True,
        normalize=True,  # Enable normalization
        augment=False,   # No augmentation for initial splits
        stats_path=config['stats_path']
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config['train_ratio'] * total_size)
    val_size = int(config['val_ratio'] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples\n")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    # Determine number of input channels from first sample
    sample_features, _, _ = full_dataset[0]
    config['in_channels'] = sample_features.shape[1]  # [T, C, H, W]
    print(f"Input channels: {config['in_channels']}\n")

    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    print("Creating model...")

    if config['model_type'] == 'bidirectional_lstm':
        print("Using BidirectionalWildfireLSTM")
        model = BidirectionalWildfireLSTM(
            input_channels=config['in_channels'],
            hidden_dims=config['hidden_dims'],
            kernel_sizes=config['kernel_sizes'],
            output_channels=config['out_channels'],
            dropout=config['dropout']
        ).to(device)
    else:
        print("Using WildfireLSTM")
        model = WildfireLSTM(
            input_channels=config['in_channels'],
            hidden_dims=config['hidden_dims'],
            kernel_sizes=config['kernel_sizes'],
            output_channels=config['out_channels'],
            dropout=config['dropout']
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")

    # ========================================================================
    # SETUP TRAINING
    # ========================================================================
    criterion = CombinedLoss(
        bce_weight=config['bce_weight'],
        dice_weight=config['dice_weight'],
        pos_weight=config['pos_weight']
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("Starting training...\n")

    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [],
        'train_recall': [], 'train_f1': [], 'train_iou': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [], 'val_iou': []
    }

    best_val_iou = 0.0

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*80}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Update learning rate
        scheduler.step(val_metrics['loss'])

        # Update history
        for key, value in train_metrics.items():
            history[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            history[f'val_{key}'].append(value)

        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save checkpoint
        is_best = val_metrics['iou'] > best_val_iou
        if is_best:
            best_val_iou = val_metrics['iou']
            print(f"  ★ New best validation IoU: {best_val_iou:.4f}")

        if epoch % config['save_every'] == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_metrics, config['checkpoint_dir'], is_best)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

    # Save final model
    save_checkpoint(model, optimizer, config['num_epochs'], val_metrics, config['checkpoint_dir'])

    # Plot training history
    plot_path = config['checkpoint_dir'] / 'training_history.png'
    plot_training_history(history, plot_path)

    # Save history as JSON
    history_path = config['checkpoint_dir'] / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history: {history_path}")

    # Save configuration
    config_path = config['checkpoint_dir'] / 'config.json'
    # Convert Path objects to strings for JSON serialization
    config_json = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    with open(config_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    print(f"Saved configuration: {config_path}")

    print(f"\nBest validation IoU: {best_val_iou:.4f}")
    print("All results saved to:", config['checkpoint_dir'])


if __name__ == "__main__":
    main()
