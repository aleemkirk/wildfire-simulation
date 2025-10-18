# 3D U-Net Training Guide
## Wildfire Spread Prediction

This guide explains how to train a 3D U-Net model for wildfire spread prediction using the PyTorch-based feature engineering pipeline.

---

## Overview

The training pipeline consists of:

1. **Data Loading**: `WildfireDatasetWithFeatures` - Loads NetCDF files and applies feature engineering
2. **DataLoader**: PyTorch DataLoader for batching and parallel loading
3. **Model**: 3D U-Net architecture for spatiotemporal prediction
4. **Training Loop**: Complete training with validation, checkpointing, and metrics

---

## Quick Start

### 1. Train the Model

```bash
python train_unet3d.py
```

This will:
- Load all `.nc` files from the Albania 2022 dataset
- Split into train/val/test sets (70%/15%/15%)
- Train a 3D U-Net model for 50 epochs
- Save checkpoints every 5 epochs
- Save the best model based on validation IoU
- Generate training plots and history

### 2. Monitor Training

The script provides:
- **Progress bars** with real-time loss and IoU
- **Epoch summaries** with train/val metrics
- **Learning rate scheduling** based on validation loss
- **Best model tracking** based on validation IoU

### 3. Results

All results are saved to `trained_models/unet3d/`:
```
trained_models/unet3d/
├── best_model.pt                  # Best model checkpoint
├── checkpoint_epoch_50.pt          # Final checkpoint
├── training_history.png            # Training curves
├── training_history.json           # Metrics history
└── config.json                     # Training configuration
```

---

## Detailed Usage

### Custom Configuration

Edit the `config` dictionary in `train_unet3d.py`:

```python
config = {
    # Data paths
    'data_dir': Path('data/raw/dataset_64_64_all_10days_final/2022/Albania'),

    # Data splits
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,

    # Model architecture
    'in_channels': 30,        # Auto-detected from data
    'out_channels': 1,        # Binary fire prediction
    'base_channels': 32,      # Base number of filters
    'depth': 3,               # U-Net depth (3-5 recommended)

    # Training hyperparameters
    'batch_size': 4,          # Adjust based on GPU memory
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,

    # Data loading
    'num_workers': 4,         # Parallel data loading

    # Loss function weights
    'bce_weight': 0.5,        # Binary Cross-Entropy weight
    'dice_weight': 0.5,       # Dice loss weight

    # Checkpoints
    'checkpoint_dir': Path('trained_models/unet3d'),
    'save_every': 5,          # Save checkpoint every N epochs
}
```

---

## Understanding the Pipeline

### 1. Dataset Class

**`WildfireDatasetWithFeatures`** (in `src/preprocessing/feature_engineering_pytorch.py`)

```python
from preprocessing.feature_engineering_pytorch import WildfireDatasetWithFeatures

# Create dataset
dataset = WildfireDatasetWithFeatures(
    file_paths=[Path('sample1.nc'), Path('sample2.nc'), ...],
    apply_feature_engineering=True,  # Compute derived features
    normalize=False,                  # Apply normalization (requires stats)
    augment=False                     # Apply data augmentation
)

# Access a sample
features, target, metadata = dataset[0]
# features: [T=10, C=30, H=64, W=64] - Environmental features over time
# target:   [T=10, H=64, W=64]       - Binary burned areas
# metadata: {'sample_id', 'country', 'date', 'burned_area_ha'}
```

**What it does:**
- Loads raw variables from NetCDF files
- Computes derived features (TRI, VPD, fuel moisture, etc.)
- Stacks all features into a single tensor
- Optionally normalizes and augments data

### 2. DataLoader

**PyTorch DataLoader** for batching and parallel loading:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,           # Shuffle for training
    num_workers=4,          # Parallel loading (CPU cores)
    pin_memory=True         # Faster GPU transfer
)

# Iterate through batches
for features, targets, metadata in dataloader:
    # features: [B=4, T=10, C=30, H=64, W=64]
    # targets:  [B=4, T=10, H=64, W=64]
    # Process batch...
```

### 3. Model Architecture

**3D U-Net** (in `src/models/unet3d.py`)

```python
from models.unet3d import UNet3D

model = UNet3D(
    in_channels=30,      # Number of feature channels
    out_channels=1,      # Binary prediction
    base_channels=32,    # Filters in first layer
    depth=3              # Downsampling/upsampling levels
)

# Input:  [B, C, T, H, W] = [4, 30, 10, 64, 64]
# Output: [B, 1, T, H, W] = [4, 1, 10, 64, 64]
```

**Architecture:**
```
Input [B, 30, 10, 64, 64]
    ↓
Encoder:
    Conv3D → BN → ReLU → Conv3D → BN → ReLU → MaxPool3D
    32 → 64 → 128 → 256 channels (depth=3)
    ↓
Bottleneck:
    256 channels
    ↓
Decoder:
    ConvTranspose3D → Concat(skip) → Conv3D → BN → ReLU
    256 → 128 → 64 → 32 channels
    ↓
Output Conv3D
    ↓
Output [B, 1, 10, 64, 64]
```

### 4. Loss Function

**CombinedLoss = BCE + Dice**

```python
# Binary Cross-Entropy: Pixel-wise classification loss
# Dice Loss: Overlap-based loss (better for imbalanced data)

loss = 0.5 * BCE(pred, target) + 0.5 * Dice(pred, target)
```

**Why this combination?**
- BCE: Good gradient signal for individual pixels
- Dice: Handles class imbalance (most cells don't burn)
- Together: Best of both worlds

### 5. Training Loop

```python
for epoch in range(num_epochs):
    # Training
    for features, targets, _ in train_loader:
        # Reshape: [B, T, C, H, W] → [B, C, T, H, W]
        features = features.permute(0, 2, 1, 3, 4)

        # Forward pass
        output = model(features)

        # Compute loss
        loss = criterion(output, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Validation
    for features, targets, _ in val_loader:
        with torch.no_grad():
            output = model(features)
            val_loss = criterion(output, targets)

    # Save checkpoint if best
    if val_iou > best_iou:
        save_checkpoint(model, optimizer, epoch)
```

---

## Evaluation Metrics

The training script computes:

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| **IoU** | Intersection over Union (spatial overlap) | 0-1 | Higher |
| **F1 Score** | Harmonic mean of precision/recall | 0-1 | Higher |
| **Precision** | True positives / All predicted positives | 0-1 | Higher |
| **Recall** | True positives / All actual positives | 0-1 | Higher |
| **Accuracy** | Correct predictions / Total predictions | 0-1 | Higher |

**Primary metric**: **IoU** (best for spatial segmentation tasks)

---

## Memory Considerations

### GPU Memory Requirements

For **batch_size=4**, **depth=3**, **base_channels=32**:
- Model parameters: ~2-5M parameters
- Forward pass: ~4-6 GB GPU memory
- Training (with gradients): ~8-12 GB GPU memory

### Reduce Memory Usage:

1. **Decrease batch size**: `batch_size = 2` or `1`
2. **Reduce depth**: `depth = 2` instead of `3`
3. **Reduce base channels**: `base_channels = 16` instead of `32`
4. **Use gradient checkpointing** (advanced)
5. **Mixed precision training** (advanced)

### Increase Speed:

1. **Increase num_workers**: `num_workers = 8` (if you have CPU cores)
2. **Use SSD** for data storage
3. **Enable pin_memory**: Already enabled for CUDA
4. **Larger batch size** (if GPU allows): `batch_size = 8`

---

## Loading a Trained Model

```python
import torch
from models.unet3d import UNet3D

# Create model
model = UNet3D(in_channels=30, out_channels=1)

# Load checkpoint
checkpoint = torch.load('trained_models/unet3d/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    output = model(input_features)
    predictions = torch.sigmoid(output) > 0.5  # Binary predictions
```

---

## Feature Engineering Details

### Raw Features (from NetCDF):
- **Terrain**: dem, slope, aspect, curvature (4)
- **Land Cover**: lc_forest, lc_shrubland, etc. (8)
- **Weather**: t2m, d2m, rh, wind_speed, etc. (8)
- **Vegetation**: ndvi, lai, smi, lst (4)
- **Wind**: u, v, wind_direction_sin/cos (4)
- **Other**: roads_distance, population (2)

**Total**: ~30 features

### Derived Features (optional):
- Terrain Ruggedness Index (TRI)
- Vapor Pressure Deficit (VPD)
- Fuel moisture proxies
- Vegetation stress
- Neighborhood statistics
- Temporal aggregations

Enable in dataset: `apply_feature_engineering=True`

---

## Troubleshooting

### Issue: "No .nc files found"
**Solution**: Update `data_dir` in config to correct path

### Issue: "CUDA out of memory"
**Solution**: Reduce `batch_size` or model `depth`/`base_channels`

### Issue: "Model not learning (loss not decreasing)"
**Solution**:
- Check learning rate (try 1e-4 or 1e-2)
- Verify data is loaded correctly
- Check for NaN values in features
- Ensure target values are 0/1

### Issue: "Training too slow"
**Solution**:
- Increase `num_workers`
- Use smaller `depth` or `base_channels`
- Enable mixed precision training (FP16)

---

## Next Steps

1. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, depths
2. **Data Augmentation**: Enable `augment=True` in dataset
3. **Feature Selection**: Remove less important features to speed up training
4. **Advanced Models**: Try attention mechanisms, transformers
5. **Multi-GPU Training**: Distribute training across multiple GPUs
6. **Inference Pipeline**: Create script for making predictions on new data

---

## References

- 3D U-Net Paper: [Çiçek et al., 2016](https://arxiv.org/abs/1606.06650)
- Dice Loss: [Milletari et al., 2016](https://arxiv.org/abs/1606.04797)
- PyTorch Documentation: https://pytorch.org/docs/

---

**Generated with Claude Code**
**Last Updated**: 2025-10-15
