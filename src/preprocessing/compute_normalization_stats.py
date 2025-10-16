"""
Compute mean and std statistics from training dataset
for feature normalization.

This script computes per-channel mean and standard deviation
statistics from the training dataset to be used for feature
normalization during training and inference.
"""

import torch
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.feature_engineering_pytorch import WildfireDatasetWithFeatures


def compute_stats(data_dir: Path, output_path: Path, max_samples: int = None):
    """
    Compute mean and std for each feature channel.

    Args:
        data_dir: Directory containing .nc files
        output_path: Path to save statistics JSON
        max_samples: Maximum number of samples to use (None = use all)
    """
    print("=" * 80)
    print("COMPUTING NORMALIZATION STATISTICS")
    print("=" * 80)

    # Load dataset
    nc_files = sorted(list(data_dir.glob('*.nc')))

    if len(nc_files) == 0:
        raise ValueError(f"No .nc files found in {data_dir}")

    print(f"\nFound {len(nc_files)} samples")

    if max_samples is not None and max_samples < len(nc_files):
        nc_files = nc_files[:max_samples]
        print(f"Using first {max_samples} samples for statistics computation")

    # Create dataset (without normalization)
    print("\nCreating dataset...")
    dataset = WildfireDatasetWithFeatures(
        file_paths=nc_files,
        apply_feature_engineering=True,
        normalize=False,  # Don't normalize yet
        augment=False     # No augmentation for stats
    )

    print(f"Dataset created with {len(dataset)} samples")

    # Get first sample to determine number of channels
    features, _, _ = dataset[0]
    num_channels = features.shape[1]  # [T, C, H, W]
    print(f"Number of feature channels: {num_channels}")

    # Accumulate statistics using Welford's online algorithm
    # This is more numerically stable than computing mean first, then variance
    print("\nComputing statistics...")

    count = 0
    mean = torch.zeros(num_channels)
    M2 = torch.zeros(num_channels)

    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        features, _, _ = dataset[idx]
        # features: [T, C, H, W]

        # Flatten spatial and temporal dimensions: [T*H*W, C]
        features_flat = features.permute(1, 0, 2, 3).reshape(num_channels, -1)  # [C, T*H*W]

        # Update statistics for each pixel
        for i in range(features_flat.shape[1]):
            pixel = features_flat[:, i]  # [C]
            count += 1
            delta = pixel - mean
            mean = mean + delta / count
            delta2 = pixel - mean
            M2 = M2 + delta * delta2

    # Compute final statistics
    variance = M2 / count
    std = torch.sqrt(variance)

    # Handle NaN and invalid values
    # Replace NaN with 0 for mean and 1 for std
    mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
    std = torch.where(torch.isnan(std), torch.ones_like(std), std)

    # Replace any zero or very small std with 1.0 to avoid division by zero
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    # Save to JSON
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'num_samples': len(dataset),
        'num_channels': num_channels,
        'num_pixels_processed': int(count)
    }

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)
    print(f"Output saved to: {output_path}")
    print(f"Samples processed: {len(dataset)}")
    print(f"Pixels processed: {count:,}")
    print(f"Feature channels: {num_channels}")
    print(f"\nMean range: [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"Std range:  [{std.min():.6f}, {std.max():.6f}]")

    # Print per-channel statistics (first 10 channels)
    print("\nPer-channel statistics (first 10 channels):")
    print("-" * 60)
    print(f"{'Channel':<10} {'Mean':<15} {'Std':<15}")
    print("-" * 60)
    for i in range(min(10, num_channels)):
        print(f"{i:<10} {mean[i]:>14.6f} {std[i]:>14.6f}")
    if num_channels > 10:
        print(f"... and {num_channels - 10} more channels")
    print("-" * 60)

    return stats


def main():
    """Main function."""
    # Configuration
    data_dir = Path('../../data/raw/dataset_64_64_all_10days_final/2022/Albania')
    output_path = Path('../../data/processed/normalization_stats.json')

    # Compute statistics on training set only (70% of data)
    # For now, we'll use all data for simplicity
    # In production, you should split the data first

    print("\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Output path: {output_path}")

    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please update the data_dir path in the script.")
        return

    # Compute statistics
    stats = compute_stats(data_dir, output_path, max_samples=None)

    print("\n" + "=" * 80)
    print("NORMALIZATION STATISTICS COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\nYou can now use these statistics in your dataset:")
    print(f"  dataset = WildfireDatasetWithFeatures(")
    print(f"      ...,")
    print(f"      normalize=True,")
    print(f"      stats_path=Path('{output_path}')")
    print(f"  )")


if __name__ == "__main__":
    main()
