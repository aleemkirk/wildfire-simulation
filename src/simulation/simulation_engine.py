"""
Core simulation engine for wildfire spread prediction.

Manages fire history, model inference, and state updates.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.unet3d import UNet3D
from simulation.environment import Environment


class WildfireSimulation:
    """
    Core simulation engine for wildfire spread prediction.

    Manages:
    - Environmental state (64x64 grid with 30 features)
    - Fire history (10 timesteps rolling buffer)
    - Model inference
    - State updates
    """

    def __init__(self, model_path, stats_path, grid_size=64, device='mps'):
        """
        Initialize simulation engine.

        Args:
            model_path: Path to trained 3D U-Net model
            stats_path: Path to normalization statistics JSON
            grid_size: Size of the grid (default 64)
            device: Device for model inference ('mps', 'cuda', or 'cpu')
        """
        self.grid_size = grid_size
        self.device = torch.device(device if torch.backends.mps.is_available() or
                                  torch.cuda.is_available() else 'cpu')

        print(f"Initializing simulation on device: {self.device}")

        # Load model
        self._load_model(model_path)

        # Load normalization statistics
        self._load_normalization_stats(stats_path)

        # Initialize environment
        self.environment = Environment(grid_size=grid_size)

        # Initialize fire history buffer (10 timesteps)
        self.fire_history = torch.zeros(10, grid_size, grid_size)

        # Initialize ignition points
        self.ignition_points = torch.zeros(grid_size, grid_size)

        # Simulation state
        self.timestep = 0
        self.is_running = False
        self.current_fire_prob = np.zeros((grid_size, grid_size))
        self.burned_area = 0.0

        print("✓ Simulation initialized successfully")

    def _load_model(self, model_path):
        """Load trained 3D U-Net model."""
        print(f"Loading model from {model_path}...")

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = UNet3D(
            in_channels=30,
            out_channels=1,
            base_channels=32,
            depth=3
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model loaded (epoch {checkpoint['epoch']}, IoU: {checkpoint['metrics']['iou']:.4f})")

    def _load_normalization_stats(self, stats_path):
        """Load normalization statistics for feature scaling."""
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        self.mean = torch.tensor(stats['mean'])
        self.std = torch.tensor(stats['std'])

        print(f"✓ Loaded normalization stats ({stats['num_channels']} channels)")

    def ignite_fire(self, x, y, strength=500.0):
        """
        Start a fire at grid position (x, y).

        Args:
            x: X coordinate (0 to grid_size-1)
            y: Y coordinate (0 to grid_size-1)
            strength: Ignition strength (default 500.0)
                     Training data uses values 5-825, with typical values around 500-800
        """
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # Add ignition point with strong signal (matching training data scale)
            # Training data uses values like 5, 136, 173, 672, 825
            self.ignition_points[y, x] = strength

            # Add to current fire state (most recent timestep)
            self.fire_history[-1, y, x] = 1.0

            print(f"🔥 Fire ignited at ({x}, {y}) with strength {strength}")

    def step(self):
        """
        Execute one simulation timestep.

        Returns:
            np.ndarray: Fire probability map [H, W]
        """
        # 0. Evolve dynamic environmental features over time
        self.environment.evolve_dynamic_features(self.timestep)

        # 1. Get environment features (28 channels)
        env_features = self.environment.get_feature_tensor()  # [28, H, W]

        # 2. Compute fire history cumulative sum (as done in training)
        fire_history_cumsum = self.fire_history.cumsum(dim=0)  # [10, H, W]

        # 3. Stack all features (30 channels × 10 timesteps)
        # Repeat static features across all timesteps
        features_all_timesteps = []
        for t in range(10):
            # Stack: 28 env features + 1 fire history + 1 ignition points
            timestep_features = torch.cat([
                env_features,  # [28, H, W]
                fire_history_cumsum[t:t+1],  # [1, H, W]
                self.ignition_points.unsqueeze(0),  # [1, H, W]
            ], dim=0)  # [30, H, W]

            features_all_timesteps.append(timestep_features)

        # Stack to get [T, C, H, W]
        features = torch.stack(features_all_timesteps, dim=0)  # [10, 30, H, W]

        # 4. Normalize features
        features = self._normalize_features(features)

        # 5. Prepare model input [B, C, T, H, W]
        model_input = features.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 30, 10, H, W]
        model_input = model_input.to(self.device)

        # 6. Run inference
        with torch.no_grad():
            prediction = self.model(model_input)  # [1, 1, 10, H, W]

        # 7. Get next timestep prediction (use LAST timestep - 10-step ahead)
        next_fire_logits = prediction[0, 0, -1]  # [H, W]
        next_fire_prob = torch.sigmoid(next_fire_logits).cpu()  # [H, W]
        # Threshold at 0.5% to capture low-probability spread
        # The last prediction (index -1) represents accumulated spread over 10 timesteps,
        # which creates a better continuous simulation effect
        next_fire_binary = (next_fire_prob > 0.005).float()

        # 8. Update fire history buffer (rolling window)
        self.fire_history = torch.cat([
            self.fire_history[1:],  # Remove oldest timestep
            next_fire_binary.unsqueeze(0)  # Add new prediction
        ], dim=0)

        # 9. Clear ignition points (they only apply for one timestep)
        self.ignition_points = torch.zeros(self.grid_size, self.grid_size)

        # 10. Update statistics
        self.current_fire_prob = next_fire_prob.numpy()
        self.burned_area = next_fire_binary.sum().item()
        self.timestep += 1

        return self.current_fire_prob

    def _normalize_features(self, features):
        """
        Normalize features using training statistics.

        Args:
            features: [T, C, H, W] tensor

        Returns:
            Normalized features [T, C, H, W]
        """
        # Expand dimensions to match features shape
        mean = self.mean.view(1, -1, 1, 1)  # [1, C, 1, 1]
        std = self.std.view(1, -1, 1, 1)    # [1, C, 1, 1]

        # Normalize: (x - mean) / std
        features_norm = (features - mean) / std

        return features_norm

    def reset(self):
        """Reset simulation to initial state."""
        self.fire_history = torch.zeros(10, self.grid_size, self.grid_size)
        self.ignition_points = torch.zeros(self.grid_size, self.grid_size)
        self.timestep = 0
        self.current_fire_prob = np.zeros((self.grid_size, self.grid_size))
        self.burned_area = 0.0
        self.is_running = False

        print("🔄 Simulation reset")

    def get_current_state(self):
        """
        Get current simulation state for visualization.

        Returns:
            dict with keys:
                - fire_prob: [H, W] fire probability
                - terrain: [H, W] terrain elevation
                - timestep: current timestep
                - burned_area: number of burned cells
        """
        return {
            'fire_prob': self.current_fire_prob,
            'terrain': self.environment.dem,
            'timestep': self.timestep,
            'burned_area': self.burned_area,
        }
