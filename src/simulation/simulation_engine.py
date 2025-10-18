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
from models.lstm import WildfireLSTM
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

    def __init__(self, model_path, stats_path, lstm_model_path=None, grid_size=64, device='mps'):
        """
        Initialize simulation engine.

        Args:
            model_path: Path to trained 3D U-Net model
            stats_path: Path to normalization statistics JSON
            lstm_model_path: Path to trained LSTM model (optional)
            grid_size: Size of the grid (default 64)
            device: Device for model inference ('mps', 'cuda', or 'cpu')
        """
        self.grid_size = grid_size
        self.device = torch.device(device if torch.backends.mps.is_available() or
                                  torch.cuda.is_available() else 'cpu')

        print(f"Initializing simulation on device: {self.device}")

        # Load UNET model
        self._load_model(model_path)

        # Load LSTM model if provided
        self.lstm_model = None
        if lstm_model_path is not None:
            self._load_lstm_model(lstm_model_path)

        # Load normalization statistics
        self._load_normalization_stats(stats_path)

        # Initialize environment
        self.environment = Environment(grid_size=grid_size, seed = 1996)

        # Initialize fire history buffer (10 timesteps)
        self.fire_history = torch.zeros(10, grid_size, grid_size)

        # Initialize ignition points
        self.ignition_points = torch.zeros(grid_size, grid_size)

        # Simulation state
        self.timestep = 0
        self.is_running = False
        self.current_fire_prob = np.zeros((grid_size, grid_size))
        self.current_fire_prob_lstm = np.zeros((grid_size, grid_size))
        self.burned_area = 0.0

        print("✓ Simulation initialized successfully")

    def _load_model(self, model_path):
        """Load trained 3D U-Net model."""
        print(f"Loading UNET model from {model_path}...")

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = UNet3D(
            in_channels=30,
            out_channels=1,
            base_channels=32,
            depth=3
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ UNET model loaded (epoch {checkpoint['epoch']}, IoU: {checkpoint['metrics']['iou']:.4f})")

    def _load_lstm_model(self, model_path):
        """Load trained LSTM model."""
        print(f"Loading LSTM model from {model_path}...")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Check the input channels from the model state dict
        encoder_weight_shape = checkpoint['model_state_dict']['encoder.0.weight'].shape
        input_channels = encoder_weight_shape[1]  # [out_channels, in_channels, H, W]

        self.lstm_model = WildfireLSTM(
            input_channels=input_channels,
            hidden_dims=[64, 64, 32],
            kernel_sizes=[3, 3, 3],
            output_channels=1
        ).to(self.device)

        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_model.eval()

        epoch = checkpoint.get('epoch', 'N/A')
        iou = checkpoint.get('metrics', {}).get('iou', 'N/A')
        print(f"✓ LSTM model loaded (epoch {epoch}, IoU: {iou})")

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
            np.ndarray: Fire probability map [H, W] from UNET model
        """
        # 0. Get current fire mask (burning cells from previous timestep)
        current_fire_mask = (self.fire_history[-1] > 0).numpy()  # [H, W] boolean

        # 1. Evolve dynamic environmental features over time (includes fuel consumption)
        self.environment.evolve_dynamic_features(self.timestep, fire_mask=current_fire_mask)

        # 2. Get environment features (28 channels)
        env_features = self.environment.get_feature_tensor()  # [28, H, W]

        # 3. Compute fire history cumulative sum (as done in training)
        fire_history_cumsum = self.fire_history.cumsum(dim=0)  # [10, H, W]

        # 4. Stack all features (30 channels × 10 timesteps) for UNET
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
        features_unet = torch.stack(features_all_timesteps, dim=0)  # [10, 30, H, W]

        # 5. Normalize features
        features_unet_norm = self._normalize_features(features_unet)

        # 6. Prepare UNET model input [B, C, T, H, W]
        model_input_unet = features_unet_norm.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 30, 10, H, W]
        model_input_unet = model_input_unet.to(self.device)

        # 7. Run UNET inference
        with torch.no_grad():
            prediction_unet = self.model(model_input_unet)  # [1, 1, 10, H, W]

        # 8. Get next timestep prediction from UNET (use LAST timestep - 10-step ahead)
        next_fire_logits = prediction_unet[0, 0, -1]  # [H, W]
        next_fire_prob = torch.sigmoid(next_fire_logits).cpu()  # [H, W]

        # 9. Run LSTM inference if model is available
        if self.lstm_model is not None:
            # LSTM uses the same 30-channel features as UNET
            # LSTM expects [B, T, C, H, W]
            model_input_lstm = features_unet_norm.unsqueeze(0).to(self.device)  # [1, 10, 30, H, W]

            with torch.no_grad():
                prediction_lstm = self.lstm_model(model_input_lstm)  # [1, 10, 1, H, W]

            # Get next timestep prediction from LSTM
            next_fire_logits_lstm = prediction_lstm[0, -1, 0]  # [H, W]
            next_fire_prob_lstm = torch.sigmoid(next_fire_logits_lstm).cpu()  # [H, W]
            self.current_fire_prob_lstm = next_fire_prob_lstm.numpy()

        # 10. Threshold at 1% - balanced to allow spread while preventing indefinite burning
        # The last prediction (index -1) represents accumulated spread over 10 timesteps,
        # which creates a better continuous simulation effect
        next_fire_binary = (next_fire_prob > 0.01).float()

        # 11. Update fire history buffer (rolling window)
        self.fire_history = torch.cat([
            self.fire_history[1:],  # Remove oldest timestep
            next_fire_binary.unsqueeze(0)  # Add new prediction
        ], dim=0)

        # 12. Clear ignition points (they only apply for one timestep)
        self.ignition_points = torch.zeros(self.grid_size, self.grid_size)

        # 13. Update statistics
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
        self.current_fire_prob_lstm = np.zeros((self.grid_size, self.grid_size))
        self.burned_area = 0.0
        self.is_running = False

        print("🔄 Simulation reset")

    def get_current_state(self):
        """
        Get current simulation state for visualization.

        Returns:
            dict with keys:
                - fire_prob: [H, W] fire probability (UNET)
                - fire_prob_lstm: [H, W] fire probability (LSTM)
                - terrain: [H, W] terrain elevation
                - timestep: current timestep
                - burned_area: number of burned cells
                - has_lstm: whether LSTM model is available
        """
        return {
            'fire_prob': self.current_fire_prob,
            'fire_prob_lstm': self.current_fire_prob_lstm,
            'terrain': self.environment.dem,
            'timestep': self.timestep,
            'burned_area': self.burned_area,
            'has_lstm': self.lstm_model is not None,
        }
