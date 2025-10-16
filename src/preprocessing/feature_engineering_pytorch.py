"""
PyTorch-based Feature Engineering for Wildfire Spread Prediction

This module implements feature engineering using PyTorch operations for:
1. Derived terrain features
2. Weather-based features
3. Vegetation indices
4. Spatial context features
5. Temporal aggregations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class FeatureEngineering:
    """
    Collection of static methods for feature engineering using PyTorch.
    Can be used both offline (preprocessing) and online (in Dataset).
    """

    # ============================================================================
    # TERRAIN FEATURES
    # ============================================================================

    @staticmethod
    def compute_slope_gradient(dem: torch.Tensor, cell_size: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute slope gradients in x and y directions.

        Args:
            dem: Digital Elevation Model [H, W]
            cell_size: Size of grid cell in meters

        Returns:
            slope_x, slope_y: Gradients in x and y directions
        """
        # Sobel operator for gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)

        # Add batch and channel dimensions
        dem_4d = dem.unsqueeze(0).unsqueeze(0)

        # Compute gradients
        grad_x = F.conv2d(dem_4d, sobel_x, padding=1) / (8 * cell_size)
        grad_y = F.conv2d(dem_4d, sobel_y, padding=1) / (8 * cell_size)

        return grad_x.squeeze(), grad_y.squeeze()

    @staticmethod
    def compute_terrain_ruggedness_index(dem: torch.Tensor) -> torch.Tensor:
        """
        Compute Terrain Ruggedness Index (TRI).
        TRI = sqrt(mean((elevation_cell - elevation_neighbors)^2))

        Args:
            dem: Digital Elevation Model [H, W]

        Returns:
            tri: Terrain Ruggedness Index [H, W]
        """
        # Create 3x3 averaging kernel
        kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0

        # Add dimensions
        dem_4d = dem.unsqueeze(0).unsqueeze(0)

        # Compute mean of neighbors
        dem_mean = F.conv2d(dem_4d, kernel, padding=1)

        # Compute squared differences
        diff_sq = (dem_4d - dem_mean) ** 2

        # Average and take sqrt
        tri = torch.sqrt(F.conv2d(diff_sq, kernel, padding=1))

        return tri.squeeze()

    @staticmethod
    def categorize_slope(slope: torch.Tensor) -> torch.Tensor:
        """
        Categorize slope into classes: flat, moderate, steep, very_steep.

        Args:
            slope: Slope values in degrees [H, W]

        Returns:
            slope_cat: Categorical slope [H, W] with values 0-3
        """
        slope_cat = torch.zeros_like(slope, dtype=torch.long)
        slope_cat[(slope >= 5) & (slope < 15)] = 1  # moderate
        slope_cat[(slope >= 15) & (slope < 30)] = 2  # steep
        slope_cat[slope >= 30] = 3  # very steep

        return slope_cat

    @staticmethod
    def aspect_to_cardinal(aspect: torch.Tensor) -> torch.Tensor:
        """
        Convert aspect (degrees) to cardinal directions (0-7).
        0: N, 1: NE, 2: E, 3: SE, 4: S, 5: SW, 6: W, 7: NW

        Args:
            aspect: Aspect in degrees [H, W]

        Returns:
            cardinal: Cardinal direction indices [H, W]
        """
        # Normalize to 0-360
        aspect_norm = aspect % 360

        # Compute bin (45 degree bins, offset by 22.5 degrees)
        cardinal = ((aspect_norm + 22.5) / 45.0).long() % 8

        return cardinal

    # ============================================================================
    # WEATHER FEATURES
    # ============================================================================

    @staticmethod
    def kelvin_to_celsius(temp_k: torch.Tensor) -> torch.Tensor:
        """Convert temperature from Kelvin to Celsius."""
        return temp_k - 273.15

    @staticmethod
    def compute_vapor_pressure_deficit(temp_k: torch.Tensor, rh: torch.Tensor) -> torch.Tensor:
        """
        Compute Vapor Pressure Deficit (VPD).
        VPD indicates atmospheric dryness (important for fire behavior).

        Args:
            temp_k: Temperature in Kelvin [T, H, W]
            rh: Relative humidity (0-1) [T, H, W]

        Returns:
            vpd: Vapor Pressure Deficit in kPa [T, H, W]
        """
        temp_c = FeatureEngineering.kelvin_to_celsius(temp_k)

        # Saturation vapor pressure (kPa) using Tetens formula
        svp = 0.6108 * torch.exp((17.27 * temp_c) / (temp_c + 237.3))

        # Actual vapor pressure
        avp = svp * rh

        # VPD
        vpd = svp - avp

        return vpd

    @staticmethod
    def compute_heat_index(temp_k: torch.Tensor, rh: torch.Tensor) -> torch.Tensor:
        """
        Compute Heat Index (feels-like temperature).

        Args:
            temp_k: Temperature in Kelvin [T, H, W]
            rh: Relative humidity (0-1) [T, H, W]

        Returns:
            hi: Heat Index in Celsius [T, H, W]
        """
        T = FeatureEngineering.kelvin_to_celsius(temp_k)
        RH = rh * 100  # Convert to percentage

        # Simplified Heat Index formula (valid for T > 27°C)
        hi = -8.784695 + 1.61139411 * T + 2.338549 * RH \
             - 0.14611605 * T * RH - 0.012308094 * T**2 \
             - 0.016424828 * RH**2 + 0.002211732 * T**2 * RH \
             + 0.00072546 * T * RH**2 - 0.000003582 * T**2 * RH**2

        # Use actual temperature if conditions don't warrant heat index
        hi = torch.where(T < 27, T, hi)

        return hi

    @staticmethod
    def compute_wind_components(wind_speed: torch.Tensor, wind_direction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert wind speed and direction to U and V components.

        Args:
            wind_speed: Wind speed in m/s [T, H, W]
            wind_direction: Wind direction in degrees [T, H, W]

        Returns:
            u, v: Wind components (east-west, north-south)
        """
        # Convert to radians
        wd_rad = wind_direction * (torch.pi / 180.0)

        # Compute components
        u = -wind_speed * torch.sin(wd_rad)  # East-west
        v = -wind_speed * torch.cos(wd_rad)  # North-south

        return u, v

    # ============================================================================
    # VEGETATION FEATURES
    # ============================================================================

    @staticmethod
    def compute_fuel_moisture_proxy(smi: torch.Tensor, rh: torch.Tensor, precip: torch.Tensor) -> torch.Tensor:
        """
        Estimate fuel moisture content.

        Args:
            smi: Soil Moisture Index [T, H, W]
            rh: Relative humidity [T, H, W]
            precip: Precipitation [T, H, W]

        Returns:
            fuel_moisture: Estimated fuel moisture [T, H, W]
        """
        # Weighted combination (weights are heuristic)
        fuel_moisture = 0.4 * smi + 0.4 * rh + 0.2 * torch.clamp(precip * 1000, 0, 1)

        return fuel_moisture

    @staticmethod
    def compute_vegetation_stress(ndvi: torch.Tensor, smi: torch.Tensor) -> torch.Tensor:
        """
        Compute vegetation stress index (higher = more stressed/dry).

        Args:
            ndvi: Normalized Difference Vegetation Index [T, H, W]
            smi: Soil Moisture Index [T, H, W]

        Returns:
            stress: Vegetation stress index [T, H, W]
        """
        # Inverse of healthy vegetation indicators
        stress = (1 - torch.clamp(ndvi, 0, 1)) * (1 - smi)

        return stress

    @staticmethod
    def compute_fuel_load(lai: torch.Tensor, lc_forest: torch.Tensor,
                          lc_shrubland: torch.Tensor, lc_grassland: torch.Tensor) -> torch.Tensor:
        """
        Estimate fuel load (amount of burnable material).

        Args:
            lai: Leaf Area Index [T, H, W]
            lc_forest: Forest cover fraction [H, W]
            lc_shrubland: Shrubland fraction [H, W]
            lc_grassland: Grassland fraction [H, W]

        Returns:
            fuel_load: Estimated fuel load [T, H, W]
        """
        # Burnable land cover
        burnable_cover = lc_forest + lc_shrubland + lc_grassland

        # Add temporal dimension to land cover if needed
        if lai.dim() == 3 and burnable_cover.dim() == 2:
            burnable_cover = burnable_cover.unsqueeze(0)

        # Fuel load is LAI weighted by burnable cover
        fuel_load = lai * burnable_cover

        return fuel_load

    # ============================================================================
    # SPATIAL CONTEXT FEATURES
    # ============================================================================

    @staticmethod
    def neighborhood_statistics(x: torch.Tensor, kernel_size: int = 3) -> Dict[str, torch.Tensor]:
        """
        Compute neighborhood statistics (mean, std, max, min) using convolution.

        Args:
            x: Input tensor [T, H, W] or [H, W]
            kernel_size: Size of neighborhood (3, 5, or 7)

        Returns:
            stats: Dictionary with 'mean', 'std', 'max', 'min'
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            squeeze_output = True
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # [T, 1, H, W]
            squeeze_output = False
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")

        # Padding
        pad = kernel_size // 2

        # Mean
        mean_kernel = torch.ones(1, 1, kernel_size, kernel_size,
                                 dtype=x.dtype, device=x.device) / (kernel_size ** 2)
        mean = F.conv2d(x, mean_kernel, padding=pad)

        # Std (using E[X^2] - E[X]^2)
        x_sq = x ** 2
        mean_sq = F.conv2d(x_sq, mean_kernel, padding=pad)
        std = torch.sqrt(torch.clamp(mean_sq - mean ** 2, min=1e-8))

        # Max (using max pooling)
        max_val = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)

        # Min (using -max(-x))
        min_val = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)

        # Squeeze if needed
        if squeeze_output:
            mean = mean.squeeze(0).squeeze(0)
            std = std.squeeze(0).squeeze(0)
            max_val = max_val.squeeze(0).squeeze(0)
            min_val = min_val.squeeze(0).squeeze(0)
        else:
            mean = mean.squeeze(1)
            std = std.squeeze(1)
            max_val = max_val.squeeze(1)
            min_val = min_val.squeeze(1)

        return {'mean': mean, 'std': std, 'max': max_val, 'min': min_val}

    @staticmethod
    def distance_to_fire(burned_areas: torch.Tensor) -> torch.Tensor:
        """
        Compute distance transform (distance to nearest burned cell).

        Args:
            burned_areas: Binary mask of burned areas [H, W]

        Returns:
            distance: Distance to nearest fire [H, W]
        """
        # This requires scipy/scikit-image, so we'll use a simple approximation
        # For exact distance transform, use scipy.ndimage.distance_transform_edt

        # Simple approximation using dilation
        device = burned_areas.device
        distance = torch.zeros_like(burned_areas, dtype=torch.float32)

        # Convert to binary
        fire_mask = (burned_areas > 0).float().unsqueeze(0).unsqueeze(0)

        # Iteratively dilate and increment distance
        kernel = torch.ones(1, 1, 3, 3, device=device)
        current_dist = 0
        remaining = 1 - fire_mask

        while remaining.sum() > 0 and current_dist < 100:  # Max distance limit
            # Dilate fire mask
            fire_mask = (F.conv2d(fire_mask, kernel, padding=1) > 0).float()

            # Update distance for newly reached cells
            newly_reached = fire_mask * remaining
            distance += newly_reached.squeeze() * current_dist

            # Update remaining
            remaining = 1 - fire_mask
            current_dist += 1

        return distance

    # ============================================================================
    # TEMPORAL FEATURES
    # ============================================================================

    @staticmethod
    def rolling_statistics(x: torch.Tensor, window_size: int = 3, dim: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute rolling statistics along temporal dimension.

        Args:
            x: Input tensor [T, H, W]
            window_size: Size of rolling window
            dim: Dimension to roll over (default 0 for time)

        Returns:
            stats: Dictionary with 'mean', 'max', 'min', 'sum'
        """
        # Unfold creates sliding windows
        x_unfolded = x.unfold(dimension=dim, size=window_size, step=1)

        # Compute statistics
        rolling_mean = x_unfolded.mean(dim=-1)
        rolling_max = x_unfolded.max(dim=-1)[0]
        rolling_min = x_unfolded.min(dim=-1)[0]
        rolling_sum = x_unfolded.sum(dim=-1)

        # Pad to match original size
        pad_size = window_size - 1

        # For 3D tensors, we need to pad along the first dimension
        # Repeat the first value pad_size times along dim 0
        if rolling_mean.dim() == 3:
            rolling_mean = torch.cat([rolling_mean[0:1].repeat(pad_size, 1, 1), rolling_mean], dim=0)
            rolling_max = torch.cat([rolling_max[0:1].repeat(pad_size, 1, 1), rolling_max], dim=0)
            rolling_min = torch.cat([rolling_min[0:1].repeat(pad_size, 1, 1), rolling_min], dim=0)
            rolling_sum = torch.cat([rolling_sum[0:1].repeat(pad_size, 1, 1), rolling_sum], dim=0)
        else:
            # For 2D or other dims, use regular padding
            rolling_mean = F.pad(rolling_mean, (0, 0, pad_size, 0), mode='replicate')
            rolling_max = F.pad(rolling_max, (0, 0, pad_size, 0), mode='replicate')
            rolling_min = F.pad(rolling_min, (0, 0, pad_size, 0), mode='replicate')
            rolling_sum = F.pad(rolling_sum, (0, 0, pad_size, 0), mode='replicate')

        return {
            'mean': rolling_mean,
            'max': rolling_max,
            'min': rolling_min,
            'sum': rolling_sum
        }

    @staticmethod
    def temporal_differences(x: torch.Tensor, order: int = 1) -> torch.Tensor:
        """
        Compute temporal differences (rate of change).

        Args:
            x: Input tensor [T, H, W]
            order: Order of difference (1 = first derivative)

        Returns:
            diff: Temporal differences [T-order, H, W]
        """
        diff = x
        for _ in range(order):
            diff = diff[1:] - diff[:-1]

        # Pad to maintain size
        # Repeat the first value order times along dim 0
        if diff.dim() == 3:
            diff = torch.cat([diff[0:1].repeat(order, 1, 1), diff], dim=0)
        else:
            diff = F.pad(diff, (0, 0, order, 0), mode='replicate')

        return diff

    # ============================================================================
    # COMPOSITE FEATURES
    # ============================================================================

    @staticmethod
    def compute_all_terrain_features(dem: torch.Tensor, slope: torch.Tensor,
                                     aspect: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all terrain-based features.

        Args:
            dem: Digital Elevation Model [H, W]
            slope: Slope in degrees [H, W]
            aspect: Aspect in degrees [H, W]

        Returns:
            features: Dictionary of terrain features
        """
        features = {
            'dem': dem,
            'slope': slope,
            'aspect': aspect,
            'tri': FeatureEngineering.compute_terrain_ruggedness_index(dem),
            'slope_cat': FeatureEngineering.categorize_slope(slope),
            'aspect_cat': FeatureEngineering.aspect_to_cardinal(aspect),
        }

        # Add neighborhood statistics for DEM
        dem_stats = FeatureEngineering.neighborhood_statistics(dem, kernel_size=5)
        features['dem_mean_5x5'] = dem_stats['mean']
        features['dem_std_5x5'] = dem_stats['std']

        return features

    @staticmethod
    def compute_all_weather_features(t2m: torch.Tensor, d2m: torch.Tensor,
                                     rh: torch.Tensor, wind_speed: torch.Tensor,
                                     wind_direction: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all weather-based features.

        Args:
            t2m: 2m temperature in Kelvin [T, H, W]
            d2m: 2m dewpoint in Kelvin [T, H, W]
            rh: Relative humidity [T, H, W]
            wind_speed: Wind speed [T, H, W]
            wind_direction: Wind direction [T, H, W]

        Returns:
            features: Dictionary of weather features
        """
        features = {
            't2m_celsius': FeatureEngineering.kelvin_to_celsius(t2m),
            'd2m_celsius': FeatureEngineering.kelvin_to_celsius(d2m),
            'vpd': FeatureEngineering.compute_vapor_pressure_deficit(t2m, rh),
            'heat_index': FeatureEngineering.compute_heat_index(t2m, rh),
        }

        # Wind components
        u, v = FeatureEngineering.compute_wind_components(wind_speed, wind_direction)
        features['wind_u'] = u
        features['wind_v'] = v

        # Temporal statistics
        temp_stats = FeatureEngineering.rolling_statistics(t2m, window_size=3)
        features['t2m_rolling_mean'] = temp_stats['mean']
        features['t2m_rolling_max'] = temp_stats['max']

        # Rate of change
        features['t2m_delta'] = FeatureEngineering.temporal_differences(t2m)

        return features


class WildfireDatasetWithFeatures(torch.utils.data.Dataset):
    """
    PyTorch Dataset that loads NetCDF files and applies feature engineering on-the-fly.
    """

    def __init__(self,
                 file_paths: List[Path],
                 apply_feature_engineering: bool = True,
                 normalize: bool = True,
                 augment: bool = False,
                 stats_path: Optional[Path] = None):
        """
        Args:
            file_paths: List of paths to NetCDF files
            apply_feature_engineering: Whether to compute derived features
            normalize: Whether to normalize features
            augment: Whether to apply data augmentation
            stats_path: Path to normalization statistics JSON file
        """
        self.file_paths = file_paths
        self.apply_feature_engineering = apply_feature_engineering
        self.normalize = normalize
        self.augment = augment

        # Load normalization statistics if provided
        self.stats = None
        if stats_path and Path(stats_path).exists():
            import json
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded normalization statistics from {stats_path}")
            print(f"  Number of channels: {self.stats['num_channels']}")
        elif normalize and stats_path:
            print(f"Warning: Normalization requested but stats file not found: {stats_path}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load sample and apply feature engineering.

        Returns:
            features: [T, C, H, W] tensor of features
            target: [T, H, W] tensor of burned areas
            metadata: dict with sample information
        """
        # Load NetCDF file
        ds = xr.open_dataset(self.file_paths[idx])

        # Extract raw features
        features_dict = self._extract_raw_features(ds)

        # Apply feature engineering
        if self.apply_feature_engineering:
            derived_features = self._compute_derived_features(features_dict)
            features_dict.update(derived_features)

        # Convert to tensors and stack
        features = self._dict_to_tensor(features_dict)
        target = torch.from_numpy(ds['burned_areas'].values).float()

        # Normalize
        if self.normalize and self.stats is not None:
            features = self._normalize(features)

        # Augment
        if self.augment:
            features, target = self._augment(features, target)

        # Metadata
        metadata = {
            'sample_id': self.file_paths[idx].stem,
            'country': ds.attrs.get('country', 'Unknown'),
            'date': ds.attrs.get('date', 'Unknown'),
            'burned_area_ha': ds.attrs.get('burned_area_ha', 0.0)
        }

        ds.close()

        return features, target, metadata

    def _extract_raw_features(self, ds: xr.Dataset) -> Dict[str, np.ndarray]:
        """Extract raw features from xarray Dataset."""
        # Static features (2D)
        static_features = ['dem', 'slope', 'aspect', 'curvature', 'roads_distance',
                          'population', 'lc_forest', 'lc_shrubland', 'lc_grassland',
                          'lc_agriculture', 'lc_settlement']

        # Dynamic features (3D)
        dynamic_features = ['t2m', 'd2m', 'rh', 'wind_speed', 'wind_direction',
                           'tp', 'sp', 'ssrd', 'ndvi', 'lai', 'smi',
                           'lst_day', 'lst_night', 'u', 'v']

        features = {}

        for var in static_features:
            if var in ds:
                features[var] = ds[var].values

        for var in dynamic_features:
            if var in ds:
                features[var] = ds[var].values

        return features

    def _compute_derived_features(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute derived features using PyTorch operations."""
        derived = {}

        # Convert to torch tensors
        dem = torch.from_numpy(features_dict.get('dem', np.zeros((64, 64)))).float()
        slope = torch.from_numpy(features_dict.get('slope', np.zeros((64, 64)))).float()
        t2m = torch.from_numpy(features_dict.get('t2m', np.zeros((10, 64, 64)))).float()
        rh = torch.from_numpy(features_dict.get('rh', np.zeros((10, 64, 64)))).float()

        # Compute features
        tri = FeatureEngineering.compute_terrain_ruggedness_index(dem)
        derived['tri'] = tri.numpy()

        vpd = FeatureEngineering.compute_vapor_pressure_deficit(t2m, rh)
        derived['vpd'] = vpd.numpy()

        # Add more as needed...

        return derived

    def _dict_to_tensor(self, features_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """Stack features into a single tensor [T, C, H, W]."""
        # Separate static and dynamic
        static = []
        dynamic = []

        for key, value in features_dict.items():
            tensor = torch.from_numpy(value).float()

            if tensor.dim() == 2:  # Static [H, W]
                static.append(tensor)
            elif tensor.dim() == 3:  # Dynamic [T, H, W]
                dynamic.append(tensor)

        # Stack static features along channel dimension
        if static:
            static_tensor = torch.stack(static, dim=0)  # [C_static, H, W]
            # Expand to time dimension
            T = dynamic[0].shape[0] if dynamic else 10
            static_tensor = static_tensor.unsqueeze(0).expand(T, -1, -1, -1)

        # Stack dynamic features
        if dynamic:
            dynamic_tensor = torch.stack(dynamic, dim=1)  # [T, C_dynamic, H, W]

        # Concatenate
        if static and dynamic:
            features = torch.cat([static_tensor, dynamic_tensor], dim=1)
        elif dynamic:
            features = dynamic_tensor
        elif static:
            features = static_tensor
        else:
            features = torch.zeros(10, 1, 64, 64)

        return features

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features using pre-computed statistics.

        Args:
            features: Feature tensor [T, C, H, W]

        Returns:
            normalized_features: Normalized tensor [T, C, H, W]
        """
        if self.stats is None:
            return features

        # features: [T, C, H, W]
        # Extract mean and std
        mean = torch.tensor(self.stats['mean'], dtype=features.dtype).view(1, -1, 1, 1)  # [1, C, 1, 1]
        std = torch.tensor(self.stats['std'], dtype=features.dtype).view(1, -1, 1, 1)    # [1, C, 1, 1]

        # Normalize: (x - mean) / std
        features_normalized = (features - mean) / (std + 1e-8)

        return features_normalized

    def _augment(self, features: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            features = torch.flip(features, dims=[-1])
            target = torch.flip(target, dims=[-1])

        # Random vertical flip
        if torch.rand(1) > 0.5:
            features = torch.flip(features, dims=[-2])
            target = torch.flip(target, dims=[-2])

        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            features = torch.rot90(features, k=k, dims=[-2, -1])
            target = torch.rot90(target, k=k, dims=[-2, -1])

        return features, target


# Example usage
if __name__ == "__main__":
    # Example: Compute terrain features
    dem = torch.randn(64, 64) * 100 + 500  # Random elevation
    slope = torch.rand(64, 64) * 45  # Random slope 0-45 degrees
    aspect = torch.rand(64, 64) * 360  # Random aspect 0-360 degrees

    terrain_features = FeatureEngineering.compute_all_terrain_features(dem, slope, aspect)

    print("Terrain features computed:")
    for key, value in terrain_features.items():
        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")

    # Example: Compute weather features
    t2m = torch.randn(10, 64, 64) * 10 + 293  # ~20°C ± 10
    d2m = t2m - 5  # Dewpoint slightly lower
    rh = torch.rand(10, 64, 64) * 0.6 + 0.2  # 20-80% humidity
    wind_speed = torch.rand(10, 64, 64) * 5  # 0-5 m/s
    wind_direction = torch.rand(10, 64, 64) * 360  # 0-360 degrees

    weather_features = FeatureEngineering.compute_all_weather_features(
        t2m, d2m, rh, wind_speed, wind_direction
    )

    print("\nWeather features computed:")
    for key, value in weather_features.items():
        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
