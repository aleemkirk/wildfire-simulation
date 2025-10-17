"""
Environment management for wildfire simulation.

Generates and manages the 64x64 grid with all 30 feature channels.
"""

import torch
import numpy as np
from pathlib import Path


class Environment:
    """
    Manages environmental data for the 64x64 grid.

    Features (30 channels):
    - Static: DEM, slope, aspect, curvature, roads, population, land cover (7)
    - Terrain: TRI (1)
    - Weather: Temperature, humidity, wind, pressure, etc. (11)
    - Vegetation: NDVI, LAI, soil moisture, LST, VPD (9)
    - Fire: Fire history, ignition points (2)
    """

    def __init__(self, grid_size=64, seed=42):
        """
        Initialize environment with random but realistic values.

        Args:
            grid_size: Size of the grid (default 64x64)
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate all 30 features
        self._generate_features()

    def _compute_tri_like_training(self, dem: torch.Tensor) -> torch.Tensor:
        """
        Compute TRI exactly like the training pipeline does.

        Training TRI has range 0-482, which suggests it's computed differently
        than the standard TRI formula. This matches the FeatureEngineering class.
        """
        import torch.nn.functional as F

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

    def _generate_features(self):
        """
        Generate all 30 feature channels with EXTREME FIRE SPREAD CONDITIONS.

        ⚠️  EXTREME FIRE DANGER - ALL VALUES AT MAXIMUM (within training ranges)

        This represents CATASTROPHIC fire conditions:
        - Temperature: 32-34°C (HOT!)
        - Humidity: 5-15% (BONE DRY!)
        - Wind: 4-5.2 m/s (STRONG!)
        - Vegetation: NDVI 0.75-0.90 (MAXIMUM FUEL!)
        - LAI: 5.5-6.8 (DENSE FOLIAGE!)
        - Soil moisture: 5-15% (PARCHED!)
        - Solar radiation: 14-15.8 MJ/m² (INTENSE SUN!)
        - LST Day: 37-47°C (SCORCHING SURFACE!)

        All values stay within training data ranges for valid model predictions.
        """
        gs = self.grid_size

        # Create a base for correlated features using Perlin-like noise
        x = np.linspace(0, 4 * np.pi, gs)
        y = np.linspace(0, 4 * np.pi, gs)
        X, Y = np.meshgrid(x, y)

        # Base terrain pattern
        terrain_base = (np.sin(X) + np.cos(Y) +
                       0.5 * np.sin(2*X) + 0.5 * np.cos(2*Y))
        terrain_base = (terrain_base - terrain_base.min()) / (terrain_base.max() - terrain_base.min())

        # 0. DEM (Digital Elevation Model) - 0 to 2000m
        self.dem = terrain_base * 1500 + 200 + np.random.randn(gs, gs) * 50
        self.dem = np.clip(self.dem, 0, 2000)

        # 1. Slope (degrees) - derived from DEM
        dy, dx = np.gradient(self.dem)
        self.slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2) / 30))  # 30m resolution
        self.slope = np.clip(self.slope, 0, 45)

        # 2. Aspect (degrees) - direction of slope
        self.aspect = np.degrees(np.arctan2(dy, dx)) % 360

        # 3. Curvature - second derivative of elevation
        dyy, dyx = np.gradient(dy)
        dxy, dxx = np.gradient(dx)
        self.curvature = (dxx + dyy) / 1000  # Normalized

        # 4. Roads Distance (km) - higher in remote areas
        roads_centers = np.random.rand(5, 2) * gs
        self.roads_distance = np.min([
            np.sqrt((X - c[0])**2 + (Y - c[1])**2)
            for c in roads_centers
        ], axis=0) * 0.5  # Scale to km

        # 5. Population (people/km²) - lower in mountainous areas
        self.population = np.exp(-terrain_base * 3) * 500 + np.random.rand(gs, gs) * 50

        # 6-10. Land Cover (5 classes: Forest, Shrubland, Grassland, Agriculture, Settlement)
        # INCREASE vegetation cover for more fuel - mostly forest and shrubland
        lc_pattern = terrain_base + np.random.randn(gs, gs) * 0.1
        self.lc_forest = ((lc_pattern > 0.2) & (lc_pattern < 0.75)).astype(float)  # MORE FOREST
        self.lc_shrubland = ((lc_pattern > 0.1) & (lc_pattern <= 0.2)).astype(float)  # MORE SHRUBLAND
        self.lc_grassland = ((lc_pattern > 0.75)).astype(float)  # LOTS OF GRASSLAND
        self.lc_agriculture = ((lc_pattern > 0.05) & (lc_pattern <= 0.1)).astype(float)
        self.lc_settlement = ((lc_pattern <= 0.05)).astype(float)  # MINIMAL SETTLEMENT

        # ========================================================================
        # WEATHER FEATURES - EXTREME FIRE CONDITIONS (within training ranges)
        # ========================================================================

        # Temperature (t2m) - Training range: 0-307 K
        # MAXIMUM fire danger: use UPPER limit (305-307K = 32-34°C) - HOT!
        self.temperature = 300 + terrain_base * 5 + np.random.randn(gs, gs) * 2
        self.temperature = np.clip(self.temperature, 298, 307)

        # Dewpoint (d2m) - Training range: 0-297 K
        # For EXTREME dry conditions: dewpoint MUCH lower than temperature (large depression)
        self.dewpoint = self.temperature - 40 + np.random.randn(gs, gs) * 3
        self.dewpoint = np.clip(self.dewpoint, 250, 270)

        # Relative Humidity (rh) - Training range: 0-0.62 (FRACTION not %)
        # EXTREME fire danger: VERY low humidity (0.05-0.15 = 5-15%)
        self.humidity = 0.08 + terrain_base * 0.05 + np.random.randn(gs, gs) * 0.02
        self.humidity = np.clip(self.humidity, 0.05, 0.15)

        # Wind Speed - Training range: 0-5.18 m/s
        # MAXIMUM fire spread: HIGH wind (4-5.18 m/s = 15-18 km/h)
        self.wind_speed = 4.5 + np.random.randn(gs, gs) * 0.4
        self.wind_speed = np.clip(self.wind_speed, 4.0, 5.18)

        # Wind Direction - Training range: 0-352 degrees
        # Consistent direction for organized fire spread
        self.wind_direction = np.ones((gs, gs)) * 45 + np.random.randn(gs, gs) * 30
        self.wind_direction = self.wind_direction % 360

        # Precipitation (tp) - Training range: 0-0.0038 meters (almost zero)
        # For fire conditions: essentially no rain
        self.precipitation = 0.0 + np.random.rand(gs, gs) * 0.001
        self.precipitation = np.clip(self.precipitation, 0.0, 0.002)

        # Surface Pressure (sp) - Training range: 0-101212 Pascals
        # Use realistic atmospheric pressure (decreases with elevation)
        self.pressure = 101325 - self.dem * 12 + np.random.randn(gs, gs) * 500
        self.pressure = np.clip(self.pressure, 85000, 101212)

        # Solar Radiation (ssrd) - Training range: 0-15,869,024 J/m²
        # MAXIMUM solar: peak radiation (14-15.8 MJ/m²) - INTENSE sun!
        self.solar_radiation = 14_500_000 + np.random.randn(gs, gs) * 500_000
        self.solar_radiation = np.clip(self.solar_radiation, 14_000_000, 15_869_024)

        # ========================================================================
        # VEGETATION FEATURES - MAXIMUM FUEL LOAD (within training ranges)
        # ========================================================================

        # NDVI - Training range: -0.18 to 0.90 (NORMAL vegetation index!)
        # MAXIMUM fire fuel: DENSE vegetation (0.75-0.90) - lots of fuel!
        self.ndvi = (self.lc_forest * 0.90 + self.lc_shrubland * 0.85 +
                    self.lc_grassland * 0.80 + self.lc_agriculture * 0.85 +
                    np.random.randn(gs, gs) * 0.02)
        self.ndvi = np.clip(self.ndvi, 0.75, 0.90)

        # LAI (Leaf Area Index) - Training range: 0-6.8
        # MAXIMUM fire fuel: MAXIMUM LAI (5.5-6.8) - dense leaf coverage!
        self.lai = self.ndvi * 7 + 0.5 + np.random.randn(gs, gs) * 0.3
        self.lai = np.clip(self.lai, 5.5, 6.8)

        # Soil Moisture Index (smi) - Training range: 0-0.79 (fraction)
        # EXTREME fire: VERY dry soil (0.05-0.15) - bone dry!
        self.soil_moisture = 0.08 + (1 - terrain_base) * 0.05 + np.random.randn(gs, gs) * 0.02
        self.soil_moisture = np.clip(self.soil_moisture, 0.05, 0.15)

        # LST Day (Land Surface Temperature Day) - Training range: 0-320 K
        # MAXIMUM fire: HOT surface temps (310-320K = 37-47°C)
        self.lst_day = self.temperature + 15 + np.random.randn(gs, gs) * 3
        self.lst_day = np.clip(self.lst_day, 310, 320)

        # LST Night - Training range: 0-302 K
        # Even nights stay HOT (295-302K) - continuous fire conditions
        self.lst_night = self.temperature + 0 + np.random.randn(gs, gs) * 3
        self.lst_night = np.clip(self.lst_night, 295, 302)

        # ========================================================================
        # WIND COMPONENTS - Training range: u=[-1.62, 3.97], v=[-3.85, 3.67]
        # ========================================================================

        # Compute wind components from speed and direction
        wind_dir_rad = np.radians(self.wind_direction)
        self.wind_u = -self.wind_speed * np.sin(wind_dir_rad)
        self.wind_v = -self.wind_speed * np.cos(wind_dir_rad)
        self.wind_u = np.clip(self.wind_u, -1.6, 3.9)
        self.wind_v = np.clip(self.wind_v, -3.8, 3.6)

        # ========================================================================
        # DERIVED FEATURES
        # ========================================================================

        # TRI (Terrain Ruggedness Index) - Training range: 0-482 (weird but that's what it is!)
        # This is from the feature engineering - compute from DEM
        # NOTE: The training data TRI values are MUCH larger than expected
        dem_torch = torch.from_numpy(self.dem).float()
        tri_torch = self._compute_tri_like_training(dem_torch)
        self.tri = tri_torch.numpy()
        self.tri = np.clip(self.tri, 0, 482)

        # VPD (Vapor Pressure Deficit) - Training range: 0-1 (but has NaN issues)
        # For now, use zeros to match training (it seems VPD wasn't computed correctly)
        self.vpd = np.zeros((gs, gs))

        # 28. Fire History - initialized as zeros (10 timesteps)
        # Will be managed by simulation engine

        # 29. Ignition Points - initialized as zeros
        # Will be set by user clicks

    def get_feature_tensor(self, timestep=0):
        """
        Get all features as a tensor (except fire history and ignition points).

        Returns 28 channels in the EXACT order expected by the training pipeline:
        - Channels 0-11: STATIC features (dem, slope, aspect, curvature, roads, population, 5× land cover, tri)
        - Channels 12-27: DYNAMIC features (t2m, d2m, rh, wind_speed, wind_direction, tp, sp, ssrd, ndvi, lai, smi, lst_day, lst_night, u, v, vpd)

        Note: fire_history and ignition_points are channels 28-29 but managed by simulation_engine.py

        Returns:
            torch.Tensor: [28, H, W] tensor of features
        """
        # MUST match the exact order from WildfireDatasetWithFeatures._dict_to_tensor
        # Static features first (0-11), then dynamic (12-27)
        features = np.stack([
            # STATIC (channels 0-11)
            self.dem,                    # 0
            self.slope,                  # 1
            self.aspect,                 # 2
            self.curvature,              # 3
            self.roads_distance,         # 4
            self.population,             # 5
            self.lc_forest,              # 6
            self.lc_shrubland,           # 7
            self.lc_grassland,           # 8
            self.lc_agriculture,         # 9
            self.lc_settlement,          # 10
            self.tri,                    # 11 (moved here - it's STATIC in training!)

            # DYNAMIC (channels 12-27)
            self.temperature,            # 12 (t2m)
            self.dewpoint,               # 13 (d2m)
            self.humidity,               # 14 (rh)
            self.wind_speed,             # 15
            self.wind_direction,         # 16
            self.precipitation,          # 17 (tp)
            self.pressure,               # 18 (sp)
            self.solar_radiation,        # 19 (ssrd)
            self.ndvi,                   # 20
            self.lai,                    # 21
            self.soil_moisture,          # 22 (smi)
            self.lst_day,                # 23
            self.lst_night,              # 24
            self.wind_u,                 # 25 (u)
            self.wind_v,                 # 26 (v)
            self.vpd,                    # 27
            # Channels 28-29 (fire_history, ignition_points) are added by simulation_engine
        ], axis=0)

        return torch.from_numpy(features).float()

    def update_weather(self, **kwargs):
        """
        Update weather parameters dynamically.

        Args:
            temperature: Temperature in °C
            humidity: Humidity in %
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in degrees
            etc.
        """
        if 'temperature' in kwargs:
            self.temperature = np.ones((self.grid_size, self.grid_size)) * kwargs['temperature']
        if 'humidity' in kwargs:
            self.humidity = np.ones((self.grid_size, self.grid_size)) * kwargs['humidity']
        if 'wind_speed' in kwargs:
            self.wind_speed = np.ones((self.grid_size, self.grid_size)) * kwargs['wind_speed']
        if 'wind_direction' in kwargs:
            self.wind_direction = np.ones((self.grid_size, self.grid_size)) * kwargs['wind_direction']
            # Update wind components
            self.wind_u = self.wind_speed * np.cos(np.radians(self.wind_direction))
            self.wind_v = self.wind_speed * np.sin(np.radians(self.wind_direction))
