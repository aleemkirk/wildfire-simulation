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

    def _generate_features(self):
        """
        Generate all 30 feature channels MATCHING TRAINING DATA RANGES.

        IMPORTANT: The training data has non-standard value ranges.
        We must match these ranges exactly for the model to work!

        Based on actual training data analysis:
        - Temperature: 0-482 (NOT Kelvin!)
        - Humidity: 0-296 (NOT fraction!)
        - NDVI: 0-15M (corrupted scaling)
        - VPD: All zeros
        - Pressure: All zeros

        Using HIGHER temperature, LOWER humidity for faster fire spread.
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

        # 11. Temperature - Match training data range: 0-482 (EXTREME HIGH for fire spread)
        # Use MAXIMUM values (350-482) for extreme fire conditions
        self.temperature = 400 + self.dem / 5 + np.random.randn(gs, gs) * 30
        self.temperature = np.clip(self.temperature, 350, 482)

        # 12. Dewpoint - Match training data range: 0-306 (VERY LOW for extreme dryness)
        # Use MINIMUM values (0-80) to indicate extremely dry conditions
        self.dewpoint = 40 + terrain_base * 20 + np.random.randn(gs, gs) * 15
        self.dewpoint = np.clip(self.dewpoint, 0, 100)

        # 13. Humidity - Match training data range: 0-296 (VERY LOW for extreme dryness)
        # Use MINIMUM values (0-60) to indicate extremely dry conditions
        self.humidity = 30 + terrain_base * 15 + np.random.randn(gs, gs) * 10
        self.humidity = np.clip(self.humidity, 0, 80)

        # 14. Wind Speed (m/s) - VERY HIGH for rapid fire spread
        # Set to 8-15 m/s (strong winds)
        self.wind_speed = 10.0 + np.random.randn(gs, gs) * 2.5
        self.wind_speed = np.clip(self.wind_speed, 6, 18)

        # 15. Wind Direction (degrees) - consistent direction with some variance
        # Set to 45° (northeast) with moderate variance
        self.wind_direction = np.ones((gs, gs)) * 45 + np.random.randn(gs, gs) * 15
        self.wind_direction = self.wind_direction % 360

        # 16. Precipitation - Match training data range: 0-294 (MINIMAL for extreme drought)
        # Use MINIMUM values (0-20) to indicate severe drought
        self.precipitation = 5 + np.random.rand(gs, gs) * 10
        self.precipitation = np.clip(self.precipitation, 0, 25)

        # 17. Pressure - Training data shows ALL ZEROS (probably masked or corrupted)
        self.pressure = np.zeros((gs, gs))

        # 18. Solar Radiation - Match training data range: 0-101017 (LOWER values for slow spread)
        # Use LOWER end (40000-80000) to indicate less intense sun
        self.solar_radiation = 60_000 + np.random.randn(gs, gs) * 15_000
        self.solar_radiation = np.clip(self.solar_radiation, 30_000, 95_000)

        # 19. NDVI - Match training data range: 0-15,104,254 (VERY HIGH - corrupted scaling)
        # Use MAXIMUM values (10M-15M) for dense vegetation
        self.ndvi = (self.lc_forest * 0.9 + self.lc_shrubland * 0.7 +
                    self.lc_grassland * 0.8 + self.lc_agriculture * 0.85 +
                    np.random.randn(gs, gs) * 0.05)
        # Scale to match training data's astronomical values - MAXIMUM vegetation
        self.ndvi = self.ndvi * 18_000_000
        self.ndvi = np.clip(self.ndvi, 10_000_000, 15_100_000)

        # 20. LAI (Leaf Area Index) - Match training data range: -0.18-0.90
        # Use MAXIMUM values (0.6-0.9) for dense leaf coverage
        self.lai = 0.75 + terrain_base * 0.1 + np.random.randn(gs, gs) * 0.05
        self.lai = np.clip(self.lai, 0.6, 0.9)

        # 21. Soil Moisture - Match training data range: 0-6.80 (MINIMAL for extreme dryness)
        # Use MINIMUM values (0-0.5) to indicate bone-dry soil
        self.soil_moisture = 0.2 + (1 - terrain_base) * 0.15 + np.random.randn(gs, gs) * 0.1
        self.soil_moisture = np.clip(self.soil_moisture, 0.0, 0.6)

        # 22. LST Day (Land Surface Temperature - day) - Match training data range: 0-0.79
        # Use values around 0.3-0.4 (moderate surface temp)
        self.lst_day = 0.35 + terrain_base * 0.08 + np.random.randn(gs, gs) * 0.05
        self.lst_day = np.clip(self.lst_day, 0.2, 0.6)

        # 23. LST Night - Match training data range: 0-320 (HIGHER values less extreme)
        # Use values around 200-290 to match training pattern
        self.lst_night = 240 + terrain_base * 30 + np.random.randn(gs, gs) * 20
        self.lst_night = np.clip(self.lst_night, 180, 310)

        # 24. Wind U - Match training data range: 0-300 (moderate values)
        # Use values around 100-200 to match training pattern
        self.wind_u = 150 + np.random.randn(gs, gs) * 40
        self.wind_u = np.clip(self.wind_u, 80, 250)

        # 25. Wind V - Match training data range: -1.62 to 1.25 (small values)
        # Keep near zero (light winds)
        self.wind_v = np.random.randn(gs, gs) * 0.5
        self.wind_v = np.clip(self.wind_v, -1.2, 1.0)

        # 26. TRI - Match training data range: -3.35 to 1.44 (mostly negative)
        # Use mostly negative values to match training pattern
        self.tri = -2.0 + np.random.randn(gs, gs) * 0.8
        self.tri = np.clip(self.tri, -3.2, 1.2)

        # 27. VPD (Vapor Pressure Deficit) - Training data shows ALL ZEROS!
        # This is critical - the model was trained with VPD = 0 everywhere
        self.vpd = np.zeros((gs, gs))

        # 28. Fire History - initialized as zeros (10 timesteps)
        # Will be managed by simulation engine

        # 29. Ignition Points - initialized as zeros
        # Will be set by user clicks

    def get_feature_tensor(self, timestep=0):
        """
        Get all features as a tensor (except fire history and ignition points).

        Returns:
            torch.Tensor: [28, H, W] tensor of features
        """
        features = np.stack([
            self.dem,
            self.slope,
            self.aspect,
            self.curvature,
            self.roads_distance,
            self.population,
            self.lc_forest,
            self.lc_shrubland,
            self.lc_grassland,
            self.lc_agriculture,
            self.lc_settlement,
            self.temperature,
            self.dewpoint,
            self.humidity,
            self.wind_speed,
            self.wind_direction,
            self.precipitation,
            self.pressure,
            self.solar_radiation,
            self.ndvi,
            self.lai,
            self.soil_moisture,
            self.lst_day,
            self.lst_night,
            self.wind_u,
            self.wind_v,
            self.tri,
            self.vpd,
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
