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
        Generate all 30 feature channels with MAXIMUM FIRE SPREAD CONDITIONS.

        ABSOLUTE MAXIMUM VALUES - EXTREME FIRE DANGER:
        All variables set to their MAXIMUM/MINIMUM extremes to create
        the most aggressive fire spread possible.

        This represents catastrophic fire conditions:
        - MAXIMUM temperature (approaching limit)
        - MINIMUM humidity (bone dry)
        - MAXIMUM wind (hurricane force)
        - ZERO precipitation (extreme drought)
        - MAXIMUM vegetation (dense fuel)
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

        # 11. Temperature - ABSOLUTE MAXIMUM (near upper limit)
        # Maximum possible values for catastrophic heat
        self.temperature = 470 + np.random.randn(gs, gs) * 8
        self.temperature = np.clip(self.temperature, 460, 482)

        # 12. Dewpoint - ABSOLUTE MINIMUM (completely dry)
        # Zero dewpoint = no moisture at all
        self.dewpoint = 0 + np.random.randn(gs, gs) * 5
        self.dewpoint = np.clip(self.dewpoint, 0, 15)

        # 13. Humidity - ABSOLUTE MINIMUM (zero moisture)
        # Completely dry air, no humidity whatsoever
        self.humidity = 0 + np.random.randn(gs, gs) * 3
        self.humidity = np.clip(self.humidity, 0, 10)

        # 14. Wind Speed - ABSOLUTE MAXIMUM (hurricane-force winds)
        # Maximum wind speed for explosive fire spread
        self.wind_speed = 25.0 + np.random.randn(gs, gs) * 5
        self.wind_speed = np.clip(self.wind_speed, 20, 35)

        # 15. Wind Direction (degrees) - consistent direction with some variance
        # Set to 45° (northeast) with moderate variance
        self.wind_direction = np.ones((gs, gs)) * 45 + np.random.randn(gs, gs) * 15
        self.wind_direction = self.wind_direction % 360

        # 16. Precipitation - ABSOLUTE ZERO (complete drought)
        # No rain whatsoever - bone dry
        self.precipitation = 0 + np.random.rand(gs, gs) * 1
        self.precipitation = np.clip(self.precipitation, 0, 2)

        # 17. Pressure - Training data shows ALL ZEROS (probably masked or corrupted)
        self.pressure = np.zeros((gs, gs))

        # 18. Solar Radiation - ABSOLUTE MAXIMUM (intense sun)
        # Maximum solar intensity for extreme heat
        self.solar_radiation = 95_000 + np.random.randn(gs, gs) * 4_000
        self.solar_radiation = np.clip(self.solar_radiation, 90_000, 101_000)

        # 19. NDVI - ABSOLUTE MAXIMUM (densest vegetation possible)
        # Maximum vegetation density for maximum fuel
        self.ndvi = (self.lc_forest * 1.0 + self.lc_shrubland * 1.0 +
                    self.lc_grassland * 1.0 + self.lc_agriculture * 1.0 +
                    np.random.randn(gs, gs) * 0.01)
        # Scale to MAXIMUM training data values
        self.ndvi = self.ndvi * 20_000_000
        self.ndvi = np.clip(self.ndvi, 14_000_000, 15_104_254)

        # 20. LAI - ABSOLUTE MAXIMUM (densest leaf coverage)
        # Maximum leaf area for maximum fuel
        self.lai = 0.88 + np.random.randn(gs, gs) * 0.015
        self.lai = np.clip(self.lai, 0.85, 0.90)

        # 21. Soil Moisture - ABSOLUTE ZERO (completely dry soil)
        # Zero soil moisture for extreme fire conditions
        self.soil_moisture = 0.0 + np.random.randn(gs, gs) * 0.02
        self.soil_moisture = np.clip(self.soil_moisture, 0.0, 0.05)

        # 22. LST Day - ABSOLUTE MAXIMUM (extreme surface heat)
        # Maximum daytime surface temperature
        self.lst_day = 0.75 + np.random.randn(gs, gs) * 0.03
        self.lst_day = np.clip(self.lst_day, 0.72, 0.79)

        # 23. LST Night - ABSOLUTE MAXIMUM (hot nights)
        # Maximum nighttime temperature for continuous fire
        self.lst_night = 310 + np.random.randn(gs, gs) * 8
        self.lst_night = np.clip(self.lst_night, 300, 320)

        # 24. Wind U - ABSOLUTE MAXIMUM (extreme easterly winds)
        # Maximum wind component for rapid fire spread
        self.wind_u = 280 + np.random.randn(gs, gs) * 15
        self.wind_u = np.clip(self.wind_u, 270, 300)

        # 25. Wind V - MAXIMUM positive (strong southerly component)
        # Strong winds in both directions
        self.wind_v = 1.1 + np.random.randn(gs, gs) * 0.1
        self.wind_v = np.clip(self.wind_v, 1.0, 1.25)

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
