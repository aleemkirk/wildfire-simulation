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
        Generate all 30 feature channels with VERY CONSERVATIVE fire spread.

        Settings create difficult conditions for fire spread:
        - VPD (Vapor Pressure Deficit): 0.3-1.2 kPa (LOW - very difficult to spread)
        - Temperature: 18-24°C (cool day)
        - Humidity: 65-85% (high moisture)
        - Wind Speed: 1-3 m/s (very light winds)
        - Precipitation: 3-8 mm (recent rain)
        - Soil Moisture: 30-45% (moist soil)

        Fire should spread VERY slowly, 1 cell at a time if at all.
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
        lc_pattern = terrain_base + np.random.randn(gs, gs) * 0.1
        self.lc_forest = ((lc_pattern > 0.4) & (lc_pattern < 0.7)).astype(float)
        self.lc_shrubland = ((lc_pattern > 0.3) & (lc_pattern <= 0.4)).astype(float)
        self.lc_grassland = ((lc_pattern > 0.7)).astype(float)
        self.lc_agriculture = ((lc_pattern > 0.15) & (lc_pattern <= 0.3)).astype(float)
        self.lc_settlement = ((lc_pattern <= 0.15)).astype(float)

        # 11. Temperature (°C) - COOL for very slow spread
        # Set to 18-24°C (cool day)
        self.temperature = 21 - self.dem / 400 + np.random.randn(gs, gs) * 1.5
        self.temperature = np.clip(self.temperature, 18, 26)

        # 12. Dewpoint (°C) - Very high dewpoint (humid)
        self.dewpoint = self.temperature - 4 + np.random.randn(gs, gs) * 1.5

        # 13. Humidity (%) - HIGH for minimal spread
        # Set to 65-85% (high moisture)
        self.humidity = 75 + terrain_base * 10 + np.random.randn(gs, gs) * 4
        self.humidity = np.clip(self.humidity, 60, 90)

        # 14. Wind Speed (m/s) - VERY LOW for minimal spread
        # Set to 1-3 m/s (very light winds, almost calm)
        self.wind_speed = 2.0 + np.random.randn(gs, gs) * 0.8
        self.wind_speed = np.clip(self.wind_speed, 0.5, 4)

        # 15. Wind Direction (degrees) - consistent direction with some variance
        # Set to 45° (northeast) with moderate variance
        self.wind_direction = np.ones((gs, gs)) * 45 + np.random.randn(gs, gs) * 15
        self.wind_direction = self.wind_direction % 360

        # 16. Precipitation (mm) - Moderate precipitation (recent rain)
        self.precipitation = 3 + np.random.rand(gs, gs) * 5  # 3-8mm (wet conditions)

        # 17. Pressure (hPa)
        self.pressure = 1013 - self.dem / 10 + np.random.randn(gs, gs) * 2

        # 18. Solar Radiation (W/m²) - Low (cloudy day)
        self.solar_radiation = 450 + np.random.randn(gs, gs) * 80
        self.solar_radiation = np.clip(self.solar_radiation, 300, 650)

        # 19. NDVI (Normalized Difference Vegetation Index) - vegetation density
        self.ndvi = (self.lc_forest * 0.8 + self.lc_shrubland * 0.5 +
                    self.lc_grassland * 0.6 + self.lc_agriculture * 0.7 +
                    np.random.randn(gs, gs) * 0.05)
        self.ndvi = np.clip(self.ndvi, -0.1, 1)

        # 20. LAI (Leaf Area Index)
        self.lai = self.ndvi * 6 + np.random.randn(gs, gs) * 0.5
        self.lai = np.clip(self.lai, 0, 8)

        # 21. Soil Moisture (m³/m³) - HIGH moisture (wet soil)
        self.soil_moisture = 0.35 + (1 - terrain_base) * 0.10 + np.random.randn(gs, gs) * 0.03
        self.soil_moisture = np.clip(self.soil_moisture, 0.28, 0.50)

        # 22. LST Day (Land Surface Temperature - day)
        self.lst_day = self.temperature + 5 + np.random.randn(gs, gs) * 2

        # 23. LST Night (Land Surface Temperature - night)
        self.lst_night = self.temperature - 10 + np.random.randn(gs, gs) * 2

        # 24. Wind U (east-west component)
        self.wind_u = self.wind_speed * np.cos(np.radians(self.wind_direction))

        # 25. Wind V (north-south component)
        self.wind_v = self.wind_speed * np.sin(np.radians(self.wind_direction))

        # 26. TRI (Terrain Ruggedness Index)
        self.tri = np.abs(self.curvature) * 100
        self.tri = np.clip(self.tri, 0, 50)

        # 27. VPD (Vapor Pressure Deficit) - MOST IMPORTANT FOR FIRE SPREAD!
        # VPD = sat_vp - actual_vp (VERY LOW for minimal spread)
        sat_vp = 0.6108 * np.exp(17.27 * self.temperature / (self.temperature + 237.3))
        actual_vp = sat_vp * (self.humidity / 100)
        self.vpd = sat_vp - actual_vp
        # With temp ~21°C and humidity ~75%, VPD should be ~0.5-1.0 kPa
        # This is VERY LOW fire danger - fire should barely spread at all
        self.vpd = np.clip(self.vpd, 0.2, 1.5)  # Very low VPD - fire struggles to spread

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
