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

    def __init__(self, grid_size=64, seed=None):
        """
        Initialize environment with random but realistic values.

        Args:
            grid_size: Size of the grid (default 64x64)
            seed: Random seed for reproducibility (default None = random each time)
        """
        self.grid_size = grid_size

        # If seed is None, use a random seed each time
        if seed is None:
            seed = np.random.randint(0, 1000000)

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

        # Create a base for correlated features using Perlin-like noise with MORE RANDOMNESS
        # Randomize frequency and phase for each run
        freq1 = np.random.uniform(2, 6)  # Random frequency between 2-6
        freq2 = np.random.uniform(2, 6)
        phase1 = np.random.uniform(0, 2 * np.pi)
        phase2 = np.random.uniform(0, 2 * np.pi)

        x = np.linspace(0, freq1 * np.pi, gs)
        y = np.linspace(0, freq2 * np.pi, gs)
        X, Y = np.meshgrid(x, y)

        # Multi-frequency terrain pattern with random phases and amplitudes
        amp1 = np.random.uniform(0.8, 1.2)
        amp2 = np.random.uniform(0.8, 1.2)
        amp3 = np.random.uniform(0.3, 0.7)
        amp4 = np.random.uniform(0.3, 0.7)
        amp5 = np.random.uniform(0.1, 0.3)

        terrain_base = (amp1 * np.sin(X + phase1) + amp2 * np.cos(Y + phase2) +
                       amp3 * np.sin(2*X + np.random.uniform(0, 2*np.pi)) +
                       amp4 * np.cos(2*Y + np.random.uniform(0, 2*np.pi)) +
                       amp5 * np.sin(3*X) * np.cos(3*Y) +
                       np.random.randn(gs, gs) * 0.3)  # Add significant noise
        terrain_base = (terrain_base - terrain_base.min()) / (terrain_base.max() - terrain_base.min())

        # 0. DEM (Digital Elevation Model) - 0 to 2000m
        # Add multiple scales of noise for realistic terrain variation
        base_elevation = np.random.uniform(100, 500)  # Random base elevation
        elevation_range = np.random.uniform(800, 1500)  # Random elevation range
        noise_scale = np.random.uniform(50, 150)  # Random noise intensity

        self.dem = terrain_base * elevation_range + base_elevation + np.random.randn(gs, gs) * noise_scale
        # Add some random peaks and valleys
        num_features = np.random.randint(3, 8)
        for _ in range(num_features):
            cx, cy = np.random.randint(0, gs, 2)
            radius = np.random.uniform(5, 20)
            height = np.random.uniform(-300, 500)
            dist = np.sqrt((np.arange(gs)[:, None] - cx)**2 + (np.arange(gs)[None, :] - cy)**2)
            self.dem += height * np.exp(-(dist**2) / (2 * radius**2))

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
        # Randomize number of roads and their positions
        num_roads = np.random.randint(3, 10)
        roads_centers = np.random.rand(num_roads, 2) * gs
        self.roads_distance = np.min([
            np.sqrt((X - c[0])**2 + (Y - c[1])**2)
            for c in roads_centers
        ], axis=0) * np.random.uniform(0.3, 0.8)  # Random scale factor

        # 5. Population (people/km²) - lower in mountainous areas
        self.population = np.exp(-terrain_base * 3) * 500 + np.random.rand(gs, gs) * 50

        # 6-10. Land Cover (5 classes: Forest, Shrubland, Grassland, Agriculture, Settlement)
        # INCREASE vegetation cover for more fuel - mostly forest and shrubland
        # Add MORE randomness to land cover distribution
        lc_pattern = terrain_base + np.random.randn(gs, gs) * np.random.uniform(0.15, 0.3)

        # Randomize thresholds for each land cover type
        forest_low = np.random.uniform(0.15, 0.25)
        forest_high = np.random.uniform(0.65, 0.80)
        shrub_low = np.random.uniform(0.05, 0.15)
        grass_threshold = np.random.uniform(0.70, 0.85)
        agr_low = np.random.uniform(0.02, 0.08)
        settlement_threshold = np.random.uniform(0.01, 0.06)

        self.lc_forest = ((lc_pattern > forest_low) & (lc_pattern < forest_high)).astype(float)
        self.lc_shrubland = ((lc_pattern > shrub_low) & (lc_pattern <= forest_low)).astype(float)
        self.lc_grassland = ((lc_pattern > grass_threshold)).astype(float)
        self.lc_agriculture = ((lc_pattern > agr_low) & (lc_pattern <= shrub_low)).astype(float)
        self.lc_settlement = ((lc_pattern <= settlement_threshold)).astype(float)

        # ========================================================================
        # WEATHER FEATURES - EXTREME FIRE CONDITIONS (within training ranges)
        # ========================================================================

        # Temperature (t2m) - Training range: 0-307 K
        # ABSOLUTE MAXIMUM: 307K (34°C) everywhere - HOTTEST POSSIBLE!
        self.temperature = np.ones((gs, gs)) * 307.0

        # Dewpoint (d2m) - Training range: 0-297 K
        # For EXTREME dry conditions: dewpoint MUCH lower than temperature (large depression)
        self.dewpoint = self.temperature - 40 + np.random.randn(gs, gs) * 3
        self.dewpoint = np.clip(self.dewpoint, 250, 270)

        # Relative Humidity (rh) - Training range: 0-0.62 (FRACTION not %)
        # ABSOLUTE MINIMUM: Close to 0% humidity - DRIEST POSSIBLE!
        self.humidity = np.ones((gs, gs)) * 0.01  # 1% humidity

        # Wind Speed and Direction - Create swirling wind patterns (vortices)
        # Generate 2-4 wind vortices (like atmospheric eddies or pressure systems)
        num_vortices = np.random.randint(2, 5)

        # Initialize with base prevailing wind
        base_wind_speed = np.random.uniform(4.0, 4.5)
        prevailing_direction = np.random.uniform(0, 360)

        self.wind_speed = np.ones((gs, gs)) * base_wind_speed
        self.wind_direction = np.ones((gs, gs)) * prevailing_direction

        # Add swirling vortices
        for i in range(num_vortices):
            # Random vortex center
            vortex_x = np.random.uniform(10, gs - 10)
            vortex_y = np.random.uniform(10, gs - 10)

            # Vortex strength and size
            vortex_strength = np.random.uniform(0.8, 1.5)  # Wind speed multiplier
            vortex_radius = np.random.uniform(15, 30)  # Radius of influence
            clockwise = np.random.choice([1, -1])  # Rotation direction

            # Calculate distance and angle from vortex center for each point
            for yi in range(gs):
                for xi in range(gs):
                    dx = xi - vortex_x
                    dy = yi - vortex_y
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance < vortex_radius:
                        # Vortex influence decreases with distance
                        influence = np.exp(-(distance / vortex_radius)**2)

                        # Tangential wind direction (perpendicular to radius)
                        angle_to_center = np.degrees(np.arctan2(dy, dx))
                        tangential_angle = (angle_to_center + clockwise * 90) % 360

                        # Blend prevailing wind with vortex wind
                        self.wind_direction[yi, xi] = (
                            self.wind_direction[yi, xi] * (1 - influence) +
                            tangential_angle * influence
                        ) % 360

                        # Wind speed increases near vortex center
                        speed_boost = vortex_strength * influence * 1.0
                        self.wind_speed[yi, xi] += speed_boost

        # Add elevation effect
        elevation_effect = (self.dem / 2000.0) * 0.3
        self.wind_speed += elevation_effect
        self.wind_speed = np.clip(self.wind_speed, 3.5, 5.18)

        # Precipitation (tp) - Training range: 0-0.0038 meters (almost zero)
        # For fire conditions: essentially no rain
        self.precipitation = 0.0 + np.random.rand(gs, gs) * 0.001
        self.precipitation = np.clip(self.precipitation, 0.0, 0.002)

        # Surface Pressure (sp) - Training range: 0-101212 Pascals
        # Use realistic atmospheric pressure (decreases with elevation)
        self.pressure = 101325 - self.dem * 12 + np.random.randn(gs, gs) * 500
        self.pressure = np.clip(self.pressure, 85000, 101212)

        # Solar Radiation (ssrd) - Training range: 0-15,869,024 J/m²
        # Uniform solar radiation like real sunlight - consistent across area
        # Only varies by elevation (slightly more at high altitude) and slope aspect
        base_solar = np.random.uniform(14_000_000, 15_000_000)  # High solar day
        # Elevation effect: slightly more radiation at higher elevations (thinner atmosphere)
        elevation_solar = (self.dem / 2000.0) * 500_000  # Up to 0.5 MJ/m² increase
        # Aspect effect: south-facing slopes get more sun (northern hemisphere)
        # aspect ranges 0-360, south is ~180 degrees
        aspect_factor = np.cos(np.radians(self.aspect - 180)) * 0.5 + 0.5  # 0 to 1
        aspect_solar = aspect_factor * 300_000  # Up to 0.3 MJ/m² variation
        self.solar_radiation = base_solar + elevation_solar + aspect_solar
        self.solar_radiation = np.clip(self.solar_radiation, 14_000_000, 15_869_024)

        # ========================================================================
        # VEGETATION FEATURES - MAXIMUM FUEL LOAD (within training ranges)
        # ========================================================================

        # NDVI - Training range: -0.18 to 0.90 (NORMAL vegetation index!)
        # MAXIMUM fire fuel: DENSE vegetation (0.75-0.90) - lots of fuel!
        # BUT decreases significantly with elevation (less vegetation at high altitude)
        base_ndvi = (self.lc_forest * 0.90 + self.lc_shrubland * 0.85 +
                    self.lc_grassland * 0.80 + self.lc_agriculture * 0.85 +
                    np.random.randn(gs, gs) * 0.02)

        # Elevation factor: reduces vegetation dramatically above 800m
        # At 800m: 100% vegetation, at 1400m: 20% vegetation, at 2000m: ~0% vegetation
        elevation_factor = np.clip(1.0 - (self.dem - 800) / 600, 0.0, 1.0)
        elevation_factor = elevation_factor ** 2  # Quadratic falloff for steeper reduction

        self.ndvi = base_ndvi * elevation_factor
        self.ndvi = np.clip(self.ndvi, 0.0, 0.90)

        # LAI (Leaf Area Index) - Training range: 0-6.8
        # MAXIMUM fire fuel: MAXIMUM LAI (5.5-6.8) - dense leaf coverage!
        # Also reduced by elevation
        self.lai = self.ndvi * 7 + 0.5 + np.random.randn(gs, gs) * 0.3
        self.lai = self.lai * elevation_factor  # Apply elevation reduction again
        self.lai = np.clip(self.lai, 0.0, 6.8)

        # Soil Moisture Index (smi) - Training range: 0-0.79 (fraction)
        # ABSOLUTE MINIMUM: 0% soil moisture - COMPLETELY DRY!
        self.soil_moisture = np.ones((gs, gs)) * 0.001  # Nearly 0

        # LST Day (Land Surface Temperature Day) - Training range: 0-320 K
        # ABSOLUTE MAXIMUM: 320K (47°C) - HOTTEST SURFACE POSSIBLE!
        self.lst_day = np.ones((gs, gs)) * 320.0

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

    def evolve_dynamic_features(self, timestep, fire_mask=None):
        """
        Evolve dynamic environmental features over time.

        Simulates realistic temporal changes in weather and vegetation conditions.
        Static features (terrain, land cover, etc.) remain unchanged.

        Args:
            timestep: Current simulation timestep
            fire_mask: Optional [H, W] boolean array indicating currently burning cells
        """
        gs = self.grid_size

        # Time-based variation patterns
        # Use sine waves with different periods to create realistic fluctuations

        # ========================================================================
        # WEATHER FEATURES - Time-varying
        # ========================================================================

        # Temperature (diurnal cycle + random fluctuations)
        # Slow oscillation over ~20 timesteps (simulating day/night cycle)
        temp_cycle = np.sin(timestep * 2 * np.pi / 20) * 3  # ±3K variation
        temp_noise = np.random.randn(gs, gs) * 0.5  # Small random changes
        self.temperature += temp_cycle + temp_noise
        self.temperature = np.clip(self.temperature, 295, 307)  # Stay in training range

        # Dewpoint (tracks temperature but with lag)
        self.dewpoint = self.temperature - 40 + np.random.randn(gs, gs) * 2
        self.dewpoint = np.clip(self.dewpoint, 250, 270)

        # Relative Humidity (inversely related to temperature)
        # Humidity drops when temperature rises
        humidity_change = -temp_cycle * 0.01 + np.random.randn(gs, gs) * 0.005
        self.humidity += humidity_change
        self.humidity = np.clip(self.humidity, 0.05, 0.20)  # Slightly expanded upper range

        # Wind Speed (realistic weather pattern evolution)
        # Slow, uniform changes like real weather fronts - not random turbulence
        # Gradual strengthening/weakening over time
        wind_cycle = np.sin(timestep * 2 * np.pi / 40) * 0.5  # ±0.5 m/s over 40 timesteps
        # Very subtle spatial variation (minimal local gusts)
        wind_noise = np.random.randn(gs, gs) * 0.05  # Minimal noise
        self.wind_speed += wind_cycle + wind_noise
        self.wind_speed = np.clip(self.wind_speed, 3.5, 5.18)

        # Wind Direction (slow, uniform rotation like weather systems)
        # Prevailing wind direction shifts gradually and uniformly
        # Like a weather front passing through
        direction_shift = np.sin(timestep * 2 * np.pi / 50) * 10  # ±10° shift over 50 timesteps
        # Minimal local variation (terrain effects only)
        direction_noise = np.random.randn(gs, gs) * 2  # Very small turbulence
        self.wind_direction += direction_shift + direction_noise
        self.wind_direction = self.wind_direction % 360

        # Update wind components based on new speed and direction
        wind_dir_rad = np.radians(self.wind_direction)
        self.wind_u = -self.wind_speed * np.sin(wind_dir_rad)
        self.wind_v = -self.wind_speed * np.cos(wind_dir_rad)
        self.wind_u = np.clip(self.wind_u, -1.6, 3.9)
        self.wind_v = np.clip(self.wind_v, -3.8, 3.6)

        # Precipitation (rare events)
        # Very low baseline with occasional small increases
        precip_event = 1.0 if np.random.rand() < 0.05 else 0.0  # 5% chance of rain
        self.precipitation = precip_event * 0.002 + np.random.rand(gs, gs) * 0.0005
        self.precipitation = np.clip(self.precipitation, 0.0, 0.0038)

        # Surface Pressure (slowly varying with temperature)
        # Decreases when temperature rises (thermal low)
        pressure_change = -temp_cycle * 50 + np.random.randn(gs, gs) * 100
        self.pressure += pressure_change
        self.pressure = np.clip(self.pressure, 85000, 101212)

        # Solar Radiation (diurnal cycle)
        # Uniform sunlight across the area - only time-based variation
        # No random spatial noise (sun shines uniformly!)
        solar_cycle = (np.sin(timestep * 2 * np.pi / 20) + 1) / 2  # 0 to 1
        # Base radiation varies with time of day, plus small elevation/aspect effects
        base_solar = 14_000_000 * solar_cycle
        elevation_solar = (self.dem / 2000.0) * 500_000 * solar_cycle
        aspect_factor = np.cos(np.radians(self.aspect - 180)) * 0.5 + 0.5
        aspect_solar = aspect_factor * 300_000 * solar_cycle
        self.solar_radiation = base_solar + elevation_solar + aspect_solar
        self.solar_radiation = np.clip(self.solar_radiation, 0, 15_869_024)

        # ========================================================================
        # VEGETATION FEATURES - Slowly evolving
        # ========================================================================

        # NDVI (decreases slightly over time due to drying/burning)
        # Very slow natural degradation
        ndvi_change = -0.001 + np.random.randn(gs, gs) * 0.002

        # FUEL CONSUMPTION: Completely burn all vegetation in fire areas IMMEDIATELY
        if fire_mask is not None:
            # Set NDVI to 0 in all burning cells (complete fuel consumption)
            self.ndvi[fire_mask] = 0.0
        else:
            # Only apply natural degradation if no fire
            self.ndvi += ndvi_change
            self.ndvi = np.clip(self.ndvi, 0.0, 0.90)

        # LAI (follows NDVI)
        self.lai = self.ndvi * 7 + 0.5
        self.lai = np.clip(self.lai, 0.0, 6.8)  # Can go to 0 (no leaves left)

        # Soil Moisture (decreases with high temperature, increases with rain)
        moisture_change = -0.002 * (self.temperature - 295) / 10  # Evaporation
        moisture_change += self.precipitation * 50  # Rain adds moisture
        moisture_change += np.random.randn(gs, gs) * 0.005

        # Fire dries out soil completely
        if fire_mask is not None:
            # Set soil moisture to 0 in burning areas (complete drying)
            self.soil_moisture[fire_mask] = 0.0
        else:
            # Apply normal moisture dynamics if no fire
            self.soil_moisture += moisture_change
            self.soil_moisture = np.clip(self.soil_moisture, 0.0, 0.20)

        # LST Day (tracks air temperature + solar radiation)
        self.lst_day = self.temperature + 12 + solar_cycle * 5 + np.random.randn(gs, gs) * 2
        self.lst_day = np.clip(self.lst_day, 305, 320)

        # LST Night (cooler than day)
        self.lst_night = self.temperature - 2 + np.random.randn(gs, gs) * 2
        self.lst_night = np.clip(self.lst_night, 290, 302)

        # VPD (Vapor Pressure Deficit) - remains zero to match training
        # Training data has VPD issues, so we keep it at zero
        self.vpd = np.zeros((gs, gs))

    def update_weather(self, **kwargs):
        """
        Update weather parameters dynamically (manual override).

        Args:
            temperature: Temperature in K
            humidity: Humidity in fraction (0-1)
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
            wind_dir_rad = np.radians(self.wind_direction)
            self.wind_u = -self.wind_speed * np.sin(wind_dir_rad)
            self.wind_v = -self.wind_speed * np.cos(wind_dir_rad)
