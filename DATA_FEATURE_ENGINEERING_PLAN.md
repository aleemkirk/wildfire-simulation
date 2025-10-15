# Data & Feature Engineering Plan
## Wildfire Spread Simulation Project

---

## 1. Dataset Overview

### Current Data Structure
- **Format**: NetCDF (.nc) files
- **Grid Size**: 64×64 spatial resolution
- **Temporal Coverage**: 10 timesteps per sample (covering ~9-10 days before/during/after fire)
- **Geographic Coverage**: Mediterranean region (27+ countries)
- **Time Range**: 2006-2022
- **Total Variables**: 34 variables (33 features + 1 spatial reference)

### Data Organization
```
dataset_64_64_all_10days_final/
├── 2006/ ... 2022/
│   ├── Albania/
│   ├── Greece/
│   ├── Spain/
│   ├── Portugal/
│   └── [27+ countries]
```

### Variable Categories

#### A. **Target Variable** (1)
- `burned_areas` (time, y, x): Binary mask showing fire spread over time

#### B. **Static Features - Terrain** (4)
- `dem`: Digital Elevation Model (elevation in meters)
- `slope`: Terrain slope (degrees)
- `aspect`: Slope direction/orientation (degrees, 0-360)
- `curvature`: Terrain curvature (concave/convex)

#### C. **Static Features - Land Cover** (8)
- `lc_agriculture`: Agricultural land fraction
- `lc_forest`: Forest cover fraction
- `lc_grassland`: Grassland fraction
- `lc_settlement`: Settlement/urban fraction
- `lc_shrubland`: Shrubland fraction
- `lc_sparse_vegetation`: Sparse vegetation fraction
- `lc_water_bodies`: Water bodies fraction
- `lc_wetland`: Wetland fraction

#### D. **Static Features - Infrastructure** (2)
- `roads_distance`: Distance to nearest road (km)
- `population`: Population density

#### E. **Dynamic Features - Weather** (8)
- `t2m`: 2-meter air temperature (Kelvin)
- `d2m`: 2-meter dewpoint temperature (Kelvin)
- `rh`: Relative humidity (0-1)
- `tp`: Total precipitation (meters)
- `sp`: Surface pressure (Pascals)
- `ssrd`: Surface solar radiation downwards (J/m²)
- `wind_speed`: Wind speed (m/s)
- `wind_direction`: Wind direction (degrees)

#### F. **Dynamic Features - Wind Components** (4)
- `u`: U-component of wind (m/s, east-west)
- `v`: V-component of wind (m/s, north-south)
- `wind_direction_sin`: Sine of wind direction
- `wind_direction_cos`: Cosine of wind direction

#### G. **Dynamic Features - Vegetation** (4)
- `ndvi`: Normalized Difference Vegetation Index (-1 to 1)
- `lai`: Leaf Area Index (0-7)
- `smi`: Soil Moisture Index (0-1)
- `lst_day`: Land Surface Temperature - Day (Kelvin)
- `lst_night`: Land Surface Temperature - Night (Kelvin)

#### H. **Fire Metadata** (1)
- `ignition_points`: Fire ignition locations (timestamped)

---

## 2. Data Preprocessing Pipeline

### Phase 1: Data Collection & Organization

#### 1.1 Inventory All Samples
```python
tasks:
- Scan all years and countries
- Count total samples
- Document data completeness
- Identify missing/corrupted files
```

**Expected Output**:
- CSV file with metadata (sample_id, country, year, date, burned_area_ha, file_path)

#### 1.2 Quality Checks
```python
checks:
- Verify all 34 variables present
- Check for NaN values
- Validate spatial dimensions (64×64)
- Verify temporal dimensions (10 timesteps)
- Check value ranges (anomalies, outliers)
```

**Action Items**:
- Flag samples with >5% missing data
- Document anomalies
- Create data quality report

---

### Phase 2: Feature Engineering

#### 2.1 Derived Terrain Features

**Topographic Wetness Index (TWI)**
```python
TWI = ln(drainage_area / tan(slope))
# Indicates water accumulation potential
```

**Terrain Ruggedness Index (TRI)**
```python
TRI = sqrt(mean((elevation_cell - elevation_neighbors)^2))
# Measures terrain variability
```

**Slope Categories**
```python
flat: slope < 5°
moderate: 5° ≤ slope < 15°
steep: 15° ≤ slope < 30°
very_steep: slope ≥ 30°
```

**Aspect Categories**
```python
north: 337.5° - 22.5°
northeast: 22.5° - 67.5°
east: 67.5° - 112.5°
southeast: 112.5° - 157.5°
south: 157.5° - 202.5°
southwest: 202.5° - 247.5°
west: 247.5° - 292.5°
northwest: 292.5° - 337.5°
```

#### 2.2 Derived Weather Features

**Temperature Metrics**
```python
temp_celsius = t2m - 273.15
dewpoint_celsius = d2m - 273.15
temp_range = lst_day - lst_night
heat_index = f(temperature, humidity)
```

**Moisture Indicators**
```python
vapor_pressure_deficit = f(t2m, rh)
# Measure of atmospheric dryness
```

**Fire Weather Index (FWI) Components**
```python
# Canadian Forest Fire Weather Index System
fine_fuel_moisture_code = f(temp, rh, wind, rain)
duff_moisture_code = f(temp, rh, rain)
drought_code = f(temp, rain)
initial_spread_index = f(wind, fine_fuel_moisture_code)
buildup_index = f(duff_moisture_code, drought_code)
fire_weather_index = f(initial_spread_index, buildup_index)
```

**Wind Features**
```python
wind_u_component = wind_speed * cos(wind_direction)
wind_v_component = wind_speed * sin(wind_direction)
wind_gust_estimate = wind_speed * 1.3
```

#### 2.3 Derived Vegetation Features

**Fuel Moisture Content Proxy**
```python
fuel_moisture = f(smi, rh, precipitation)
# Estimate how dry vegetation is
```

**Vegetation Dryness**
```python
vegetation_stress = (1 - ndvi) * (1 - smi)
# Combined indicator of dry, sparse vegetation
```

**Fuel Load Estimate**
```python
fuel_load = lai * (lc_forest + lc_shrubland + lc_grassland)
# Amount of burnable material
```

#### 2.4 Spatial Context Features

**Neighborhood Statistics** (3×3, 5×5, 7×7 windows)
```python
for each variable:
    - mean_3x3, mean_5x5, mean_7x7
    - std_3x3, std_5x5, std_7x7
    - max_3x3, max_5x5, max_7x7
    - min_3x3, min_5x5, min_7x7
```

**Gradient Features**
```python
slope_x = gradient along x-axis
slope_y = gradient along y-axis
magnitude = sqrt(slope_x^2 + slope_y^2)
direction = atan2(slope_y, slope_x)
```

**Distance to Fire**
```python
for each timestep t:
    distance_to_fire_front = euclidean_distance(cell, nearest_burning_cell)
    distance_to_ignition = euclidean_distance(cell, ignition_point)
```

#### 2.5 Temporal Features

**Time-Based**
```python
day_of_year = julian_day
month = 1-12
season = spring/summer/fall/winter
days_since_ignition = timestep_index
```

**Temporal Aggregations**
```python
For weather variables:
    - rolling_mean_3days
    - rolling_max_3days
    - rolling_min_3days
    - cumulative_precipitation
    - days_since_last_rain
```

**Rate of Change**
```python
delta_temperature = t2m[t] - t2m[t-1]
delta_humidity = rh[t] - rh[t-1]
delta_wind_speed = wind_speed[t] - wind_speed[t-1]
```

#### 2.6 Fire History Features

**Previous Burn State**
```python
burned_t_minus_1 = burned_areas[t-1]
burned_t_minus_2 = burned_areas[t-2]
burned_cumulative = sum(burned_areas[0:t])
```

**Fire Intensity Proxy**
```python
fire_age = days_since_cell_burned
fire_spread_rate = new_burned_area / time_delta
```

---

## 3. Data Normalization & Scaling

### 3.1 Normalization Strategy

**Static Features**: StandardScaler (z-score normalization)
```python
X_normalized = (X - mean) / std
```

**Dynamic Features**:
- Option 1: Per-sample normalization (maintains temporal relationships)
- Option 2: Global normalization (across all samples)

**Categorical Features** (aspect direction, slope category):
- One-hot encoding

**Binary Features** (land cover fractions):
- Keep as-is (already 0-1 range)

### 3.2 Handling Special Cases

**Circular Features** (wind_direction, aspect):
```python
sin_component = sin(angle * 2π / 360)
cos_component = cos(angle * 2π / 360)
```

**Log-Transform Candidates**:
- Population density (highly skewed)
- Roads distance (long tail)
- Precipitation (sparse, extreme values)

---

## 4. Data Splitting Strategy

### 4.1 Temporal Split (Recommended)
```
Train: 2006-2018 (70%)
Validation: 2019-2020 (15%)
Test: 2021-2022 (15%)
```
**Rationale**: Prevents temporal leakage, tests model on recent fires

### 4.2 Geographic Split (Alternative)
```
Train: Countries A-M
Validation: Countries N-S
Test: Countries T-Z
```
**Rationale**: Tests geographic generalization

### 4.3 Stratified Split (Recommended for balance)
```python
Stratify by:
- Burned area size (small: <1000ha, medium: 1000-3000ha, large: >3000ha)
- Season (spring, summer, fall, winter)
- Country/region
```

### 4.4 Cross-Validation Strategy
```
K-Fold (k=5) with stratification
- Ensures all data used for training
- Robust performance estimates
```

---

## 5. Data Augmentation Techniques

### 5.1 Spatial Augmentation
```python
- Horizontal flip (mirror)
- Vertical flip
- 90°, 180°, 270° rotation
- Random crops (if scaling to larger models)
```

**Note**: Wind directions must be adjusted for flips/rotations!

### 5.2 Temporal Augmentation
```python
- Random timestep jittering (±1 day)
- Temporal window shifting
- Varying sequence length (7, 8, 9, 10 days)
```

### 5.3 Weather Perturbation
```python
- Add Gaussian noise (σ=0.05) to temperature, humidity
- Simulate measurement uncertainty
```

---

## 6. Feature Selection & Importance

### 6.1 Correlation Analysis
```python
- Remove highly correlated features (r > 0.95)
- Example: t2m and lst_day are likely correlated
```

### 6.2 Feature Importance Methods
```python
1. Random Forest feature importance
2. SHAP values
3. Permutation importance
4. Ablation studies
```

### 6.3 Expected Key Features (Hypothesis)
**Top predictors**:
1. Wind speed & direction
2. Temperature (t2m, lst_day)
3. Humidity (rh, d2m)
4. Slope & aspect
5. Vegetation type (lc_forest, lc_shrubland)
6. Fuel moisture (smi, ndvi)
7. Previous burn state (t-1, t-2)

---

## 7. Implementation Roadmap

### Week 1: Data Collection & EDA
- [ ] Create data inventory script
- [ ] Generate metadata CSV
- [ ] Run quality checks
- [ ] Create visualization notebooks

### Week 2: Core Preprocessing
- [ ] Implement data loading pipeline
- [ ] Handle missing values
- [ ] Normalize/scale features
- [ ] Create train/val/test splits

### Week 3: Feature Engineering
- [ ] Implement derived terrain features
- [ ] Implement weather-based features
- [ ] Implement spatial context features
- [ ] Create temporal aggregations

### Week 4: Advanced Features
- [ ] Implement Fire Weather Index
- [ ] Create neighborhood statistics
- [ ] Implement data augmentation
- [ ] Feature selection analysis

---

## 8. Key Challenges & Solutions

### Challenge 1: Imbalanced Classes
**Problem**: Most cells don't burn (burned_areas is sparse)
**Solutions**:
- Weighted loss function (higher weight to burned cells)
- Focal loss (focuses on hard examples)
- Oversampling burned regions
- Use IoU/Dice metrics instead of accuracy

### Challenge 2: Temporal Dependencies
**Problem**: Fire spread depends on previous timesteps
**Solutions**:
- Use LSTM/GRU layers
- Temporal attention mechanisms
- ConvLSTM (combines CNN + LSTM)

### Challenge 3: Spatial Dependencies
**Problem**: Fire spreads to neighboring cells
**Solutions**:
- Use CNN with appropriate receptive field
- Graph Neural Networks
- U-Net architecture for segmentation

### Challenge 4: Scale Variance
**Problem**: Different features have vastly different ranges
**Solutions**:
- Robust normalization (median, IQR)
- Per-feature scaling strategies
- Batch normalization in model

### Challenge 5: Missing/Noisy Data
**Problem**: Satellite data can have gaps
**Solutions**:
- Interpolation (temporal/spatial)
- Masking layers in model
- Robust loss functions

---

## 9. Data Storage & Formats

### 9.1 Preprocessed Data Format
**Option 1: HDF5**
```python
structure:
  /train/
    /features/ [N, 10, 64, 64, F]
    /targets/  [N, 10, 64, 64]
    /metadata/ [N, M]
  /val/
  /test/
```

**Option 2: Zarr** (better for cloud/parallel access)
```python
Similar structure to HDF5 but chunked for efficiency
```

**Option 3: TFRecord / PyTorch Dataset** (framework-specific)

### 9.2 Feature Matrices
**Shape**: `[N_samples, N_timesteps, Height, Width, N_features]`
- N_samples: ~thousands
- N_timesteps: 10
- Height: 64
- Width: 64
- N_features: 50-100 (after feature engineering)

**Estimated Size**:
- Float32: 4 bytes
- 1000 samples × 10 × 64 × 64 × 80 features
- ≈ 1.3 GB per split

---

## 10. Success Metrics

### Data Quality Metrics
- [ ] <1% missing values across all samples
- [ ] All features within expected ranges
- [ ] No duplicate samples
- [ ] Balanced temporal/geographic distribution

### Feature Engineering Metrics
- [ ] Feature importance analysis completed
- [ ] Correlation matrix verified (<0.95 for any pair)
- [ ] Domain expert validation of derived features

### Pipeline Efficiency
- [ ] Data loading: <1 second per sample
- [ ] Preprocessing: <10 seconds per sample
- [ ] Total pipeline: process full dataset in <2 hours

---

## 11. Next Steps After Feature Engineering

1. **Baseline Model**: Simple logistic regression or random forest
2. **CNN Model**: 2D CNN for spatial features
3. **CNN-LSTM**: Spatial + temporal modeling
4. **U-Net**: Segmentation-based approach
5. **Advanced**: Graph Neural Networks, Transformers

---

## References & Resources

### Fire Weather Index
- Canadian Forest Service FWI System
- McArthur Forest Fire Danger Index

### Remote Sensing
- MODIS vegetation indices
- ERA5 weather reanalysis

### ML for Wildfire
- Recent papers on fire spread prediction
- FireCast, Next Day Wildfire Spread models

---

**Document Version**: 1.0
**Last Updated**: 2025-10-15
**Author**: Generated with Claude Code
