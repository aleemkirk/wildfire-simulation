# Wildfire Spread Simulation 🔥

An interactive machine learning-powered wildfire spread simulation that predicts fire propagation patterns based on environmental factors using Python.

## 📋 Project Objectives

The primary goal of this project is to create a realistic, interactive wildfire simulation system that:

1. **Predicts Wildfire Spread** - Uses machine learning models to simulate how wildfires spread across terrain based on environmental conditions
2. **Interactive Visualization** - Provides real-time graphical simulations with adjustable parameters
3. **Educational Tool** - Helps understand wildfire behavior and the impact of various factors (wind, terrain, vegetation)
4. **Research Platform** - Serves as a foundation for studying fire dynamics and testing mitigation strategies

### Key Features
- Real-time fire spread prediction using trained ML models
- Interactive parameter controls (wind speed/direction, temperature, humidity, vegetation)
- Visual representation of fire intensity and spread patterns
- Support for multiple ignition points
- Playback controls (play, pause, reset, speed adjustment)

## 🎯 Development Plan

### Phase 1: Foundation & Data
- [x] Set up project structure and dependencies
- [x] Create project documentation
- [ ] Gather wildfire datasets (NASA FIRMS, Kaggle, LANDFIRE)
- [ ] Build data preprocessing pipeline
- [ ] Explore and analyze historical wildfire patterns

### Phase 2: ML Model Development
- [ ] Research and select optimal ML architecture
- [ ] Implement CNN-LSTM or U-Net model for spatiotemporal prediction
- [ ] Create training pipeline with data augmentation
- [ ] Train model on historical wildfire data
- [ ] Validate and evaluate model performance

### Phase 3: Simulation Engine
- [ ] Design simulation loop architecture
- [ ] Integrate trained ML model for predictions
- [ ] Implement state management and updates
- [ ] Add support for multiple fire sources
- [ ] Optimize for real-time performance

### Phase 4: Visualization & Interactivity
- [ ] Build interactive graphics interface (PyGame or Plotly Dash)
- [ ] Create 2D grid visualization with color-coded fire intensity
- [ ] Implement animation system for fire spread
- [ ] Add user controls and parameter sliders
- [ ] Create information panels and legends

### Phase 5: Testing & Documentation
- [ ] Write unit tests for core functionality
- [ ] Performance testing and optimization
- [ ] Create user documentation and tutorials
- [ ] Add example scenarios and demo videos
- [ ] Prepare for deployment

## 🛠️ Technologies Used

### Machine Learning & Data Science
- **PyTorch** / **TensorFlow** - Deep learning frameworks for model development
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and preprocessing
- **Scikit-learn** - ML utilities and evaluation metrics

### Visualization & Graphics
- **PyGame** - Real-time 2D graphics and interactive visualization
- **Pygame-GUI** - User interface components and controls
- **Plotly** / **Dash** - Alternative web-based interactive visualization
- **Matplotlib** - Static plots and analysis visualization
- **Seaborn** - Statistical data visualization

### Geospatial & Image Processing
- **Rasterio** - Geospatial raster data processing
- **GeoPandas** - Geographic data manipulation
- **OpenCV** - Image processing and computer vision
- **Pillow** - Image handling and manipulation

### Development Tools
- **Jupyter Notebooks** - Experimentation and prototyping
- **pytest** - Unit testing and code quality
- **YAML** - Configuration management
- **tqdm** - Progress bars and monitoring

## 🏗️ Project Structure

```
Wildfire-Spread-Simulation/
├── data/                 # Dataset storage
│   ├── raw/             # Raw wildfire data
│   └── processed/       # Preprocessed training data
├── src/                  # Source code
│   ├── models/          # ML model architectures
│   ├── preprocessing/   # Data preprocessing scripts
│   ├── simulation/      # Simulation engine
│   ├── visualization/   # Graphics and UI code
│   └── utils/           # Helper functions
├── notebooks/            # Jupyter notebooks for experimentation
├── trained_models/       # Saved model checkpoints
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 📦 Dataset Overview

This project uses the **MesoGEOS Wildfire Dataset**, a comprehensive collection of wildfire events across the Mediterranean region from 2006-2022, featuring 64×64 spatial grids with 10-day temporal sequences.

### Dataset Characteristics
- **Format**: NetCDF (.nc) files
- **Spatial Resolution**: 64×64 grid cells
- **Temporal Coverage**: 10 timesteps per sample (~10 days)
- **Geographic Coverage**: 27+ countries across Mediterranean region
- **Time Range**: 2006-2022
- **Total Variables**: 34 features + target

### Dataset Variables

| Category | Variable | Description | Dimensions | Units/Range |
|----------|----------|-------------|------------|-------------|
| **Target** | `burned_areas` | Binary mask showing fire spread over time | (time, y, x) | 0-1 |
| **Terrain** | `dem` | Digital Elevation Model (elevation) | (y, x) | meters |
| | `slope` | Terrain slope | (y, x) | degrees |
| | `aspect` | Slope direction/orientation | (y, x) | degrees (0-360) |
| | `curvature` | Terrain curvature (concave/convex) | (y, x) | unitless |
| **Land Cover** | `lc_agriculture` | Agricultural land fraction | (y, x) | 0-1 |
| | `lc_forest` | Forest cover fraction | (y, x) | 0-1 |
| | `lc_grassland` | Grassland fraction | (y, x) | 0-1 |
| | `lc_settlement` | Settlement/urban fraction | (y, x) | 0-1 |
| | `lc_shrubland` | Shrubland fraction | (y, x) | 0-1 |
| | `lc_sparse_vegetation` | Sparse vegetation fraction | (y, x) | 0-1 |
| | `lc_water_bodies` | Water bodies fraction | (y, x) | 0-1 |
| | `lc_wetland` | Wetland fraction | (y, x) | 0-1 |
| **Infrastructure** | `roads_distance` | Distance to nearest road | (y, x) | kilometers |
| | `population` | Population density | (y, x) | persons/cell |
| **Weather** | `t2m` | 2-meter air temperature | (time, y, x) | Kelvin |
| | `d2m` | 2-meter dewpoint temperature | (time, y, x) | Kelvin |
| | `rh` | Relative humidity | (time, y, x) | 0-1 |
| | `tp` | Total precipitation | (time, y, x) | meters |
| | `sp` | Surface pressure | (time, y, x) | Pascals |
| | `ssrd` | Surface solar radiation downwards | (time, y, x) | J/m² |
| | `wind_speed` | Wind speed | (time, y, x) | m/s |
| | `wind_direction` | Wind direction | (time, y, x) | degrees (0-360) |
| **Wind Components** | `u` | U-component of wind (east-west) | (time, y, x) | m/s |
| | `v` | V-component of wind (north-south) | (time, y, x) | m/s |
| | `wind_direction_sin` | Sine of wind direction | (time, y, x) | -1 to 1 |
| | `wind_direction_cos` | Cosine of wind direction | (time, y, x) | -1 to 1 |
| **Vegetation** | `ndvi` | Normalized Difference Vegetation Index | (time, y, x) | -1 to 1 |
| | `lai` | Leaf Area Index | (time, y, x) | 0-7 |
| | `smi` | Soil Moisture Index | (time, y, x) | 0-1 |
| | `lst_day` | Land Surface Temperature (day) | (time, y, x) | Kelvin |
| | `lst_night` | Land Surface Temperature (night) | (time, y, x) | Kelvin |
| **Fire Metadata** | `ignition_points` | Fire ignition locations | (time, y, x) | timestamped |

### Data Processing Pipeline

The project includes a comprehensive PyTorch-based feature engineering pipeline that:

1. **Loads NetCDF files** - Reads raw wildfire data from disk
2. **Computes derived features** - Generates additional predictive features:
   - Terrain Ruggedness Index (TRI)
   - Vapor Pressure Deficit (VPD)
   - Fuel moisture proxies
   - Vegetation stress indices
   - Neighborhood statistics (3×3, 5×5, 7×7 windows)
   - Rolling temporal aggregations
   - Rate of change metrics
3. **Normalizes features** - Applies standardization and scaling
4. **Augments data** - Spatial transformations (flips, rotations)
5. **Batches for training** - Creates efficient DataLoader for PyTorch

See `DATA_FEATURE_ENGINEERING_PLAN.md` for detailed feature engineering documentation.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Wildfire-Spread-Simulation.git
cd Wildfire-Spread-Simulation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

_Coming soon - detailed usage instructions will be added as features are implemented_

## 📊 ML Model Architecture

The project uses a hybrid **CNN-LSTM** or **U-Net** architecture:

- **Input**: Current fire state + environmental parameters (terrain, weather, vegetation)
- **CNN Layers**: Extract spatial features from terrain and vegetation maps
- **LSTM/RNN Layers**: Capture temporal dynamics of fire spread over time
- **Output**: Probability map showing fire spread likelihood for the next timestep

### Input Features
- Terrain: elevation, slope, aspect
- Weather: wind speed/direction, temperature, humidity
- Vegetation: fuel type, fuel moisture, density
- Fire state: current perimeter, intensity, duration

### Output
- 2D grid with fire spread probabilities
- Fire intensity levels
- Predicted burn area and direction

## 📈 Data Sources

- **NASA FIRMS** - Near real-time active fire data
- **Kaggle Wildfire Datasets** - Historical wildfire records
- **LANDFIRE** - Vegetation and fuel data
- **NOAA** - Weather and climate data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🔮 Future Enhancements

- 3D terrain visualization
- Real-time weather API integration
- Reinforcement learning for firefighting strategies
- Evacuation route optimization
- GIS system integration
- Web deployment for broader accessibility

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

*Built with Python, Machine Learning, and PyGame*