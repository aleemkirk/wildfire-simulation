# Wildfire Spread Simulation - ML Project

## Project Overview
An interactive wildfire spread simulation powered by machine learning that predicts fire propagation based on environmental factors and provides real-time visualization.

## Project Goals
1. Build an ML model to simulate realistic wildfire spread patterns
2. Create interactive visualizations using Python graphics engines
3. Allow users to adjust environmental parameters and observe impacts
4. Provide accurate predictions based on terrain, weather, and vegetation data

## Technical Architecture

### ML Model Approach
- **Primary**: Hybrid CNN-LSTM or U-Net architecture
  - CNN layers: Extract spatial features from terrain/vegetation
  - LSTM/RNN layers: Capture temporal dynamics of fire spread
  - Output: Probability map of fire spread for next timestep

- **Alternative**: Physics-Informed Neural Networks (PINNs)
  - Combines physics equations (Rothermel fire spread model) with neural networks
  - Ensures predictions follow physical laws

### Input Features
- **Terrain**: Elevation, slope, aspect
- **Weather**: Wind speed/direction, temperature, humidity, rainfall
- **Vegetation**: Fuel type, fuel moisture, vegetation density
- **Fire State**: Current fire perimeter, intensity, duration

### Output
- 2D grid showing fire spread probability for next timestep
- Fire intensity levels
- Predicted burn area and direction

## Project Structure
```
Wildfire-Spread-Simulation/
├── data/
│   ├── raw/              # Raw wildfire datasets
│   ├── processed/        # Preprocessed data for training
│   └── README.md         # Data sources and descriptions
├── src/
│   ├── models/           # ML model architectures
│   ├── preprocessing/    # Data preprocessing scripts
│   ├── simulation/       # Simulation engine
│   ├── visualization/    # PyGame/Plotly visualization code
│   └── utils/            # Helper functions
├── notebooks/            # Jupyter notebooks for experimentation
├── trained_models/       # Saved model checkpoints
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── Claude.md            # This file
└── README.md            # User-facing documentation
```

## Data Sources
1. **NASA FIRMS** (Fire Information for Resource Management System)
   - Near real-time wildfire data
   - https://firms.modaps.eosdis.nasa.gov/

2. **Kaggle Wildfire Datasets**
   - Historical wildfire records
   - Various preprocessed datasets

3. **LANDFIRE** (Landscape Fire and Resource Management Planning Tools)
   - Vegetation and fuel data
   - https://landfire.gov/

4. **NOAA Weather Data**
   - Historical weather records
   - Wind, temperature, humidity data

## Development Phases

### Phase 1: Setup & Data ✓
- [x] Create requirements.txt
- [x] Create Claude.md
- [ ] Set up project directory structure
- [ ] Gather and explore wildfire datasets
- [ ] Create data preprocessing pipeline

### Phase 2: Model Development
- [ ] Research and select ML architecture
- [ ] Implement model in PyTorch/TensorFlow
- [ ] Create training pipeline with validation
- [ ] Train model on historical data
- [ ] Evaluate model performance (accuracy, precision, recall)

### Phase 3: Simulation Engine
- [ ] Design simulation loop
- [ ] Integrate trained ML model
- [ ] Implement state update mechanism
- [ ] Add multiple fire ignition points
- [ ] Optimize performance

### Phase 4: Visualization
- [ ] Choose graphics engine (PyGame vs Plotly Dash)
- [ ] Create 2D grid visualization
- [ ] Add color coding for fire intensity
- [ ] Implement animation for fire spread
- [ ] Add legends and information panels

### Phase 5: Interactivity
- [ ] Add user controls (sliders, buttons)
- [ ] Implement parameter adjustment (wind, temperature, etc.)
- [ ] Allow manual fire placement
- [ ] Add play/pause/reset controls
- [ ] Implement speed controls

### Phase 6: Polish & Documentation
- [ ] Write comprehensive README
- [ ] Add usage examples
- [ ] Create demo videos/GIFs
- [ ] Document model architecture
- [ ] Add inline code documentation

## Key Technologies
- **ML Frameworks**: PyTorch, TensorFlow
- **Visualization**: PyGame, Plotly Dash, Matplotlib
- **Data Processing**: NumPy, Pandas, SciPy
- **Geospatial**: Rasterio, GeoPandas (optional)
- **Image Processing**: OpenCV, Pillow

## Model Evaluation Metrics
- **Spatial Accuracy**: IoU (Intersection over Union) between predicted and actual fire spread
- **Temporal Accuracy**: Time-to-spread predictions
- **Precision/Recall**: For binary fire/no-fire classification
- **RMSE**: Root Mean Square Error for continuous intensity predictions

## Future Enhancements
- 3D terrain visualization
- Real-time weather API integration
- Multi-agent reinforcement learning for firefighting strategies
- Evacuation route optimization
- Integration with GIS systems
- Web deployment for broader accessibility

## Notes
- Start with simplified 2D grid (e.g., 100x100)
- Use synthetic data if real datasets are difficult to obtain
- Consider cellular automata as baseline before ML
- Balance model complexity with inference speed for real-time simulation

---
*Generated with Claude Code*
*Last Updated: 2025-10-15*