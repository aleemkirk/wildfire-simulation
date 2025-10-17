# Wildfire Simulation Prototype

Interactive wildfire spread simulation using the trained 3D U-Net model.

## Features

- **64×64 Grid**: Randomly generated terrain and environmental conditions
- **Click-to-Ignite**: Click anywhere on the grid to start a fire
- **Real-time Predictions**: Model predicts fire spread based on 30 environmental features
- **Interactive Controls**: Play/pause and reset simulation

## Usage

### Running the Simulation

```bash
python src/simulation/prototype_sim.py
```

### Controls

| Action | Control |
|--------|---------|
| Ignite Fire | **Left-click** on the right panel (Fire Probability) |
| Play/Pause | **SPACEBAR** |
| Reset | **R key** |
| Exit | Close window |

## How It Works

### 1. Environment Generation (`environment.py`)

Generates a realistic 64×64 grid with 30 features:

**Terrain Features (8):**
- DEM (elevation)
- Slope
- Aspect
- Curvature
- TRI (Terrain Ruggedness Index)
- Roads distance
- Population density
- Land cover (5 types: forest, shrubland, grassland, agriculture, settlement)

**Weather Features (11):**
- Temperature
- Humidity
- Wind speed & direction (U, V components)
- Precipitation
- Pressure
- Dewpoint
- Solar radiation

**Vegetation Features (9):**
- NDVI (vegetation index)
- LAI (leaf area index)
- Soil moisture
- LST (land surface temperature - day & night)
- VPD (Vapor Pressure Deficit - **most important for fire spread!**)

**Fire Features (2):**
- Fire history (10 timesteps rolling buffer)
- Ignition points (from user clicks)

### 2. Simulation Engine (`simulation_engine.py`)

**Core Functionality:**
- Maintains 10-timestep fire history buffer (rolling window)
- Normalizes features using training statistics
- Runs model inference each step
- Updates fire state based on predictions

**Simulation Loop:**
```
For each timestep:
  1. Get environment features (28 channels)
  2. Add fire history (cumulative sum over 10 timesteps)
  3. Add ignition points (user clicks)
  4. Normalize all 30 features
  5. Reshape to [B, C, T, H, W] = [1, 30, 10, 64, 64]
  6. Run 3D U-Net inference → [1, 1, 10, 64, 64]
  7. Extract last timestep prediction
  8. Update fire history buffer (shift + append)
  9. Update visualization
```

### 3. Visualization (`prototype_sim.py`)

**Two-Panel Display:**
- **Left Panel**: Terrain elevation (colored by height)
- **Right Panel**: Fire probability (hot colormap: black → red → yellow → white)

**Real-time Updates:**
- Fire spreads based on model predictions
- Statistics show timestep and burned area
- 0.5 second delay between timesteps (adjustable)

## Key Implementation Details

### Fire History Buffer

The model expects 10 timesteps of fire history. We maintain a rolling window:

```python
# Initialize with zeros
fire_history = torch.zeros(10, 64, 64)

# After each prediction, shift and append
fire_history = torch.cat([
    fire_history[1:],      # Remove oldest
    new_prediction.unsqueeze(0)  # Add newest
], dim=0)
```

### Feature Engineering

Must match training pipeline exactly:
- Fire history uses **cumulative sum** over timesteps
- All features normalized with training statistics
- Input shape: [B, C, T, H, W] with C=30, T=10

### Model Input/Output

**Input**: `[1, 30, 10, 64, 64]`
- Batch size: 1
- Channels: 30 features
- Timesteps: 10
- Grid: 64×64

**Output**: `[1, 1, 10, 64, 64]`
- Predicts fire probability for next 10 timesteps
- We use only the **last timestep** (most recent prediction)

## Example Session

```bash
$ python src/simulation/prototype_sim.py

================================================================================
WILDFIRE SPREAD SIMULATION - PROTOTYPE
================================================================================
Initializing simulation on device: mps
Loading model from ../prod_models/UNET3D/3D-UNET-WILDFIRE-1.pt...
✓ Model loaded (epoch 47, IoU: 0.9946)
✓ Loaded normalization stats (30 channels)
✓ Simulation initialized successfully

================================================================================
INSTRUCTIONS
================================================================================
  • Click on the RIGHT panel to ignite fires
  • Press SPACEBAR to play/pause simulation
  • Press 'R' to reset simulation
  • Close window to exit
================================================================================

[Window opens with terrain and fire panels]

# User clicks at (32, 32)
🔥 Fire ignited at (32, 32)

# User presses SPACEBAR
▶ Simulation playing
Step 1: 1 cells burned
Step 2: 3 cells burned
Step 3: 8 cells burned
Step 4: 15 cells burned
...
```

## Environment Variables

The environment features are randomly generated but follow realistic patterns:

- **Temperature**: 10-45°C (higher at lower elevations)
- **Humidity**: 10-90% (higher at higher elevations)
- **Wind Speed**: 0-15 m/s
- **VPD**: 0-5 kPa (computed from temperature & humidity)
- **Elevation**: 0-2000m (using sin/cos patterns)
- **Slope**: 0-45° (derived from elevation gradients)

## Performance

- **Inference Time**: ~50-100ms per step (on MPS/GPU)
- **Frame Rate**: ~2 FPS (0.5s interval)
- **Grid Size**: 64×64 = 4,096 cells

## Files Created

```
src/simulation/
├── environment.py          # Environment data generation
├── simulation_engine.py    # Core simulation logic
├── prototype_sim.py        # Matplotlib visualization
└── README.md              # This file
```

## Next Steps

This prototype validates the simulation logic. For the full 3D version:

1. **PyVista 3D Viewer** - Replace matplotlib with 3D terrain
2. **UI Controls** - Add sliders for weather parameters
3. **Multiple Ignitions** - Allow multiple fire points
4. **Synthetic Terrain** - Use Perlin noise for varied landscapes
5. **Export Animations** - Save simulations as GIF/MP4

## Troubleshooting

**Issue**: "Model not found"
- **Solution**: Ensure model exists at `prod_models/UNET3D/3D-UNET-WILDFIRE-1.pt`

**Issue**: "Normalization stats not found"
- **Solution**: Ensure stats exist at `data/processed/normalization_stats.json`

**Issue**: Simulation is slow
- **Solution**: Change `device='cpu'` or adjust `interval=500` in `FuncAnimation`

**Issue**: Fire doesn't spread
- **Solution**: Try clicking multiple cells or check VPD values (should be high for spread)

## Technical Notes

### Why Fire History Cumsum?

During training, the fire history feature is computed as:
```python
fire_history = burned_areas.cumsum(dim=1)  # Cumulative over time
```

This gives the model information about:
- How long fire has been burning at each location
- Spatial pattern of fire progression
- Total burned area over time

### Why Use Last Timestep Only?

The model predicts 10 future timesteps, but for simulation we only need the immediate next step. Using all 10 would:
- Jump too far ahead
- Miss intermediate fire states
- Make the simulation less controllable

By taking only the last prediction and feeding it back, we create a smooth, step-by-step simulation.

### Fire Spread Factors

The model learned that fire spreads faster when:
- **VPD is high** (dry air) - most important feature!
- **Wind speed is high** - pushes fire forward
- **Slope is upward** - fire moves uphill faster
- **Vegetation is dense** - more fuel
- **Humidity is low** - drier conditions

## Demo

The simulation should show:
1. Click → small fire starts (1 cell)
2. After a few steps → fire grows (2-5 cells)
3. Continues spreading → larger patches form
4. Eventually → fire intensity map shows probabilities

Fire will spread realistically based on the learned patterns from the training data!
