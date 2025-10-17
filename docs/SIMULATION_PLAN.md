# Interactive Wildfire Spread Simulation - Implementation Plan

## Overview
Create an interactive 3D visualization of wildfire spread on a 64×64 grid where users can:
- View terrain and environmental conditions in 3D
- Click to ignite fires at any grid location
- Watch the trained 3D U-Net model predict fire spread in real-time
- Adjust environmental parameters (wind, temperature, humidity, etc.)
- Control simulation speed and reset

---

## Architecture Design

### 1. System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  3D Viewer  │  │   Controls   │  │  Info Display    │   │
│  │  (PyVista)  │  │  (Sliders)   │  │  (Statistics)    │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  SIMULATION ENGINE LAYER                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  SimulationEngine                                     │   │
│  │  - State management (fire history, timesteps)        │   │
│  │  - Model inference loop                              │   │
│  │  - Feature computation (fire history, ignitions)     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     MODEL LAYER                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3D U-Net Model                                       │   │
│  │  - Input: [1, 30, 10, 64, 64] (B, C, T, H, W)       │   │
│  │  - Output: [1, 1, 10, 64, 64] fire probabilities    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Environment Data                                     │   │
│  │  - Terrain (DEM, slope, aspect, curvature, TRI)     │   │
│  │  - Weather (temp, humidity, wind, pressure, etc.)    │   │
│  │  - Vegetation (NDVI, LAI, soil moisture, VPD, etc.) │   │
│  │  - Fire state (fire history, ignition points)       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack Selection

### Option A: PyVista + PyQt (RECOMMENDED)
**Pros:**
- True 3D visualization with terrain elevation
- Interactive camera controls (rotate, zoom, pan)
- High-quality rendering
- Excellent integration with PyQt for UI controls
- Can handle real-time updates efficiently

**Cons:**
- Slightly more complex setup
- Requires VTK backend

### Option B: PyGame
**Pros:**
- Simple 2D rendering
- Fast performance
- Easy to learn
- Good for simple visualizations

**Cons:**
- Limited 3D capabilities (pseudo-3D only)
- Manual implementation of many features
- Less polished UI

### Option C: Plotly Dash
**Pros:**
- Web-based (accessible anywhere)
- Beautiful default styling
- Good for dashboards

**Cons:**
- Not ideal for real-time simulation
- Limited interactivity for clicking grid cells
- Slower updates

**RECOMMENDATION: PyVista + PyQt** for best balance of 3D visualization, interactivity, and performance.

---

## 3. Detailed Implementation Plan

### Phase 3.1: Simulation Engine Core (2-3 days)

#### File: `src/simulation/simulation_engine.py`

```python
class WildfireSimulation:
    """
    Core simulation engine for wildfire spread prediction.

    Manages:
    - Environmental state (64x64 grid with 30 features)
    - Fire history (10 timesteps)
    - Model inference
    - State updates
    """

    def __init__(self, model_path, environment_data, normalization_stats):
        # Load trained 3D U-Net model
        # Initialize environment features (terrain, weather, vegetation)
        # Initialize fire history buffer (10 timesteps)
        # Initialize ignition points

    def ignite_fire(self, x, y):
        """Start a fire at grid position (x, y)"""

    def step(self):
        """
        Execute one simulation timestep:
        1. Update fire history (shift + add new ignition points)
        2. Compute all 30 input features
        3. Run model inference
        4. Update fire state based on predictions
        5. Return new fire map
        """

    def reset(self):
        """Reset simulation to initial state"""

    def update_weather(self, param, value):
        """Update weather parameters (wind, temp, etc.)"""

    def get_current_state(self):
        """Return current visualization data"""
```

**Key Challenges:**
- **Fire History Buffer**: Model expects 10 timesteps of fire history. Need to maintain a rolling window.
- **Feature Engineering**: Must replicate the same feature engineering from training (fire_history_cumsum, etc.)
- **Real-time Inference**: Model input is [B, C, T, H, W] = [1, 30, 10, 64, 64]. Need efficient tensor management.

#### File: `src/simulation/environment.py`

```python
class Environment:
    """
    Manages environmental data for the 64x64 grid.

    Features (30 channels):
    - Static: DEM, slope, aspect, curvature, roads, population, land cover
    - Dynamic: Temperature, humidity, wind, precipitation, etc.
    - Derived: VPD, TRI, fire history, ignition points
    """

    def __init__(self, terrain_source='sample' or 'real'):
        # Option 1: Load from real .nc file (Albania sample)
        # Option 2: Generate synthetic terrain using noise functions

    def get_features(self, timestep):
        """Return all 30 features for given timestep"""

    def update_dynamic_features(self, **kwargs):
        """Update weather parameters"""
```

---

### Phase 3.2: 3D Visualization System (3-4 days)

#### File: `src/visualization/viewer_3d.py`

```python
import pyvista as pv
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QSlider, QPushButton

class WildfireViewer3D(QMainWindow):
    """
    Interactive 3D viewer for wildfire simulation.

    Features:
    - 3D terrain mesh colored by elevation
    - Fire overlay (colored by intensity/probability)
    - Camera controls (rotate, zoom, pan)
    - Interactive grid selection (click to ignite)
    - Real-time updates
    """

    def __init__(self, simulation_engine):
        # Create PyVista plotter
        # Load terrain mesh
        # Initialize fire layer
        # Setup UI controls

    def create_terrain_mesh(self):
        """
        Create 3D mesh from DEM data
        - X, Y: 64x64 grid coordinates
        - Z: Elevation from DEM
        - Color: Terrain color map
        """

    def update_fire_layer(self, fire_map):
        """
        Update fire visualization
        - fire_map: [64, 64] probability values
        - Use color map: blue (no fire) -> yellow -> red (intense fire)
        - Add transparency for low probabilities
        """

    def on_click(self, point):
        """Handle mouse click to ignite fire"""

    def setup_controls(self):
        """
        Create UI controls:
        - Play/Pause button
        - Reset button
        - Speed slider
        - Weather sliders (wind speed, direction, temperature, humidity)
        - Statistics panel (burned area, timestep, etc.)
        """
```

**Visualization Features:**
1. **Terrain Layer**: 3D surface mesh with elevation
2. **Fire Layer**: Semi-transparent overlay showing fire intensity
3. **Wind Vector**: Arrows showing wind direction
4. **Legend**: Color bar for fire intensity
5. **Info Panel**: Statistics (burned area, current timestep, etc.)

---

### Phase 3.3: User Interaction System (2 days)

#### File: `src/visualization/controls.py`

```python
class SimulationControls:
    """
    UI controls for simulation parameters.

    Controls:
    - Playback: Play, Pause, Reset, Step Forward
    - Speed: Simulation speed multiplier (0.5x - 5x)
    - Weather:
        * Wind Speed (0-30 m/s)
        * Wind Direction (0-360°)
        * Temperature (0-50°C)
        * Humidity (0-100%)
        * Precipitation (0-50 mm)
    - Vegetation:
        * Fuel Moisture (adjusts NDVI/LAI)
    """

    def create_slider(self, name, min_val, max_val, default, callback):
        """Create labeled slider with value display"""

    def on_parameter_change(self, param_name, value):
        """Update simulation environment when slider changes"""
```

---

### Phase 3.4: Integration & Main Application (2 days)

#### File: `src/simulation/app.py`

```python
import sys
from PyQt5.QtWidgets import QApplication
from simulation_engine import WildfireSimulation
from visualization.viewer_3d import WildfireViewer3D

def main():
    """
    Main application entry point.

    Steps:
    1. Load trained model
    2. Initialize environment (terrain, weather)
    3. Create simulation engine
    4. Launch 3D viewer
    5. Start event loop
    """

    # Configuration
    MODEL_PATH = 'prod_models/UNET3D/3D-UNET-WILDFIRE-1.pt'
    STATS_PATH = 'data/processed/normalization_stats.json'

    # Option 1: Use real sample data
    SAMPLE_DATA = 'data/raw/dataset_64_64_all_10days_final/2022/Albania/sample.nc'

    # Option 2: Generate synthetic terrain
    # environment = Environment(terrain_source='synthetic')

    # Initialize simulation
    simulation = WildfireSimulation(MODEL_PATH, SAMPLE_DATA, STATS_PATH)

    # Create Qt application
    app = QApplication(sys.argv)
    viewer = WildfireViewer3D(simulation)
    viewer.show()

    # Start simulation loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
```

---

## 4. Simulation Loop Design

```python
# Pseudocode for simulation timestep
def simulation_step():
    # 1. Get current fire state (last 10 timesteps)
    fire_history = state.fire_buffer  # [10, 64, 64]

    # 2. Get current ignition points (user clicks)
    ignition_points = state.ignition_map  # [64, 64]

    # 3. Compute all 30 features
    features = environment.get_features(timestep)
    features[28] = fire_history  # Fire history channel
    features[29] = ignition_points  # Ignition points channel

    # 4. Normalize features
    features_normalized = normalize(features, stats)

    # 5. Prepare model input [B, C, T, H, W]
    model_input = features_normalized.unsqueeze(0)  # [1, 30, 10, 64, 64]
    model_input = model_input.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]

    # 6. Run inference
    with torch.no_grad():
        prediction = model(model_input)  # [1, 1, 10, 64, 64]

    # 7. Get next timestep prediction
    next_fire_prob = torch.sigmoid(prediction[0, 0, -1])  # [64, 64]
    next_fire_binary = (next_fire_prob > 0.5).float()

    # 8. Update fire history buffer (rolling window)
    fire_history = torch.cat([fire_history[1:], next_fire_binary.unsqueeze(0)], dim=0)

    # 9. Update visualization
    viewer.update_fire_layer(next_fire_prob.cpu().numpy())

    # 10. Update statistics
    burned_area = next_fire_binary.sum().item() * grid_area
    stats.update(burned_area, timestep)
```

---

## 5. Key Implementation Challenges & Solutions

### Challenge 1: Model Input Format
**Problem**: Model expects 10 timesteps of data, but simulation starts with no fire history.

**Solution**:
- Initialize fire history buffer with zeros: `torch.zeros(10, 64, 64)`
- As simulation runs, maintain rolling window of last 10 timesteps
- User ignition creates new fire point in current timestep
- Model predicts next 10 timesteps, use only the last one

### Challenge 2: Real-time Performance
**Problem**: Model inference might be slow for real-time visualization.

**Solution**:
- Use GPU if available (`device='mps'` or `'cuda'`)
- Add frame skip option (update every N frames)
- Add speed control (0.5x to 5x)
- Consider model quantization if needed

### Challenge 3: Environment Data Source
**Problem**: Need realistic terrain and weather data for any location.

**Options**:
1. **Use Sample Data**: Load from existing Albania .nc files
2. **Synthetic Generation**: Use Perlin noise for terrain, random weather
3. **User Upload**: Allow users to upload their own .nc files
4. **Web API**: Fetch real DEM/weather data (future enhancement)

**Recommendation**: Start with Option 1 (sample data), add Option 2 (synthetic) for flexibility.

### Challenge 4: Click-to-Ignite Interaction
**Problem**: Mapping 2D screen clicks to 3D grid coordinates.

**Solution**:
- PyVista has built-in `enable_cell_picking()` method
- Returns 3D coordinates → convert to grid indices (x, y)
- Update ignition_points[y, x] = 1 for clicked cell
- Visual feedback: highlight clicked cell

---

## 6. File Structure

```
src/
├── simulation/
│   ├── __init__.py
│   ├── app.py                    # Main application entry point
│   ├── simulation_engine.py      # Core simulation logic
│   ├── environment.py             # Environment data management
│   └── terrain_generator.py      # Synthetic terrain generation
│
├── visualization/
│   ├── __init__.py
│   ├── viewer_3d.py              # PyVista 3D viewer
│   ├── controls.py                # UI controls (sliders, buttons)
│   └── colormaps.py               # Custom color schemes
│
└── utils/
    ├── __init__.py
    ├── feature_engineering.py     # Feature computation utilities
    └── config.py                  # Configuration management
```

---

## 7. Implementation Timeline

### Week 1: Simulation Engine (Phase 3.1)
- **Day 1-2**: Environment data loading & feature engineering
- **Day 3-4**: Simulation engine core logic
- **Day 5**: Testing with model inference

### Week 2: Visualization (Phase 3.2)
- **Day 1-2**: PyVista 3D terrain rendering
- **Day 3**: Fire layer overlay
- **Day 4**: Interactive clicking
- **Day 5**: Polish and styling

### Week 3: UI & Integration (Phase 3.3 + 3.4)
- **Day 1-2**: Control panel (sliders, buttons)
- **Day 3**: Integration with simulation engine
- **Day 4**: Statistics display
- **Day 5**: Testing and bug fixes

### Week 4: Enhancements
- **Day 1**: Synthetic terrain generation
- **Day 2**: Multiple ignition points
- **Day 3**: Save/load simulation state
- **Day 4**: Export animations (GIF/MP4)
- **Day 5**: Documentation and demo

---

## 8. Testing Strategy

1. **Unit Tests**:
   - Test feature engineering matches training pipeline
   - Test fire history buffer rolling window
   - Test ignition point updates

2. **Integration Tests**:
   - Test model inference with sample data
   - Test simulation loop completes without errors
   - Test UI interaction triggers correct updates

3. **Visual Tests**:
   - Verify fire spreads uphill (elevation effect)
   - Verify fire spreads with wind direction
   - Verify high VPD increases spread rate

4. **Performance Tests**:
   - Measure FPS (target: 30+ FPS)
   - Measure inference time (target: <50ms per step)
   - Test with different grid sizes

---

## 9. Configuration File

#### File: `config/simulation_config.yaml`

```yaml
# Simulation Configuration
simulation:
  grid_size: 64
  timesteps: 10
  initial_speed: 1.0  # 1x speed
  update_interval: 100  # milliseconds

model:
  path: 'prod_models/UNET3D/3D-UNET-WILDFIRE-1.pt'
  device: 'mps'  # or 'cuda', 'cpu'
  threshold: 0.5  # Fire probability threshold

environment:
  source: 'sample'  # 'sample', 'synthetic', 'upload'
  sample_path: 'data/raw/dataset_64_64_all_10days_final/2022/Albania/sample.nc'

weather_defaults:
  temperature: 25.0  # °C
  humidity: 50.0  # %
  wind_speed: 5.0  # m/s
  wind_direction: 0.0  # degrees
  precipitation: 0.0  # mm

visualization:
  window_size: [1200, 800]
  terrain_colormap: 'terrain'
  fire_colormap: 'hot'
  show_wind_vectors: true
  camera_elevation: 45
  camera_azimuth: 45

controls:
  enable_click_ignition: true
  max_ignition_points: 10
  allow_runtime_weather_change: true
```

---

## 10. Next Steps - Prioritized TODO List

### Immediate (Do First):
1. ✅ Review and approve this plan
2. Install PyVista: `pip install pyvista PyQt5`
3. Create simulation engine skeleton
4. Load sample environment data from .nc file
5. Test model inference with sample data

### Core Development:
6. Implement fire history buffer management
7. Implement simulation step loop
8. Create basic 3D terrain viewer
9. Add fire overlay visualization
10. Implement click-to-ignite interaction

### Polish:
11. Add UI controls (sliders, buttons)
12. Add statistics panel
13. Implement play/pause/reset
14. Add synthetic terrain generation
15. Create demo and documentation

---

## 11. Alternative: Quick Prototype with Matplotlib

If you want a **faster prototype** before full 3D implementation:

```python
# Quick 2D prototype with matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Terrain elevation
terrain_img = ax1.imshow(dem_data, cmap='terrain')
ax1.set_title('Terrain Elevation')

# Right: Fire spread
fire_img = ax2.imshow(fire_state, cmap='hot', vmin=0, vmax=1)
ax2.set_title('Fire Probability')

def onclick(event):
    if event.inaxes == ax2:
        x, y = int(event.xdata), int(event.ydata)
        simulation.ignite_fire(x, y)

def update(frame):
    fire_state = simulation.step()
    fire_img.set_data(fire_state)
    return [fire_img]

fig.canvas.mpl_connect('button_press_event', onclick)
anim = FuncAnimation(fig, update, interval=100, blit=True)
plt.show()
```

This gives you a working simulation in ~100 lines while you build the full 3D version.

---

## Summary

**Recommended Approach:**
1. Start with **Matplotlib prototype** (1-2 days) to validate simulation logic
2. Build **PyVista 3D version** (2-3 weeks) for final product
3. Focus on core features first, then add polish

**Key Success Factors:**
- Maintain fire history buffer correctly (critical for model input)
- Replicate feature engineering from training exactly
- Optimize for real-time performance (GPU, efficient updates)
- Make UI intuitive and responsive

Ready to start implementing? Let me know which component you'd like to tackle first!
