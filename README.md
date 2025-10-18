# Wildfire Spread Simulation

A machine learning-powered wildfire spread simulation that predicts fire propagation patterns using deep learning models trained on real-world Mediterranean wildfire data.

![Wildfire Simulation Demo](results/simulation_2000_steps.gif)

## Overview

This project implements spatiotemporal deep learning models (3D U-Net and ConvLSTM) to predict wildfire spread based on environmental factors including terrain, weather, vegetation, and fire history. The simulation provides real-time visualization of fire progression with dual model predictions.

## Technologies Used

- **PyTorch** - Deep learning framework for model development and training
- **NumPy & Pandas** - Data manipulation and numerical computations
- **xarray & netCDF4** - NetCDF file handling for spatiotemporal datasets
- **Matplotlib & Seaborn** - Visualization and plotting
- **Pillow** - Image processing and GIF creation
- **Scikit-learn** - Model evaluation metrics

## Dataset

**MesoGEOS Wildfire Dataset** - Mediterranean wildfire events from 2006-2022

- **Format**: NetCDF (.nc) files
- **Spatial Resolution**: 64×64 grid cells
- **Temporal Sequence**: 10 timesteps per sample
- **Geographic Coverage**: Mediterranean region (27+ countries)
- **Total Features**: 34 environmental variables

### Input Variables

**Terrain Features** (Static)
- Digital Elevation Model (DEM), slope, aspect, curvature
- Land cover types (forest, grassland, shrubland, agriculture, settlement, water, wetland)
- Population density, distance to roads

**Weather Features** (Temporal)
- Temperature (2m air temp, dewpoint, land surface temp day/night)
- Humidity (relative humidity, soil moisture)
- Wind (speed, direction, U/V components, directional sine/cosine)
- Solar radiation, surface pressure, precipitation

**Vegetation Features** (Temporal)
- NDVI (Normalized Difference Vegetation Index)
- LAI (Leaf Area Index)
- Soil Moisture Index

**Target Variable**
- `burned_areas`: Binary mask showing fire spread over 10 timesteps (64×64×10)

## Feature Engineering

The preprocessing pipeline computes derived features to enhance prediction accuracy:

1. **Terrain Ruggedness Index (TRI)** - Quantifies terrain complexity
2. **Vapor Pressure Deficit (VPD)** - Atmospheric dryness indicator
3. **Fuel Moisture Proxies** - Derived from temperature, humidity, and precipitation
4. **Vegetation Stress Indices** - Combined NDVI/temperature metrics
5. **Spatial Statistics** - Neighborhood aggregations (3×3, 5×5, 7×7 windows)
6. **Temporal Aggregations** - Rolling means and rate of change
7. **Normalization** - Z-score standardization per feature
8. **Data Augmentation** - Spatial flips and rotations

## Model Architectures

### 1. 3D U-Net
Spatiotemporal convolutional architecture for volumetric prediction

**Architecture:**
- **Encoder**: 4 downsampling blocks with 3D convolutions (64→128→256→512 channels)
- **Bottleneck**: Double 3D convolution at lowest resolution
- **Decoder**: 4 upsampling blocks with skip connections (512→256→128→64→32 channels)
- **Output**: 3D convolution + sigmoid activation (10 timesteps × 64×64 spatial grid)

**Key Features:**
- Batch normalization and ReLU activations
- Skip connections preserve spatial details
- 3D max pooling for downsampling, transposed convolutions for upsampling
- Processes entire temporal sequence in parallel

### 2. ConvLSTM
Recurrent architecture preserving spatial structure while modeling temporal dependencies

**Architecture:**
- **Initial Conv Block**: 2D convolution (input channels → 64)
- **ConvLSTM Layers**: 2 stacked layers (64→128 hidden channels)
  - Convolutional gates (input, forget, output, cell)
  - Spatial 3×3 kernels maintain spatial relationships
- **Decoder**: 2D convolutions (128→64→32→1)
- **Output**: Sigmoid activation for binary fire prediction

**Key Features:**
- Processes timesteps sequentially, maintaining hidden/cell states
- Preserves spatial structure through convolutional operations
- Better captures temporal dynamics and fire progression patterns

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Wildfire-Spread-Simulation.git
cd Wildfire-Spread-Simulation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Models

**Train 3D U-Net:**
```bash
python src/training/train_unet3d.py
```

**Train ConvLSTM:**
```bash
python src/training/train_lstm.py
```

### Running Simulation

Generate a wildfire spread simulation GIF with dual model predictions:

```bash
python src/simulation/create_simulation_gif.py
```

**Configuration:**
- Models: Trained U-Net and LSTM checkpoints from `trained_models/`
- Normalization stats: `trained_models/unet3d/normalization_stats.json`
- Output: `results/simulation_2000_steps.gif`
- Grid size: 64×64
- Timesteps: 2000 (configurable)

The simulation visualizes:
- **Environmental inputs** (top 2 rows): VPD, DEM, fire history, curvature, humidity, NDVI, solar radiation, wind speed, temperature, soil moisture
- **Model predictions** (bottom row): U-Net and LSTM fire spread predictions side-by-side

## Results

### Model Performance

Both models were trained for 50 epochs on the MesoGEOS wildfire dataset. Performance metrics are evaluated on the validation set at the best epoch (minimum validation loss).

![Confusion Matrices](results/confusion_matrices.png)

#### 3D U-Net (Epoch 50)
- **Precision**: 0.9616 - High accuracy when predicting fire presence (few false positives)
- **Recall**: 0.9973 - Excellent at detecting actual fire spread (captures almost all fire cells)
- **F1 Score**: 0.9791 - Outstanding balance between precision and recall
- **IoU**: 0.9590 - Very high overlap between predicted and actual burned areas

#### ConvLSTM (Epoch 50)
- **Precision**: 0.2855 - More false positives (predicts fire in non-burning areas)
- **Recall**: 0.9327 - Good at detecting actual fire cells
- **F1 Score**: 0.4369 - Moderate overall performance
- **IoU**: 0.2799 - Lower spatial overlap with ground truth

### Key Findings

1. **U-Net Superior Performance**: The 3D U-Net significantly outperforms ConvLSTM across all metrics, achieving near-perfect recall (99.73%) and high precision (96.16%).

2. **Precision-Recall Trade-off**: ConvLSTM shows high recall (93.27%) but low precision (28.55%), indicating it tends to overpredict fire spread but successfully captures most actual fire events.

3. **Spatial Accuracy**: U-Net's IoU of 95.90% demonstrates excellent spatial alignment with ground truth, while ConvLSTM's 27.99% IoU suggests difficulty in precise boundary prediction.

4. **Architecture Insights**:
   - U-Net's skip connections and parallel 3D convolutions effectively capture spatiotemporal patterns
   - ConvLSTM's sequential processing may require longer training or architecture adjustments to match U-Net performance

---

*Built with PyTorch and NumPy*