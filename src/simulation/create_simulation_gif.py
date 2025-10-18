"""
Create a GIF of the wildfire simulation showing both UNET and LSTM model predictions.

Runs the simulation for a specified number of timesteps and saves the output as a GIF.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import io

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.simulation_engine import WildfireSimulation


def create_simulation_gif(
    model_path,
    stats_path,
    lstm_model_path,
    output_path,
    num_timesteps=2000,
    grid_size=64,
    ignition_points=None,
    frame_interval=10
):
    """
    Create a GIF of the wildfire simulation with environmental inputs and predictions.

    Layout:
    - Row 1: VPD, DEM, Fire History, Curvature, Humidity
    - Row 2: NDVI, Solar Radiation, Wind Speed, Temperature, Soil Moisture
    - Row 3: UNET Prediction (centered), LSTM Prediction (centered)

    Args:
        model_path: Path to trained UNET model
        stats_path: Path to normalization stats
        lstm_model_path: Path to trained LSTM model
        output_path: Path to save output GIF
        num_timesteps: Number of simulation steps to run
        grid_size: Size of grid (default 64)
        ignition_points: List of (x, y) tuples for initial fire ignitions
        frame_interval: Save a frame every N timesteps (default 10)
    """
    print("=" * 80)
    print("WILDFIRE SIMULATION GIF CREATOR")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Timesteps: {num_timesteps}")
    print(f"Frame interval: {frame_interval} (saving every {frame_interval}th frame)")
    print("=" * 80)

    # Initialize simulation engine
    simulation = WildfireSimulation(
        model_path=model_path,
        stats_path=stats_path,
        lstm_model_path=lstm_model_path,
        grid_size=grid_size
    )

    # Get initial state
    state = simulation.get_current_state()
    has_lstm = state['has_lstm']

    # Default ignition points if none provided
    if ignition_points is None:
        # Start fires in a few locations
        ignition_points = [
            (grid_size // 4, grid_size // 4),
            (3 * grid_size // 4, grid_size // 2),
            (grid_size // 2, 3 * grid_size // 4)
        ]

    print(f"\nIgniting fires at: {ignition_points}")

    # Ignite initial fires
    for x, y in ignition_points:
        simulation.ignite_fire(x, y)

    # List to store frames
    frames = []

    # Create figure with 3 rows
    # Row 1: 5 input variables
    # Row 2: 5 input variables
    # Row 3: 2 predictions (centered and closer together)
    if has_lstm:
        fig = plt.figure(figsize=(20, 12), facecolor='black')
        # Use 10 columns for finer control over prediction spacing
        gs = fig.add_gridspec(3, 10, hspace=0.3, wspace=0.3)

        # Row 1: 5 input panels (each spans 2 columns)
        ax_vpd = fig.add_subplot(gs[0, 0:2])
        ax_dem = fig.add_subplot(gs[0, 2:4])
        ax_fire_hist = fig.add_subplot(gs[0, 4:6])
        ax_curv = fig.add_subplot(gs[0, 6:8])
        ax_humid = fig.add_subplot(gs[0, 8:10])

        # Row 2: 5 input panels (each spans 2 columns)
        ax_ndvi = fig.add_subplot(gs[1, 0:2])
        ax_solar = fig.add_subplot(gs[1, 2:4])
        ax_wind = fig.add_subplot(gs[1, 4:6])
        ax_temp = fig.add_subplot(gs[1, 6:8])
        ax_soil = fig.add_subplot(gs[1, 8:10])

        # Row 3: 2 predictions (centered and closer)
        # Leave 2 columns on each side, use 3 columns each for predictions
        ax_unet = fig.add_subplot(gs[2, 2:5])
        ax_lstm = fig.add_subplot(gs[2, 5:8])
    else:
        fig = plt.figure(figsize=(20, 12), facecolor='black')
        gs = fig.add_gridspec(3, 10, hspace=0.3, wspace=0.3)

        # Row 1: 5 input panels (each spans 2 columns)
        ax_vpd = fig.add_subplot(gs[0, 0:2])
        ax_dem = fig.add_subplot(gs[0, 2:4])
        ax_fire_hist = fig.add_subplot(gs[0, 4:6])
        ax_curv = fig.add_subplot(gs[0, 6:8])
        ax_humid = fig.add_subplot(gs[0, 8:10])

        # Row 2: 5 input panels (each spans 2 columns)
        ax_ndvi = fig.add_subplot(gs[1, 0:2])
        ax_solar = fig.add_subplot(gs[1, 2:4])
        ax_wind = fig.add_subplot(gs[1, 4:6])
        ax_temp = fig.add_subplot(gs[1, 6:8])
        ax_soil = fig.add_subplot(gs[1, 8:10])

        # Row 3: 1 prediction (centered)
        ax_unet = fig.add_subplot(gs[2, 3:7])
        ax_lstm = None

    # Store axes for easy clearing
    input_axes = [ax_vpd, ax_dem, ax_fire_hist, ax_curv, ax_humid,
                  ax_ndvi, ax_solar, ax_wind, ax_temp, ax_soil]

    print(f"\nRunning simulation for {num_timesteps} timesteps...")
    print("Progress: ", end='', flush=True)

    # Run simulation and capture frames
    for timestep in range(num_timesteps):
        # Run simulation step
        fire_prob = simulation.step()
        state = simulation.get_current_state()

        # Save frame at intervals
        if timestep % frame_interval == 0:
            # Get environment data
            env = simulation.environment

            # Clear all axes
            for ax in input_axes:
                ax.clear()
            ax_unet.clear()
            if ax_lstm is not None:
                ax_lstm.clear()

            # Helper function to plot input variable
            def plot_input(ax, data, title, cmap, vmin=None, vmax=None):
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_title(title, fontsize=10, fontweight='bold', color='white')
                ax.axis('off')
                return im

            # Row 1: VPD, DEM, Fire History, Curvature, Humidity
            plot_input(ax_vpd, env.vpd, 'VPD', 'RdYlGn_r', 0, 1)
            plot_input(ax_dem, env.dem, 'DEM (m)', 'terrain', 0, 2000)

            # Fire history - sum over last 10 timesteps from simulation engine
            fire_history = simulation.fire_history.sum(dim=0).cpu().numpy()
            plot_input(ax_fire_hist, fire_history, 'Fire History', 'Reds', 0, 10)

            plot_input(ax_curv, env.curvature, 'Curvature', 'RdBu_r')
            plot_input(ax_humid, env.humidity * 100, 'Humidity (%)', 'Blues', 0, 100)

            # Row 2: NDVI, Solar Radiation, Wind Speed, Temperature, Soil Moisture
            plot_input(ax_ndvi, env.ndvi, 'NDVI', 'YlGn', -0.2, 1)
            plot_input(ax_solar, env.solar_radiation / 1e6, 'Solar Rad (MJ/m²)', 'hot', 0, 16)
            plot_input(ax_wind, env.wind_speed, 'Wind Speed (m/s)', 'viridis', 0, 6)
            plot_input(ax_temp, env.temperature - 273.15, 'Temperature (°C)', 'RdYlBu_r', 0, 50)
            plot_input(ax_soil, env.soil_moisture * 100, 'Soil Moisture (%)', 'BrBG', 0, 100)

            # Row 3: Predictions
            # Plot UNET predictions
            im_unet = ax_unet.imshow(
                fire_prob,
                cmap='hot',
                vmin=0,
                vmax=0.2,
                interpolation='nearest'
            )
            ax_unet.set_title(
                f'UNET Fire Probability',
                fontsize=12,
                fontweight='bold',
                color='white'
            )
            ax_unet.set_xlabel('X coordinate', color='white', fontsize=10)
            ax_unet.set_ylabel('Y coordinate', color='white', fontsize=10)
            ax_unet.tick_params(colors='white', labelsize=8)

            # Plot LSTM predictions
            if has_lstm:
                im_lstm = ax_lstm.imshow(
                    state['fire_prob_lstm'],
                    cmap='hot',
                    vmin=0,
                    vmax=0.2,
                    interpolation='nearest'
                )
                ax_lstm.set_title(
                    f'LSTM Fire Probability',
                    fontsize=12,
                    fontweight='bold',
                    color='white'
                )
                ax_lstm.set_xlabel('X coordinate', color='white', fontsize=10)
                ax_lstm.set_ylabel('Y coordinate', color='white', fontsize=10)
                ax_lstm.tick_params(colors='white', labelsize=8)

            # Add overall title with statistics
            fig.suptitle(
                f'Wildfire Simulation - Timestep: {timestep} | Burned Cells: {int(state["burned_area"])}',
                fontsize=16,
                color='white',
                y=0.98
            )

            # Save frame to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor='black', dpi=100, bbox_inches='tight')
            buf.seek(0)

            # Convert to PIL Image and add to frames
            frame = Image.open(buf)
            frames.append(frame.copy())
            buf.close()

            # Print progress
            if timestep % 100 == 0:
                print(f"{timestep}...", end='', flush=True)

    print(f"\n\nCaptured {len(frames)} frames")

    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )

    plt.close(fig)

    print(f"✓ GIF saved successfully!")
    print(f"  - Total frames: {len(frames)}")
    print(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  - Duration: ~{len(frames) * 0.1:.1f} seconds at 10 fps")
    print("=" * 80)


def main():
    """Main entry point."""
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = PROJECT_ROOT / 'prod_models' / 'UNET3D' / '3D-UNET-WILDFIRE-1.pt'
    LSTM_MODEL_PATH = PROJECT_ROOT / 'prod_models' / 'LSTM' / 'LSTM.pt'
    STATS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'normalization_stats.json'
    OUTPUT_PATH = PROJECT_ROOT / 'results' / 'simulation_2000_steps.gif'

    # Check files exist
    if not MODEL_PATH.exists():
        print(f"❌ Error: UNET model not found at {MODEL_PATH}")
        return

    if not STATS_PATH.exists():
        print(f"❌ Error: Normalization stats not found at {STATS_PATH}")
        return

    if not LSTM_MODEL_PATH.exists():
        print(f"⚠️  LSTM model not found at {LSTM_MODEL_PATH}, will use UNET only")
        lstm_path = None
    else:
        lstm_path = LSTM_MODEL_PATH

    # Create results directory
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create GIF
    create_simulation_gif(
        model_path=MODEL_PATH,
        stats_path=STATS_PATH,
        lstm_model_path=lstm_path,
        output_path=OUTPUT_PATH,
        num_timesteps=2000,
        grid_size=64,
        ignition_points=[(16, 16), (48, 32), (32, 48)],  # Three ignition points
        frame_interval=10  # Save every 10th frame (200 total frames)
    )


if __name__ == '__main__':
    main()
