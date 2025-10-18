"""
Prototype wildfire simulation with matplotlib visualization.

Click anywhere on the grid to start a fire and watch the model predictions.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from simulation.simulation_engine import WildfireSimulation


class WildfirePrototype:
    """
    Interactive prototype for wildfire spread simulation.

    Features:
    - Click to ignite fires
    - Real-time model predictions
    - Side-by-side terrain and fire visualization
    """

    def __init__(self, model_path, stats_path, lstm_model_path=None, grid_size=64):
        """
        Initialize prototype simulation.

        Args:
            model_path: Path to trained UNET model
            stats_path: Path to normalization stats
            lstm_model_path: Path to trained LSTM model (optional)
            grid_size: Size of grid (default 64)
        """
        print("=" * 80)
        print("WILDFIRE SPREAD SIMULATION - PROTOTYPE")
        print("=" * 80)

        # Initialize simulation engine
        self.simulation = WildfireSimulation(
            model_path=model_path,
            stats_path=stats_path,
            lstm_model_path=lstm_model_path,
            grid_size=grid_size
        )

        # Animation state
        self.is_playing = False
        self.animation = None

        # Current variable to display in left panel
        self.current_variable = 'Elevation'

        # Setup visualization
        self._setup_visualization()

        print("\n" + "=" * 80)
        print("INSTRUCTIONS")
        print("=" * 80)
        print("  • Use radio buttons on LEFT to select input variable to view")
        print("  • Click on the FIRE panels to ignite fires")
        print("  • Press SPACEBAR to play/pause simulation")
        print("  • Press 'R' to reset simulation")
        print("  • Close window to exit")
        print("=" * 80)

    def _setup_visualization(self):
        """Setup matplotlib figure and axes."""
        # Get initial state to determine layout
        state = self.simulation.get_current_state()
        has_lstm = state['has_lstm']

        if has_lstm:
            # Create figure with 4 panels: radio buttons | terrain | UNET fire | LSTM fire
            self.fig = plt.figure(figsize=(20, 6), facecolor='black')
            gs = self.fig.add_gridspec(1, 4, width_ratios=[1, 5, 5, 5], wspace=0.3)

            # Radio buttons axis (left)
            self.ax_radio = self.fig.add_subplot(gs[0], facecolor='black')
            self.ax_radio.axis('off')

            # Terrain/variable plot (center-left)
            self.ax_terrain = self.fig.add_subplot(gs[1], facecolor='black')

            # UNET fire plot (center-right)
            self.ax_fire = self.fig.add_subplot(gs[2], facecolor='black')

            # LSTM fire plot (right)
            self.ax_fire_lstm = self.fig.add_subplot(gs[3], facecolor='black')
        else:
            # Create figure with 3 panels: radio buttons | terrain | fire
            self.fig = plt.figure(figsize=(16, 6), facecolor='black')
            gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 5, 5], wspace=0.3)

            # Radio buttons axis (left)
            self.ax_radio = self.fig.add_subplot(gs[0], facecolor='black')
            self.ax_radio.axis('off')

            # Terrain/variable plot (center)
            self.ax_terrain = self.fig.add_subplot(gs[1], facecolor='black')

            # Fire plot (right)
            self.ax_fire = self.fig.add_subplot(gs[2], facecolor='black')
            self.ax_fire_lstm = None

        # Left panel: Variable display (initially elevation)
        var_data, var_cmap, var_label = self._get_variable_data('Elevation')
        self.terrain_img = self.ax_terrain.imshow(
            var_data,
            cmap=var_cmap,
            interpolation='nearest'
        )
        self.ax_terrain.set_title('Terrain Elevation (m)', fontsize=14, fontweight='bold', color='white')
        self.ax_terrain.set_xlabel('X coordinate', color='white')
        self.ax_terrain.set_ylabel('Y coordinate', color='white')
        self.ax_terrain.tick_params(colors='white')
        self.terrain_colorbar = plt.colorbar(self.terrain_img, ax=self.ax_terrain, label=var_label)
        self.terrain_colorbar.ax.yaxis.set_tick_params(color='white')
        self.terrain_colorbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(self.terrain_colorbar.ax.axes, 'yticklabels'), color='white')

        # UNET fire panel
        # Use vmax=0.2 to make fires more visible (most fires are 0.005-0.15 probability)
        self.fire_img = self.ax_fire.imshow(
            state['fire_prob'],
            cmap='hot',
            vmin=0,
            vmax=0.2,  # Adjusted from 1.0 to show low probabilities as brighter
            interpolation='nearest'
        )
        self.ax_fire.set_title('Fire Probability (UNET)', fontsize=14, fontweight='bold', color='white')
        self.ax_fire.set_xlabel('X coordinate', color='white')
        self.ax_fire.set_ylabel('Y coordinate', color='white')
        self.ax_fire.tick_params(colors='white')
        fire_colorbar = plt.colorbar(self.fire_img, ax=self.ax_fire, label='Probability')
        fire_colorbar.ax.yaxis.set_tick_params(color='white')
        fire_colorbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(fire_colorbar.ax.axes, 'yticklabels'), color='white')

        # LSTM fire panel (if available)
        if has_lstm:
            self.fire_img_lstm = self.ax_fire_lstm.imshow(
                state['fire_prob_lstm'],
                cmap='hot',
                vmin=0,
                vmax=0.2,
                interpolation='nearest'
            )
            self.ax_fire_lstm.set_title('Fire Probability (LSTM)', fontsize=14, fontweight='bold', color='white')
            self.ax_fire_lstm.set_xlabel('X coordinate', color='white')
            self.ax_fire_lstm.set_ylabel('Y coordinate', color='white')
            self.ax_fire_lstm.tick_params(colors='white')
            fire_colorbar_lstm = plt.colorbar(self.fire_img_lstm, ax=self.ax_fire_lstm, label='Probability')
            fire_colorbar_lstm.ax.yaxis.set_tick_params(color='white')
            fire_colorbar_lstm.ax.yaxis.label.set_color('white')
            plt.setp(plt.getp(fire_colorbar_lstm.ax.axes, 'yticklabels'), color='white')
        else:
            self.fire_img_lstm = None

        # Create radio buttons for variable selection
        variable_options = [
            'Elevation',
            'Slope',
            'NDVI',
            'Temperature',
            'Humidity',
            'Wind Speed',
            'Wind Direction',
            'Soil Moisture',
            'LAI',
            'Solar Radiation'
        ]

        # Position radio buttons
        radio_ax = plt.axes([0.02, 0.3, 0.10, 0.5], facecolor='black')
        self.radio = RadioButtons(radio_ax, variable_options, active=0)
        self.radio.on_clicked(self._on_variable_change)

        # Style radio buttons for black background
        for label in self.radio.labels:
            label.set_color('white')
        # Style radio button circles (if they exist in this matplotlib version)
        if hasattr(self.radio, 'circles'):
            for circle in self.radio.circles:
                circle.set_edgecolor('white')

        # Add statistics text
        self.stats_text = self.fig.text(
            0.5, 0.02,
            self._format_stats(state),
            ha='center',
            fontsize=12,
            color='white',
            bbox=dict(boxstyle='round', facecolor='darkslategray', alpha=0.8, edgecolor='white')
        )

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        plt.subplots_adjust(bottom=0.1, left=0.15)

    def _get_variable_data(self, variable_name):
        """
        Get data, colormap, and label for a given variable.

        Args:
            variable_name: Name of variable to display

        Returns:
            tuple: (data, colormap, label)
        """
        env = self.simulation.environment

        variable_map = {
            'Elevation': (env.dem, 'terrain', 'Elevation (m)'),
            'Slope': (env.slope, 'YlOrRd', 'Slope (degrees)'),
            'NDVI': (env.ndvi, 'YlGn', 'NDVI'),
            'Temperature': (env.temperature - 273.15, 'RdYlBu_r', 'Temperature (°C)'),
            'Humidity': (env.humidity * 100, 'Blues', 'Humidity (%)'),
            'Wind Speed': (env.wind_speed, 'viridis', 'Wind Speed (m/s)'),
            'Wind Direction': (env.wind_direction, 'twilight', 'Wind Direction (degrees)'),
            'Soil Moisture': (env.soil_moisture * 100, 'BrBG', 'Soil Moisture (%)'),
            'LAI': (env.lai, 'Greens', 'LAI (Leaf Area Index)'),
            'Solar Radiation': (env.solar_radiation / 1e6, 'hot', 'Solar Radiation (MJ/m²)')
        }

        return variable_map.get(variable_name, variable_map['Elevation'])

    def _on_variable_change(self, label):
        """
        Handle variable selection change from radio buttons.

        Args:
            label: Selected variable name
        """
        self.current_variable = label

        # Get new variable data
        var_data, var_cmap, var_label = self._get_variable_data(label)

        # Update image data and colormap
        self.terrain_img.set_data(var_data)
        self.terrain_img.set_cmap(var_cmap)

        # Update colorbar limits
        self.terrain_img.autoscale()

        # Update title
        self.ax_terrain.set_title(f'{label}', fontsize=14, fontweight='bold', color='white')

        # Update colorbar
        self.terrain_colorbar.update_normal(self.terrain_img)
        self.terrain_colorbar.set_label(var_label)
        # Maintain white styling for colorbar
        self.terrain_colorbar.ax.yaxis.set_tick_params(color='white')
        self.terrain_colorbar.ax.yaxis.label.set_color('white')
        plt.setp(plt.getp(self.terrain_colorbar.ax.axes, 'yticklabels'), color='white')

        # Redraw
        self.fig.canvas.draw_idle()

        print(f"📊 Switched to viewing: {label}")

    def _format_stats(self, state):
        """Format statistics text."""
        return (f"Timestep: {state['timestep']} | "
                f"Burned Cells: {int(state['burned_area'])} | "
                f"Playing: {self.is_playing}")

    def _on_click(self, event):
        """Handle mouse click to ignite fire."""
        print(f"\n[DEBUG] Click detected: inaxes={event.inaxes}, button={event.button}")

        # Respond to clicks on either fire panel
        clicked_fire_panel = (event.inaxes == self.ax_fire or
                             (self.ax_fire_lstm is not None and event.inaxes == self.ax_fire_lstm))

        if clicked_fire_panel and event.button == 1:  # Left click
            x = int(event.xdata + 0.5)
            y = int(event.ydata + 0.5)

            print(f"[DEBUG] Click on fire panel at grid position ({x}, {y})")

            # Ignite fire
            self.simulation.ignite_fire(x, y)

            # Only run a step if simulation is NOT playing
            # If playing, the animation loop will pick up the ignition automatically
            if not self.is_playing:
                print(f"[DEBUG] Running simulation step (paused mode)...")
                # Run one simulation step to show the fire immediately
                fire_prob = self.simulation.step()

                print(f"[DEBUG] Fire probability stats: min={fire_prob.min():.4f}, max={fire_prob.max():.4f}, mean={fire_prob.mean():.4f}")
                print(f"[DEBUG] Cells > 0.15: {(fire_prob > 0.15).sum()} / 4096")

                # Update fire visualizations
                self.fire_img.set_data(fire_prob)
                if self.fire_img_lstm is not None:
                    state = self.simulation.get_current_state()
                    self.fire_img_lstm.set_data(state['fire_prob_lstm'])

                # Update statistics
                state = self.simulation.get_current_state()
                self.stats_text.set_text(self._format_stats(state))

                print(f"[DEBUG] Updated visualization")
            else:
                print(f"[DEBUG] Simulation is playing, ignition will appear on next frame")

            # Highlight clicked cell briefly on the clicked panel
            rect = Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                linewidth=2,
                edgecolor='cyan',
                facecolor='none'
            )
            event.inaxes.add_patch(rect)

            # Draw everything
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # Remove highlight after a moment
            import time
            time.sleep(0.2)
            rect.remove()
            self.fig.canvas.draw()

            burned = self.simulation.get_current_state()['burned_area']
            status = "added to simulation" if self.is_playing else f"{int(burned)} cells burning"
            print(f"🔥 Fire ignited at ({x}, {y}) - {status}")
        else:
            if event.inaxes == self.ax_terrain:
                print("[INFO] ⚠️  You clicked on the TERRAIN panel (left). Click on the FIRE panel instead!")
            elif event.inaxes is None:
                print("[INFO] Click was outside the panels")
            else:
                print(f"[INFO] Click not on fire panel or not left button")

    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == ' ':  # Spacebar: play/pause
            self.toggle_play()
        elif event.key == 'r':  # R: reset
            self.reset()

    def toggle_play(self):
        """Toggle play/pause state."""
        self.is_playing = not self.is_playing

        if self.is_playing:
            # Start animation
            if self.animation is None or not self.animation.event_source:
                self.animation = FuncAnimation(
                    self.fig,
                    self._update_frame,
                    interval=100,  # 500ms = 0.1 seconds per step
                    blit=False,
                    cache_frame_data=False
                )
            print("▶ Simulation playing")
        else:
            # Stop animation
            if self.animation is not None:
                self.animation.event_source.stop()
            print("⏸ Simulation paused")

        # Update display
        state = self.simulation.get_current_state()
        self.stats_text.set_text(self._format_stats(state))
        self.fig.canvas.draw()

    def reset(self):
        """Reset simulation."""
        # Stop animation
        if self.animation is not None:
            self.animation.event_source.stop()
        self.is_playing = False

        # Reset simulation
        self.simulation.reset()

        # Update visualization
        self._update_frame(0)

    def _update_frame(self, frame):
        """
        Update animation frame.

        Args:
            frame: Frame number (unused)
        """
        if self.is_playing:
            # Run simulation step
            fire_prob = self.simulation.step()

            # Update fire visualizations
            self.fire_img.set_data(fire_prob)

            # Get state for LSTM update
            state = self.simulation.get_current_state()

            if self.fire_img_lstm is not None:
                self.fire_img_lstm.set_data(state['fire_prob_lstm'])

            # Update left panel (in case displaying dynamic variables)
            var_data, var_cmap, var_label = self._get_variable_data(self.current_variable)
            self.terrain_img.set_data(var_data)

            # Update statistics
            self.stats_text.set_text(self._format_stats(state))

            # Print progress
            print(f"Step {state['timestep']}: {int(state['burned_area'])} cells burned")

        return_list = [self.fire_img, self.terrain_img, self.stats_text]
        if self.fire_img_lstm is not None:
            return_list.append(self.fire_img_lstm)

        return return_list

    def run(self):
        """Start the interactive visualization."""
        plt.show()


def main():
    """Main entry point."""
    # Configuration
    MODEL_PATH = Path(__file__).parent.parent.parent / 'prod_models' / 'UNET3D' / '3D-UNET-WILDFIRE-1.pt'
    LSTM_MODEL_PATH = Path(__file__).parent.parent.parent / 'prod_models' / 'LSTM' / 'LSTM.pt'
    STATS_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'normalization_stats.json'

    # Check files exist
    if not MODEL_PATH.exists():
        print(f"❌ Error: UNET model not found at {MODEL_PATH}")
        return

    if not STATS_PATH.exists():
        print(f"❌ Error: Normalization stats not found at {STATS_PATH}")
        return

    # Check if LSTM model exists
    if LSTM_MODEL_PATH.exists():
        print(f"✓ Found LSTM model at {LSTM_MODEL_PATH}")
        lstm_path = LSTM_MODEL_PATH
    else:
        print(f"⚠️  LSTM model not found at {LSTM_MODEL_PATH}, running with UNET only")
        lstm_path = None

    # Create and run prototype
    prototype = WildfirePrototype(
        model_path=MODEL_PATH,
        stats_path=STATS_PATH,
        lstm_model_path=lstm_path,
        grid_size=64
    )

    prototype.run()


if __name__ == '__main__':
    main()
