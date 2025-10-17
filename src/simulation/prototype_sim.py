"""
Prototype wildfire simulation with matplotlib visualization.

Click anywhere on the grid to start a fire and watch the model predictions.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
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

    def __init__(self, model_path, stats_path, grid_size=64):
        """
        Initialize prototype simulation.

        Args:
            model_path: Path to trained model
            stats_path: Path to normalization stats
            grid_size: Size of grid (default 64)
        """
        print("=" * 80)
        print("WILDFIRE SPREAD SIMULATION - PROTOTYPE")
        print("=" * 80)

        # Initialize simulation engine
        self.simulation = WildfireSimulation(
            model_path=model_path,
            stats_path=stats_path,
            grid_size=grid_size
        )

        # Animation state
        self.is_playing = False
        self.animation = None

        # Setup visualization
        self._setup_visualization()

        print("\n" + "=" * 80)
        print("INSTRUCTIONS")
        print("=" * 80)
        print("  • Click on the RIGHT panel to ignite fires")
        print("  • Press SPACEBAR to play/pause simulation")
        print("  • Press 'R' to reset simulation")
        print("  • Close window to exit")
        print("=" * 80)

    def _setup_visualization(self):
        """Setup matplotlib figure and axes."""
        # Create figure with two subplots
        self.fig, (self.ax_terrain, self.ax_fire) = plt.subplots(
            1, 2, figsize=(14, 6)
        )

        # Get initial state
        state = self.simulation.get_current_state()

        # Left panel: Terrain elevation
        self.terrain_img = self.ax_terrain.imshow(
            state['terrain'],
            cmap='terrain',
            interpolation='nearest'
        )
        self.ax_terrain.set_title('Terrain Elevation (m)', fontsize=14, fontweight='bold')
        self.ax_terrain.set_xlabel('X coordinate')
        self.ax_terrain.set_ylabel('Y coordinate')
        plt.colorbar(self.terrain_img, ax=self.ax_terrain, label='Elevation (m)')

        # Right panel: Fire spread
        self.fire_img = self.ax_fire.imshow(
            state['fire_prob'],
            cmap='hot',
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )
        self.ax_fire.set_title('Fire Probability', fontsize=14, fontweight='bold')
        self.ax_fire.set_xlabel('X coordinate')
        self.ax_fire.set_ylabel('Y coordinate')
        plt.colorbar(self.fire_img, ax=self.ax_fire, label='Probability')

        # Add statistics text
        self.stats_text = self.fig.text(
            0.5, 0.02,
            self._format_stats(state),
            ha='center',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

    def _format_stats(self, state):
        """Format statistics text."""
        return (f"Timestep: {state['timestep']} | "
                f"Burned Cells: {int(state['burned_area'])} | "
                f"Playing: {self.is_playing}")

    def _on_click(self, event):
        """Handle mouse click to ignite fire."""
        # Only respond to clicks on the fire panel
        if event.inaxes == self.ax_fire and event.button == 1:  # Left click
            x = int(event.xdata + 0.5)
            y = int(event.ydata + 0.5)

            # Ignite fire
            self.simulation.ignite_fire(x, y)

            # Update visualization immediately
            self._update_frame(0)

            # Highlight clicked cell briefly
            rect = Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                linewidth=2,
                edgecolor='cyan',
                facecolor='none'
            )
            self.ax_fire.add_patch(rect)
            self.fig.canvas.draw()

            # Remove highlight after a moment
            self.fig.canvas.flush_events()
            import time
            time.sleep(0.1)
            rect.remove()
            self.fig.canvas.draw()

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
                    interval=500,  # 500ms = 0.5 seconds per step
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

            # Update fire visualization
            self.fire_img.set_data(fire_prob)

            # Update statistics
            state = self.simulation.get_current_state()
            self.stats_text.set_text(self._format_stats(state))

            # Print progress
            print(f"Step {state['timestep']}: {int(state['burned_area'])} cells burned")

        return [self.fire_img, self.stats_text]

    def run(self):
        """Start the interactive visualization."""
        plt.show()


def main():
    """Main entry point."""
    # Configuration
    MODEL_PATH = Path(__file__).parent.parent.parent / 'prod_models' / 'UNET3D' / '3D-UNET-WILDFIRE-1.pt'
    STATS_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'normalization_stats.json'

    # Check files exist
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    if not STATS_PATH.exists():
        print(f"❌ Error: Normalization stats not found at {STATS_PATH}")
        return

    # Create and run prototype
    prototype = WildfirePrototype(
        model_path=MODEL_PATH,
        stats_path=STATS_PATH,
        grid_size=64
    )

    prototype.run()


if __name__ == '__main__':
    main()
