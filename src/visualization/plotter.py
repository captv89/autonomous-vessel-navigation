"""
Plotter - Visualization tools for the grid world and vessel navigation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from typing import Optional, List, Tuple


class Plotter:
    """
    Handles visualization of the grid world, obstacles, and paths.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10)):
        """
        Initialize the plotter.
        
        Args:
            figsize: Figure size in inches (width, height)
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def plot_grid(self, grid_world, title: str = "Grid World", 
                  start: Optional[Tuple[int, int]] = None,
                  goal: Optional[Tuple[int, int]] = None,
                  path: Optional[List[Tuple[int, int]]] = None,
                  show: bool = True):
        """
        Plot the grid world with obstacles.
        
        Args:
            grid_world: GridWorld object to visualize
            title: Plot title
            start: Optional start position (x, y)
            goal: Optional goal position (x, y)
            path: Optional path as list of (x, y) tuples
            show: Whether to display the plot immediately
        """
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        # Plot the grid
        # Use 'Blues' colormap: white = navigable, dark blue = obstacle
        im = self.ax.imshow(grid_world.grid, cmap='Blues', origin='lower', 
                           vmin=0, vmax=1, interpolation='nearest')
        
        # Add grid lines for better visibility
        self.ax.set_xticks(np.arange(-0.5, grid_world.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, grid_world.height, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Plot start position (green circle)
        if start is not None:
            self.ax.plot(start[0], start[1], 'go', markersize=15, 
                        label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Plot goal position (red star)
        if goal is not None:
            self.ax.plot(goal[0], goal[1], 'r*', markersize=20, 
                        label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        
        # Plot path (yellow line)
        if path is not None and len(path) > 0:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            self.ax.plot(path_x, path_y, 'y-', linewidth=3, 
                        label='Path', alpha=0.7)
            self.ax.plot(path_x, path_y, 'yo', markersize=4, alpha=0.5)
        
        # Set labels and title
        self.ax.set_xlabel('X (cells)', fontsize=12)
        self.ax.set_ylabel('Y (cells)', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend if we have start/goal/path
        if start is not None or goal is not None or path is not None:
            self.ax.legend(loc='upper right', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        cbar.set_label('Obstacle Density', fontsize=10)
        
        # Set aspect ratio to equal
        self.ax.set_aspect('equal')
        
        # Tight layout
        plt.tight_layout()
        
        if show:
            plt.show()
    
    def save_plot(self, filename: str, dpi: int = 300):
        """
        Save the current plot to a file.
        
        Args:
            filename: Output filename (e.g., 'grid.png')
            dpi: Resolution in dots per inch
        """
        if self.fig is None:
            print("No plot to save. Create a plot first.")
            return
        
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    def close(self):
        """Close the current plot."""
        if self.fig is not None:
            plt.close(self.fig)


def demo_visualization():
    """Demonstrate the visualization capabilities."""
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.environment.grid_world import GridWorld
    
    print("Creating visualization demo...\n")
    
    # Create a grid world
    grid = GridWorld(width=50, height=50, cell_size=10.0)
    
    # Add various obstacles
    print("Adding obstacles...")
    
    # Add some rectangular obstacles (like docks or structures)
    grid.add_obstacle(10, 10, width=8, height=3)
    grid.add_obstacle(30, 15, width=5, height=10)
    grid.add_obstacle(5, 35, width=15, height=5)
    
    # Add some circular obstacles (like islands)
    grid.add_circular_obstacle(25, 8, radius=3)
    grid.add_circular_obstacle(40, 35, radius=5)
    grid.add_circular_obstacle(15, 25, radius=4)
    
    # Create plotter
    plotter = Plotter(figsize=(12, 10))
    
    # Define start and goal positions
    start = (5, 5)
    goal = (45, 45)
    
    # Create a simple example path (we'll do real pathfinding later)
    example_path = [(5, 5), (10, 8), (15, 12), (20, 18), (25, 25), 
                    (30, 32), (35, 38), (40, 42), (45, 45)]
    
    # Plot everything
    print("\nGenerating visualization...")
    plotter.plot_grid(
        grid_world=grid,
        title="Autonomous Vessel Navigation - Grid World",
        start=start,
        goal=goal,
        path=example_path,
        show=True
    )
    
    print("\n✓ Visualization complete!")
    print("\nColor legend:")
    print("  - White: Navigable water (safe)")
    print("  - Blue: Obstacles (islands, shallow water, no-go zones)")
    print("  - Green circle: Start position")
    print("  - Red star: Goal position")
    print("  - Yellow line: Example path")


if __name__ == "__main__":
    demo_visualization()