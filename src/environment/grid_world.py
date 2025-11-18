"""
GridWorld - Represents the navigation environment as a 2D grid.

Each cell in the grid can be:
- 0: Navigable water (safe)
- 1: Obstacle (island, shallow water, no-go zone)
- Values between 0-1 can represent different danger levels
"""

import numpy as np
from typing import Tuple, List, Optional


class GridWorld:
    """
    A 2D grid environment for vessel navigation.
    
    Attributes:
        width: Number of cells in x-direction
        height: Number of cells in y-direction
        cell_size: Size of each cell in meters (for real-world scale)
        grid: 2D numpy array representing the environment
    """
    
    def __init__(self, width: int = 100, height: int = 100, cell_size: float = 10.0):
        
        """
        Initialize the grid world.
        
        Args:
            width: Grid width in cells
            height: Grid height in cells
            cell_size: Size of each cell in meters
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Initialize empty grid (all navigable water)
        self.grid = np.zeros((height, width), dtype=np.float32)
        
        print(f"Created GridWorld: {width}x{height} cells, {cell_size}m per cell")
        print(f"Total area: {width * cell_size}m x {height * cell_size}m")
    
    def add_obstacle(self, x: int, y: int, width: int = 1, height: int = 1):
        
        """
        Add a rectangular obstacle to the grid.
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Width of obstacle in cells
            height: Height of obstacle in cells
        """
        # Ensure coordinates are within bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        x_end = min(x + width, self.width)
        y_end = min(y + height, self.height)
        
        # Mark cells as obstacles (value = 1)
        self.grid[y:y_end, x:x_end] = 1.0
        print(f"Added obstacle at ({x}, {y}) with size {width}x{height}")
    
    def add_circular_obstacle(self, center_x: int, center_y: int, radius: int):
        
        """
        Add a circular obstacle (like an island).
        
        Args:
            center_x: Center x coordinate
            center_y: Center y coordinate
            radius: Radius in cells
        """
        for i in range(self.height):
            for j in range(self.width):
                distance = np.sqrt((j - center_x)**2 + (i - center_y)**2)
                if distance <= radius:
                    self.grid[i, j] = 1.0
        
        print(f"Added circular obstacle at ({center_x}, {center_y}) with radius {radius}")
    
    def is_valid(self, x: int, y: int) -> bool:
        
        """
        Check if a cell is valid (within bounds and navigable).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if cell is valid and navigable
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.grid[y, x] < 0.5  # Cells with value < 0.5 are navigable
    
    def get_neighbors(self, x: int, y: int, allow_diagonal: bool = True) -> List[Tuple[int, int]]:
        
        """
        Get valid neighboring cells.
        
        Args:
            x: X coordinate
            y: Y coordinate
            allow_diagonal: Whether to include diagonal neighbors
            
        Returns:
            List of (x, y) tuples for valid neighbors
        """
        neighbors = []
        
        # 4-directional neighbors (N, E, S, W)
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        # Add diagonal directions if allowed
        if allow_diagonal:
            directions += [(1, -1), (1, 1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def clear(self):
        
        """Clear all obstacles from the grid."""
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        print("Grid cleared")
    
    def get_random_free_position(self) -> Tuple[int, int]:
        
        """
        Get a random navigable position in the grid.
        
        Returns:
            (x, y) tuple of a free position
        """
        free_cells = np.argwhere(self.grid < 0.5)
        if len(free_cells) == 0:
            raise ValueError("No free cells available in grid")
        
        idx = np.random.randint(0, len(free_cells))
        y, x = free_cells[idx]
        return int(x), int(y)
    
    def __repr__(self) -> str:
        """String representation of the grid."""
        obstacle_count = np.sum(self.grid >= 0.5)
        free_count = np.sum(self.grid < 0.5)
        return f"GridWorld({self.width}x{self.height}, obstacles={obstacle_count}, free={free_count})"


# Simple test function
def test_grid_world():
    """Test the GridWorld class."""
    print("Testing GridWorld...\n")
    
    # Create a small grid
    grid = GridWorld(width=20, height=20, cell_size=10.0)
    
    # Add some obstacles
    grid.add_obstacle(5, 5, width=3, height=3)
    grid.add_circular_obstacle(15, 15, radius=2)
    
    # Test validity checks
    print(f"\nIs (5, 5) valid? {grid.is_valid(5, 5)}")  # Should be False (obstacle)
    print(f"Is (0, 0) valid? {grid.is_valid(0, 0)}")    # Should be True (free)
    
    # Get neighbors
    neighbors = grid.get_neighbors(10, 10)
    print(f"\nNeighbors of (10, 10): {len(neighbors)} cells")
    
    # Get random position
    rand_pos = grid.get_random_free_position()
    print(f"Random free position: {rand_pos}")
    
    print(f"\n{grid}")
    print("\n✓ GridWorld test complete!")


if __name__ == "__main__":
    test_grid_world()