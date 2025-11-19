"""
GridWorld - Represents the navigation environment as a 2D grid.

Each cell in the grid can be:
- 0: Navigable water (safe)
- 1: Obstacle (island, shallow water, no-go zone)
- Values between 0-1 can represent different danger levels
"""

import numpy as np
from typing import Tuple, List, Optional
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Obstacle inflation will use fallback method.")
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Obstacle inflation will use fallback method.")
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Obstacle inflation will use fallback method.")


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
        
        # Inflated grid for path planning (safety buffer around obstacles)
        self._inflated_grid = None
        self._inflation_radius = 0
        
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
    
    def inflate_obstacles(self, radius_cells: int = 2) -> np.ndarray:
        """
        Create a safety buffer around obstacles (costmap).
        Any cell within 'radius_cells' of an obstacle becomes an obstacle.
        
        This is used for path planning to keep vessels away from walls.
        
        Args:
            radius_cells: Buffer radius in cells (e.g., 2 cells = 20m for cell_size=10m)
        
        Returns:
            Inflated grid with safety buffer
        
        Example:
            If a vessel is 20m long and cell_size is 10m, use radius_cells=2
            to maintain a 2-cell (20m) safety margin around obstacles.
        """
        # Cache the inflated grid if radius hasn't changed
        if self._inflated_grid is not None and self._inflation_radius == radius_cells:
            return self._inflated_grid
        
        if radius_cells <= 0:
            # No inflation
            self._inflated_grid = self.grid.copy()
            self._inflation_radius = 0
            return self._inflated_grid
        
        # Create binary mask of obstacles
        obstacles = self.grid > 0.5
        
        if SCIPY_AVAILABLE:
            # Use scipy for efficient morphological dilation
            structure = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1))
            buffered = ndimage.binary_dilation(obstacles, structure=structure).astype(np.float32)
        else:
            # Fallback: manual dilation (slower but works without scipy)
            buffered = obstacles.astype(np.float32)
            temp = obstacles.copy()
            
            for _ in range(radius_cells):
                # Dilate by checking neighbors
                new_obstacles = temp.copy()
                for y in range(self.height):
                    for x in range(self.width):
                        if temp[y, x]:
                            # Mark all neighbors as obstacles
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < self.height and 0 <= nx < self.width:
                                        new_obstacles[ny, nx] = True
                temp = new_obstacles
            buffered = temp.astype(np.float32)
        
        # Cache the result
        self._inflated_grid = buffered
        self._inflation_radius = radius_cells
        
        # Print statistics
        original_obstacles = np.sum(self.grid > 0.5)
        inflated_obstacles = np.sum(buffered > 0.5)
        print(f"Inflated obstacles: {original_obstacles} → {inflated_obstacles} cells "
              f"(+{inflated_obstacles - original_obstacles} safety buffer, radius={radius_cells})")
        
        return self._inflated_grid
    
    def get_planning_grid(self, safety_margin_cells: int = 2) -> np.ndarray:
        """
        Get the grid to use for path planning with safety margins.
        
        Args:
            safety_margin_cells: Safety buffer radius in cells (0 = no buffer)
        
        Returns:
            Grid with inflated obstacles for safe path planning
        """
        if safety_margin_cells <= 0:
            return self.grid
        return self.inflate_obstacles(safety_margin_cells)
    
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