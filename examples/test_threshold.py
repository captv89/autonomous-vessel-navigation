"""Test to visualize the threshold concept."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.grid_world import GridWorld
import numpy as np

# Create small grid
grid = GridWorld(10, 10)

# Set different values manually
grid.grid[2, 2] = 0.0  # Safe
grid.grid[2, 3] = 0.3  # Still safe
grid.grid[2, 4] = 0.49 # Just barely safe
grid.grid[2, 5] = 0.5  # Obstacle (at threshold)
grid.grid[2, 6] = 0.7  # Obstacle
grid.grid[2, 7] = 1.0  # Full obstacle

print("Grid row 2, columns 2-7:")
print("Value  | Is Valid?")
print("-" * 20)
for col in range(2, 8):
    value = grid.grid[2, col]
    valid = grid.is_valid(col, 2)
    print(f"{value:.2f}   | {valid}")

print("\nThreshold is 0.5:")
print("< 0.5  = Valid (navigable)")
print(">= 0.5 = Invalid (obstacle)")


