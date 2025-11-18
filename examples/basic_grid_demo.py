"""
Basic Grid Demo - Demonstrates the grid world and visualization capabilities.

This example creates a maritime navigation scenario with:
- Various obstacles (islands, shallow water zones)
- Start and goal positions
- Basic visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.grid_world import GridWorld
from src.visualization.plotter import Plotter


def create_maritime_scenario():
    """
    Create a realistic maritime navigation scenario.
    
    Returns:
        GridWorld with obstacles representing a coastal area
    """
    print("Creating maritime navigation scenario...")
    print("=" * 60)
    
    # Create a 100x100 grid (1km x 1km area, 10m per cell)
    grid = GridWorld(width=100, height=100, cell_size=10.0)
    
    print("\nAdding obstacles:")
    print("-" * 60)
    
    # Large island on the left
    print("1. Adding main island (left side)...")
    grid.add_circular_obstacle(center_x=20, center_y=50, radius=12)
    
    # Small islands scattered
    print("2. Adding small islands...")
    grid.add_circular_obstacle(center_x=45, center_y=25, radius=5)
    grid.add_circular_obstacle(center_x=60, center_y=70, radius=6)
    grid.add_circular_obstacle(center_x=75, center_y=40, radius=4)
    
    # Shallow water zones (represented as rectangular obstacles)
    print("3. Adding shallow water zones...")
    grid.add_obstacle(x=35, y=55, width=15, height=8)
    grid.add_obstacle(x=50, y=10, width=10, height=12)
    
    # No-go zone (e.g., restricted military area)
    print("4. Adding restricted zone...")
    grid.add_obstacle(x=10, y=10, width=12, height=12)
    
    # Reef or rocky area
    print("5. Adding reef area...")
    grid.add_circular_obstacle(center_x=85, center_y=20, radius=7)
    
    print("-" * 60)
    print(f"\n{grid}")
    
    return grid


def main():
    """Main demonstration function."""
    
    print("\n" + "=" * 60)
    print("AUTONOMOUS VESSEL NAVIGATION - GRID DEMO")
    print("=" * 60)
    
    # Create the scenario
    grid = create_maritime_scenario()
    
    # Define navigation task
    print("\nNavigation Task:")
    print("-" * 60)
    start_pos = (10, 90)  # Top-left (Port A)
    goal_pos = (90, 10)   # Bottom-right (Port B)
    
    print(f"Start Position (Port A): {start_pos}")
    print(f"Goal Position (Port B):  {goal_pos}")
    print(f"Straight-line distance:  {((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2)**0.5:.1f} cells")
    print(f"Actual distance:         ~{((goal_pos[0]-start_pos[0])**2 + (goal_pos[1]-start_pos[1])**2)**0.5 * grid.cell_size / 1000:.2f} km")
    
    # Verify positions are valid
    if not grid.is_valid(start_pos[0], start_pos[1]):
        print("\n⚠️  Warning: Start position is in an obstacle!")
    if not grid.is_valid(goal_pos[0], goal_pos[1]):
        print("\n⚠️  Warning: Goal position is in an obstacle!")
    
    # Create visualization
    print("\n" + "=" * 60)
    print("Generating visualization...")
    print("=" * 60)
    
    plotter = Plotter(figsize=(12, 12))
    plotter.plot_grid(
        grid_world=grid,
        title="Maritime Navigation Scenario\nPort A → Port B",
        start=start_pos,
        goal=goal_pos,
        show=True
    )
    
    print("\n✓ Visualization displayed!")
    print("\nLegend:")
    print("  🟢 Green circle  = Start position (Port A)")
    print("  🔴 Red star      = Goal position (Port B)")
    print("  🔵 Blue areas    = Obstacles (islands, shallow water, restricted zones)")
    print("  ⬜ White areas   = Navigable water")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("  - Implement A* pathfinding algorithm")
    print("  - Find optimal path avoiding all obstacles")
    print("  - Add vessel dynamics and movement")
    print("=" * 60)


if __name__ == "__main__":
    main()