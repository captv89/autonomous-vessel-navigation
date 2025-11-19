"""
A* Pathfinding Demo - Visualize the A* algorithm finding paths.

This demonstrates:
- A* pathfinding on the maritime scenario
- Visual representation of the found path
- Comparison with straight-line distance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.grid_world import GridWorld
from src.pathfinding.astar import AStar
from src.visualization.plotter import Plotter


def create_simple_scenario():
    """Create a simple scenario for clear visualization."""
    print("Creating simple navigation scenario...")
    
    grid = GridWorld(width=50, height=50, cell_size=10.0)
    
    # Add obstacles
    grid.add_obstacle(15, 10, width=5, height=20)
    grid.add_circular_obstacle(35, 35, radius=5)
    grid.add_obstacle(25, 25, width=8, height=3)
    
    return grid


def create_complex_scenario():
    """Create the full maritime scenario."""
    print("Creating complex maritime scenario...")
    
    grid = GridWorld(width=100, height=100, cell_size=10.0)
    
    # Large island on the left
    grid.add_circular_obstacle(center_x=20, center_y=50, radius=12)
    
    # Small islands
    grid.add_circular_obstacle(center_x=45, center_y=25, radius=5)
    grid.add_circular_obstacle(center_x=60, center_y=70, radius=6)
    grid.add_circular_obstacle(center_x=75, center_y=40, radius=4)
    
    # Shallow water zones
    grid.add_obstacle(x=35, y=55, width=15, height=8)
    grid.add_obstacle(x=50, y=10, width=10, height=12)
    
    # Restricted zone
    grid.add_obstacle(x=10, y=10, width=12, height=12)
    
    # Reef area
    grid.add_circular_obstacle(center_x=85, center_y=20, radius=7)
    
    return grid


def demo_pathfinding(scenario_type='simple'):
    """
    Demonstrate A* pathfinding with visualization.
    
    Args:
        scenario_type: 'simple' or 'complex'
    """
    print("\n" + "=" * 70)
    print("A* PATHFINDING DEMONSTRATION")
    print("=" * 70)
    
    # Create scenario
    if scenario_type == 'simple':
        grid = create_simple_scenario()
        start = (5, 5)
        goal = (45, 45)
    else:
        grid = create_complex_scenario()
        start = (10, 90)
        goal = (90, 10)
    
    print(f"\n{grid}")
    
    # Calculate straight-line distance
    straight_distance = ((goal[0] - start[0])**2 + (goal[1] - start[1])**2)**0.5
    
    print("\nNavigation Task:")
    print("-" * 70)
    print(f"  Start Position:  {start}")
    print(f"  Goal Position:   {goal}")
    print(f"  Straight-line:   {straight_distance:.2f} cells ({straight_distance * grid.cell_size / 1000:.2f} km)")
    
    # Create A* pathfinder
    print("\nInitializing A* pathfinder...")
    astar = AStar(grid, allow_diagonal=True)
    
    # Find path
    print("\nSearching for optimal path...")
    print("-" * 70)
    path = astar.find_path(start, goal, verbose=True)
    
    if path:
        path_length = len(path)
        
        # Calculate actual path distance
        path_distance = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_distance += (dx**2 + dy**2)**0.5
        
        print("-" * 70)
        print("Path Statistics:")
        print(f"  Waypoints:       {path_length}")
        print(f"  Path distance:   {path_distance:.2f} cells ({path_distance * grid.cell_size / 1000:.2f} km)")
        print(f"  Extra distance:  {((path_distance - straight_distance) / straight_distance * 100):.1f}%")
        print(f"  Efficiency:      {(straight_distance / path_distance * 100):.1f}%")
        
        # Visualize
        print("\n" + "=" * 70)
        print("Generating visualization...")
        print("=" * 70)
        
        plotter = Plotter(figsize=(14, 12))
        
        title = f"A* Pathfinding Result\n"
        title += f"Path: {path_length} waypoints, {path_distance:.1f} cells"
        
        plotter.plot_grid(
            grid_world=grid,
            title=title,
            start=start,
            goal=goal,
            path=path,
            show=True
        )
        
        print("\n✓ Path found and visualized!")
        print("\nVisualization Legend:")
        print("  🟢 Green circle  = Start position")
        print("  🔴 Red star      = Goal position")
        print("  🟡 Yellow line   = A* optimal path")
        print("  🔵 Blue areas    = Obstacles")
        print("  ⬜ White areas   = Navigable water")
        
    else:
        print("\n❌ No path could be found!")
        print("   The goal may be unreachable or surrounded by obstacles.")
        
        # Still show the grid
        plotter = Plotter(figsize=(14, 12))
        plotter.plot_grid(
            grid_world=grid,
            title="No Path Found",
            start=start,
            goal=goal,
            show=True
        )


def main():
    """Main function - run both demos."""
    
    # # Simple scenario first
    # print("\n" + "=" * 70)
    # print("DEMO 1: SIMPLE SCENARIO")
    # print("=" * 70)
    # demo_pathfinding('simple')
    
    # input("\nPress Enter to continue to complex scenario...")
    
    # Complex scenario
    print("\n" + "=" * 70)
    print("DEMO 2: COMPLEX MARITIME SCENARIO")
    print("=" * 70)
    demo_pathfinding('complex')
    
    print("\n" + "=" * 70)
    print("DEMOS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()