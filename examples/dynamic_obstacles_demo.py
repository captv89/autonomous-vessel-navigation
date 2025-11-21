"""
Dynamic Obstacles Visualization Demo

Shows multiple vessels moving through the environment with different behaviors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.environment.grid_world import GridWorld
from src.environment.dynamic_obstacles import DynamicObstacleManager
from src.visualization.dynamic_animator import DynamicObstaclesAnimator


def create_scenario_1():
    """Scenario 1: Crossing paths."""
    print("\nScenario 1: Crossing Paths")
    print("-" * 70)
    
    grid = GridWorld(width=60, height=60, cell_size=10.0)
    
    # Add some static obstacles
    grid.add_circular_obstacle(30, 30, radius=5)
    grid.add_obstacle(10, 45, width=8, height=3)
    
    # Create dynamic obstacles
    manager = DynamicObstacleManager()
    
    # Vessel 1: Moving from left to right
    vessel1 = manager.add_obstacle(
        x=5, y=30, 
        heading=0,  # East
        speed=1.0,
        behavior='straight'
    )
    print(f"  Added {vessel1.obstacle}")
    
    # Vessel 2: Moving from bottom to top
    vessel2 = manager.add_obstacle(
        x=30, y=5,
        heading=np.pi/2,  # North
        speed=1.5,
        behavior='straight'
    )
    print(f"  Added {vessel2.obstacle}")
    
    # Vessel 3: Circular pattern
    vessel3 = manager.add_obstacle(
        x=45, y=45,
        heading=0,
        speed=1.0,
        behavior='circular'
    )
    vessel3.set_circular_path(center=(45, 45), radius=8.0)
    print(f"  Added {vessel3.obstacle} (circular)")
    
    return grid, manager


def create_scenario_2():
    """Scenario 2: Harbor traffic."""
    print("\nScenario 2: Harbor Traffic")
    print("-" * 70)
    
    grid = GridWorld(width=80, height=60, cell_size=10.0)
    
    # Harbor entrance (narrow channel)
    grid.add_obstacle(0, 0, width=80, height=10)  # South wall
    grid.add_obstacle(0, 50, width=80, height=10)  # North wall
    grid.add_obstacle(0, 10, width=10, height=40)  # West wall
    grid.add_circular_obstacle(15, 30, radius=4)  # Buoy
    
    # Create traffic
    manager = DynamicObstacleManager()
    
    # Vessel entering harbor
    vessel1 = manager.add_obstacle(
        x=5, y=30,
        heading=0,
        speed=1.5,
        behavior='waypoint'
    )
    vessel1.set_waypoints([(15, 30), (30, 35), (50, 35), (70, 30)])
    print(f"  Added {vessel1.obstacle} (entering)")
    
    # Vessel leaving harbor
    vessel2 = manager.add_obstacle(
        x=70, y=25,
        heading=np.pi,  # West
        speed=1.8,
        behavior='waypoint'
    )
    vessel2.set_waypoints([(50, 25), (30, 20), (15, 25), (5, 30)])
    print(f"  Added {vessel2.obstacle} (leaving)")
    
    # Patrol vessel
    vessel3 = manager.add_obstacle(
        x=40, y=30,
        heading=0,
        speed=1.2,
        behavior='circular'
    )
    vessel3.set_circular_path(center=(40, 30), radius=10.0)
    print(f"  Added {vessel3.obstacle} (patrol)")
    
    return grid, manager


def main():
    """Main demo function."""
    print("=" * 70)
    print("DYNAMIC OBSTACLES VISUALIZATION")
    print("=" * 70)
    
    # Choose scenario
    print("\nChoose scenario:")
    print("  1. Crossing paths (simple)")
    print("  2. Harbor traffic (complex)")
    print("=" * 70)
    
    choice = input("Enter 1 or 2 (default=1): ").strip()
    
    if choice == '2':
        grid, manager = create_scenario_2()
    else:
        grid, manager = create_scenario_1()
    
    # Create animator
    animator = DynamicObstaclesAnimator(grid, manager)
    
    # Simulate
    print("\n" + "=" * 70)
    print("SIMULATING...")
    print("=" * 70)
    animator.simulate(duration=30.0, dt=0.1)
    
    # Animate
    print("\n" + "=" * 70)
    print("ANIMATION STARTING...")
    print("=" * 70)
    print("\nAnimation legend:")
    print("  ▬ Colored vessels    = Moving vessels (Rectangle + Triangle bow)")
    print("  ━━ Colored trails    = Past trajectories")
    print("  ⭕ Dashed circles    = Predicted position (5s ahead)")
    print("\nClose window when done.")
    print("=" * 70)
    
    animator.animate(
        interval=20,
        show_trails=True,
        show_predictions=True
    )


if __name__ == "__main__":
    main()