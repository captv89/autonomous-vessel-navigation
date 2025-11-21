"""
Integrated Navigation Demo

Shows our vessel following A* path while dynamic obstacles move around.
Visualizes the need for collision avoidance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.environment.grid_world import GridWorld
from src.pathfinding.astar import AStar, create_test_scenario
from src.vessel.vessel_model import NomotoVessel
from src.vessel.path_follower import PurePursuitController, LOSController
from src.environment.dynamic_obstacles import DynamicObstacleManager
from src.visualization.integrated_animator import IntegratedNavigationAnimator


def main():
    """Main demo function."""
    print("=" * 70)
    print("INTEGRATED NAVIGATION WITH DYNAMIC OBSTACLES")
    print("=" * 70)
    
    # Create scenario
    grid, start, goal = create_test_scenario()
    
    # Find path
    print("\nFinding path with A*...")
    astar = AStar(grid, allow_diagonal=True)
    waypoints = astar.find_path(start, goal, verbose=True)
    
    if not waypoints:
        print("❌ No path found!")
        return
    
    # Create dynamic obstacles
    print("\n" + "=" * 70)
    print("CREATING DYNAMIC OBSTACLES")
    print("=" * 70)
    
    manager = DynamicObstacleManager()
    
    # Obstacle 1: Crossing from west to east
    vessel1 = manager.add_obstacle(
        x=5, y=50,
        heading=0,  # East
        speed=1.5,
        behavior='straight'
    )
    print(f"  Added {vessel1.obstacle}")
    
    # Obstacle 2: Moving north through our path
    vessel2 = manager.add_obstacle(
        x=40, y=10,
        heading=np.pi/2,  # North
        speed=1.2,
        behavior='straight'
    )
    print(f"  Added {vessel2.obstacle}")
    
    # Obstacle 3: Circular patrol in middle area
    vessel3 = manager.add_obstacle(
        x=50, y=50,
        heading=0,
        speed=1.0,
        behavior='circular'
    )
    vessel3.set_circular_path(center=(50, 50), radius=10.0)
    print(f"  Added {vessel3.obstacle} (circular)")
    
    # Create our vessel
    print("\n" + "=" * 70)
    print("CREATING OUR VESSEL")
    print("=" * 70)
    
    our_vessel = NomotoVessel(
        x=float(start[0]),
        y=float(start[1]),
        heading=0.0,
        speed=0.5,
        K=0.5,
        T=3.0
    )
    print(f"  Our vessel at start: ({start[0]}, {start[1]})")
    
    # Choose controller
    print("\nChoose path following controller:")
    print("  1. Pure Pursuit")
    print("  2. LOS (Line-of-Sight)")
    
    choice = input("Enter 1 or 2 (default=2): ").strip()
    
    if choice == '1':
        controller = PurePursuitController(lookahead_distance=8.0)
        controller_name = "Pure Pursuit"
    else:
        controller = LOSController(lookahead_distance=8.0, path_tolerance=3.0)
        controller_name = "LOS"
    
    print(f"  Using {controller_name} controller")
    
    # Create simulator
    print("\n" + "=" * 70)
    print("SIMULATING...")
    print("=" * 70)
    
    simulator = IntegratedNavigationAnimator(
        grid, our_vessel, controller, manager, controller_name
    )
    
    # Simulate
    simulator.simulate(waypoints, duration=60.0, dt=0.1)
    
    # Animate
    print("\n" + "=" * 70)
    print("ANIMATION STARTING...")
    print("=" * 70)
    print("\nAnimation legend:")
    print("  🟢 Green vessel      = Our vessel (Rectangle + Triangle bow)")
    print("  🟢 Green line        = Our actual path")
    print("  🟢 Green dashed      = Planned A* path")
    print("  🔴 Colored vessels   = Dynamic obstacles (smaller)")
    print("  ⭕ Dashed circles    = Predicted positions (5s)")
    print("  🟡 Yellow circle     = Our target point")
    print("  ⚠️  Red box          = Collision warning")
    print("\nWatch for close encounters and potential collisions!")
    print("=" * 70)
    
    simulator.animate(waypoints, interval=5, show_predictions=True)
    
    print("\n" + "=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)
    print("\nWhat you should notice:")
    print("  1. Our vessel blindly follows the planned path")
    print("  2. Dynamic obstacles move independently")
    print("  3. Close encounters occur (red warning box)")
    print("  4. NO collision avoidance yet - vessels can collide!")
    print("\nNext step: Add collision detection and avoidance!")
    print("=" * 70)


if __name__ == "__main__":
    main()