"""
Path Following Controller Demo

Demonstrates and compares different path following strategies:
1. Naive (just aim at next waypoint)
2. Pure Pursuit
3. Line-of-Sight (LOS)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from src.environment.grid_world import GridWorld
from src.pathfinding.astar import AStar, simplify_path
from src.vessel.vessel_model import KinematicVessel, NomotoVessel
from src.vessel.path_follower import PurePursuitController, LOSController
from src.visualization.path_animator import PathAnimator


def main():
    """Main demo function."""
    print("=" * 70)
    print("PATH FOLLOWING CONTROLLERS COMPARISON")
    print("=" * 70)
    
    # Use shared test scenario for consistency
    from src.pathfinding.astar import create_test_scenario
    grid, start, goal = create_test_scenario()
    
    # Find path
    print("\nFinding path with A*...")
    
    astar = AStar(grid, allow_diagonal=True)
    waypoints = astar.find_path(start, goal, verbose=True)
    
    if not waypoints:
        print("❌ No path found!")
        return
    
    # Simplify path for path following
    print(f"\nSimplifying path from {len(waypoints)} to fewer waypoints...")
    simplified_waypoints = simplify_path(waypoints, min_distance=8.0)
    print(f"Simplified path has {len(simplified_waypoints)} waypoints")
    print(f"Sample waypoints: {simplified_waypoints[:3]} ... {simplified_waypoints[-2:]}")
    
    # Debug: print all simplified waypoints
    print("\nAll simplified waypoints:")
    for i, wp in enumerate(simplified_waypoints):
        print(f"  {i}: {wp}")
    
    # Choose comparison mode
    print("\n" + "=" * 70)
    print("Choose demo mode:")
    print("  1. Compare controllers (Pure Pursuit vs LOS)")
    print("  2. Pure Pursuit only")
    print("  3. LOS only")
    print("=" * 70)
    
    choice = input("Enter 1, 2, or 3 (default=1): ").strip()
    
    # Create animator
    animator = PathAnimator(grid)
    trajectories = {}
    
    if choice in ['1', '']:
        # Compare both controllers
        print("\n" + "=" * 70)
        print("SIMULATING BOTH CONTROLLERS")
        print("=" * 70)
        
        # Pure Pursuit
        vessel_pp = NomotoVessel(x=float(start[0]), y=float(start[1]),
                                 heading=0.0, speed=0.5, K=0.5, T=3.0)
        controller_pp = PurePursuitController(lookahead_distance=10.0)
        trajectories["Pure Pursuit"] = animator.simulate_controller(
            vessel_pp, controller_pp, simplified_waypoints, "Pure Pursuit", dt=0.1, verbose=True)
        
        # LOS
        vessel_los = NomotoVessel(x=float(start[0]), y=float(start[1]),
                                  heading=0.0, speed=0.5, K=0.5, T=3.0)
        controller_los = LOSController(lookahead_distance=10.0, path_tolerance=2.0)
        trajectories["LOS"] = animator.simulate_controller(
            vessel_los, controller_los, simplified_waypoints, "LOS", dt=0.1, verbose=True)
        
    elif choice == '2':
        # Pure Pursuit only
        vessel = NomotoVessel(x=float(start[0]), y=float(start[1]),
                             heading=0.0, speed=0.5, K=0.5, T=3.0)
        controller = PurePursuitController(lookahead_distance=10.0)
        trajectories["Pure Pursuit"] = animator.simulate_controller(
            vessel, controller, simplified_waypoints, "Pure Pursuit", dt=0.1, verbose=True)
        
    elif choice == '3':
        # LOS only
        vessel = NomotoVessel(x=float(start[0]), y=float(start[1]),
                             heading=0.0, speed=0.5, K=0.5, T=3.0)
        controller = LOSController(lookahead_distance=10.0, path_tolerance=2.0)
        trajectories["LOS"] = animator.simulate_controller(
            vessel, controller, simplified_waypoints, "LOS", dt=0.1, verbose=True)
    
    # Animate
    print("\n" + "=" * 70)
    print("CREATING ANIMATION")
    print("=" * 70)
    print("\nAnimation legend:")
    print("  🟢 Green dashed  = A* planned waypoints")
    print("  🔴 Red line      = Actual vessel trajectory")
    print("  🔴 Red triangle  = Vessel")
    print("  🟡 Yellow circle = Target point (where controller is aiming)")
    print("  🟡 Yellow dashed = Line to target")
    print("Watch how each controller follows the path!")
    print("=" * 70)
    
    # Display animation
    animator.animate_comparison(waypoints, trajectories, interval=10)
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print("\nKey observations:")
    print("  - Pure Pursuit: Smoother, may cut corners slightly")
    print("  - LOS: Follows path more precisely, better cross-track control")
    print("  - Both handle the same path but with different strategies")
    print("=" * 70)


if __name__ == "__main__":
    main()