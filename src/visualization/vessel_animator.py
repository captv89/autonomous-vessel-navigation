"""
Vessel Animation - Visualize vessel following waypoints with physics.

Shows:
- Vessel moving through the environment
- Path being followed
- Vessel orientation (heading)
- Real-time position updates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow, Circle, Wedge
from typing import List, Tuple, Optional
import sys
import os


class VesselAnimator:
    """
    Animates a vessel following a path through the grid world.
    """
    
    def __init__(self, grid_world, vessel, figsize: Tuple[int, int] = (14, 12)):
        """
        Initialize vessel animator.
        
        Args:
            grid_world: GridWorld object
            vessel: Vessel object (KinematicVessel or NomotoVessel)
            figsize: Figure size
        """
        self.grid_world = grid_world
        self.vessel = vessel
        self.figsize = figsize
        
        # Animation data
        self.trajectory = []  # Store vessel positions
        self.time = 0.0
        
    def simulate_path_following(self, waypoints: List[Tuple[float, float]], 
                               controller,
                               dt: float = 0.1) -> List[dict]:
        """
        Simulate vessel following waypoints using a path following controller.
        
        Args:
            waypoints: List of (x, y) waypoints to follow
            controller: Path following controller (PurePursuitController or LOSController)
            dt: Simulation timestep
            
        Returns:
            List of simulation states for animation
        """
        print(f"Simulating vessel following {len(waypoints)} waypoints...")
        
        trajectory = []
        max_steps = 10000  # Prevent infinite loops
        steps = 0
        
        # Reset controller
        controller.reset()
        
        while steps < max_steps:
            steps += 1
            
            # Get current position
            x, y = self.vessel.get_position()
            heading = self.vessel.get_heading()
            
            # Get desired heading from controller
            desired_heading = controller.compute_desired_heading((x, y), waypoints)
            
            # Check if we've completed the path
            if desired_heading is None:
                print(f"✓ Path following complete!")
                break
            
            # Record state
            trajectory.append({
                'x': x,
                'y': y,
                'heading': heading,
                'speed': self.vessel.get_speed(),
                'time': steps * dt,
                'target_waypoint': controller.current_waypoint_idx if hasattr(controller, 'current_waypoint_idx') else controller.current_segment
            })
            
            # Update vessel based on model type
            vessel_type = type(self.vessel).__name__
            
            if vessel_type == 'KinematicVessel':
                # Kinematic model - direct heading control
                self.vessel.update(dt, desired_heading=desired_heading)
            
            elif vessel_type == 'NomotoVessel':
                # Nomoto model - use rudder control
                # Simple proportional controller
                heading_error = desired_heading - heading
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                
                # Proportional gain for rudder
                Kp = 2.0
                rudder_command = Kp * heading_error
                
                self.vessel.update(dt, rudder_command=rudder_command)
            
        print(f"  Simulation complete: {steps} steps ({steps * dt:.1f}s)")
        print(f"  Final position: ({self.vessel.state.x:.2f}, {self.vessel.state.y:.2f})")
        
        return trajectory
    
    def animate_trajectory(self, waypoints: List[Tuple[float, float]],
                          trajectory: List[dict],
                          interval: int = 20,
                          show_trajectory: bool = True,
                          show_waypoints: bool = True):
        """
        Animate the vessel following the path.
        
        Args:
            waypoints: Original waypoints from A*
            trajectory: Simulated trajectory from simulate_path_following
            interval: Milliseconds between frames
            show_trajectory: Show past trajectory trail
            show_waypoints: Show A* waypoints
        """
        print(f"\nCreating animation with {len(trajectory)} frames...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot grid
        im = ax.imshow(self.grid_world.grid, cmap='Blues', origin='lower',
                      vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
        
        # Plot waypoints if requested
        if show_waypoints and waypoints:
            waypoint_x = [w[0] for w in waypoints]
            waypoint_y = [w[1] for w in waypoints]
            ax.plot(waypoint_x, waypoint_y, 'g--', linewidth=2, alpha=0.5, 
                   label='A* Waypoints')
            ax.plot(waypoint_x, waypoint_y, 'go', markersize=4, alpha=0.5)
        
        # Plot start and goal
        if waypoints:
            start = waypoints[0]
            goal = waypoints[-1]
            ax.plot(start[0], start[1], 'go', markersize=15, label='Start',
                   markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)
            ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal',
                   markeredgecolor='darkred', markeredgewidth=2, zorder=5)
        
        # Initialize vessel representation
        vessel_size = 2.0
        
        # Vessel body (triangle pointing in heading direction)
        vessel_patch = Wedge((0, 0), vessel_size, 0, 360, 
                            facecolor='red', edgecolor='darkred', 
                            linewidth=2, zorder=10)
        ax.add_patch(vessel_patch)
        
        # Trajectory trail
        trajectory_line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, 
                                  label='Vessel Path', zorder=3)
        
        # Heading indicator (arrow) - will be created in update function
        heading_arrow = None
        
        # Setup plot
        ax.set_xlabel('X (cells)', fontsize=12)
        ax.set_ylabel('Y (cells)', fontsize=12)
        ax.set_title('Vessel Navigation Simulation', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal')
        
        # Info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                          fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                          zorder=12)
        
        # Grid lines
        ax.set_xticks(np.arange(-0.5, self.grid_world.width, 5), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_world.height, 5), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        
        plt.tight_layout()
        
        # Animation data
        trail_x = []
        trail_y = []
        heading_arrow_ref = [None]  # Use list to allow modification in nested function
        
        def init():
            """Initialize animation."""
            trajectory_line.set_data([], [])
            return vessel_patch, trajectory_line, info_text
        
        def update(frame):
            """Update animation frame."""
            if frame >= len(trajectory):
                return vessel_patch, trajectory_line, info_text
            
            state = trajectory[frame]
            x, y = state['x'], state['y']
            heading = state['heading']
            speed = state['speed']
            
            # Update vessel position (as a wedge/triangle pointing forward)
            vessel_patch.set_center((x, y))
            # Wedge angles in degrees, 0° is East
            heading_deg = np.degrees(heading)
            vessel_patch.set_theta1(heading_deg - 150)
            vessel_patch.set_theta2(heading_deg + 150)
            
            # Update heading arrow - remove old one if exists
            if heading_arrow_ref[0] is not None:
                try:
                    heading_arrow_ref[0].remove()
                except:
                    pass
            
            # Create new arrow
            arrow_length = 3.0
            dx = arrow_length * np.cos(heading)
            dy = arrow_length * np.sin(heading)
            
            heading_arrow_ref[0] = FancyArrow(x, y, dx, dy, width=0.5,
                                          head_width=1.5, head_length=1.0,
                                          facecolor='yellow', edgecolor='orange',
                                          linewidth=1, zorder=11)
            ax.add_patch(heading_arrow_ref[0])
            
            # Update trajectory trail
            if show_trajectory:
                trail_x.append(x)
                trail_y.append(y)
                trajectory_line.set_data(trail_x, trail_y)
            
            # Update info text
            info_text.set_text(
                f'Time: {state["time"]:.1f}s\n'
                f'Position: ({x:.1f}, {y:.1f})\n'
                f'Heading: {np.degrees(heading):.1f}°\n'
                f'Speed: {speed:.1f} units/s\n'
                f'Waypoint: {state["target_waypoint"]}/{len(waypoints)-1}'
            )
            
            return vessel_patch, trajectory_line, info_text
        
        # Create animation
        # Sample frames to speed up animation (every nth frame)
        frame_skip = max(1, len(trajectory) // 500)  # Max 500 frames
        frames = range(0, len(trajectory), frame_skip)
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=frames, interval=interval,
                           blit=False, repeat=False)
        
        plt.show()
        
        print("\n✓ Animation complete!")


def demo_vessel_animation():
    """Demonstrate vessel animation."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.environment.grid_world import GridWorld
    from src.pathfinding.astar import AStar, create_test_scenario
    from src.vessel.vessel_model import KinematicVessel, NomotoVessel
    
    print("=" * 70)
    print("VESSEL NAVIGATION ANIMATION")
    print("=" * 70)
    
    # Use the shared test scenario
    grid, start, goal = create_test_scenario()
    
    # Find path with A*
    print("\nFinding path with A*...")
    
    astar = AStar(grid, allow_diagonal=True)
    waypoints = astar.find_path(start, goal, verbose=True)
    
    if not waypoints:
        print("❌ No path found!")
        return
    
    # Ask user which model to use
    print("\n" + "=" * 70)
    print("Choose vessel model:")
    print("  1. Kinematic (instant response, simpler)")
    print("  2. Nomoto (realistic turning dynamics)")
    print("=" * 70)
    
    choice = input("Enter 1 or 2 (default=1): ").strip()
    
    if choice == '2':
        print("\nUsing Nomoto vessel model...")
        vessel = NomotoVessel(
            x=float(start[0]), 
            y=float(start[1]), 
            heading=0.0,
            speed=2.0,
            K=0.5,
            T=3.0
        )
    else:
        print("\nUsing Kinematic vessel model...")
        vessel = KinematicVessel(
            x=float(start[0]),
            y=float(start[1]),
            heading=0.0,
            speed=2.0,
            max_turn_rate=0.15
        )
    
    # Create animator
    animator = VesselAnimator(grid, vessel)
    
    # Simulate path following
    print("\n" + "=" * 70)
    print("Simulating vessel navigation...")
    print("=" * 70)
    
    # Use Pure Pursuit controller
    from src.vessel.path_follower import PurePursuitController
    controller = PurePursuitController(lookahead_distance=3.0)
    
    trajectory = animator.simulate_path_following(
        waypoints, 
        controller=controller,
        dt=0.1
    )
    
    # Animate
    print("\n" + "=" * 70)
    print("Starting animation...")
    print("=" * 70)
    print("\nAnimation legend:")
    print("  🔴 Red triangle  = Vessel")
    print("  🟡 Yellow arrow  = Heading direction")
    print("  🔴 Red line      = Vessel trajectory (actual path)")
    print("  🟢 Green dashed  = A* waypoints (planned path)")
    print("  🟢 Green circle  = Start")
    print("  🔴 Red star      = Goal")
    print("\nClose window when done.")
    print("=" * 70)
    
    animator.animate_trajectory(
        waypoints=waypoints,
        trajectory=trajectory,
        interval=10,
        show_trajectory=True,
        show_waypoints=True
    )


if __name__ == "__main__":
    demo_vessel_animation()