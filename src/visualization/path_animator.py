"""
Path Following Animation

Animates vessel following waypoints with path following controllers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from typing import List, Tuple, Dict


class PathAnimator:
    """
    Animates vessel path following with multiple controllers for comparison.
    """
    
    def __init__(self, grid_world):
        """
        Initialize path animator.
        
        Args:
            grid_world: GridWorld object containing the environment
        """
        self.grid_world = grid_world
    
    def simulate_controller(self, vessel, controller, waypoints: List[Tuple[float, float]],
                           controller_name: str, dt: float = 0.1, 
                           max_steps: int = 5000, verbose: bool = True) -> List[Dict]:
        """
        Simulate vessel following waypoints with a controller.
        
        Args:
            vessel: Vessel model instance
            controller: Path following controller instance
            waypoints: Path waypoints to follow
            controller_name: Name for display
            dt: Time step
            max_steps: Maximum simulation steps
            verbose: Print detailed logging
            
        Returns:
            List of trajectory states
        """
        print(f"\nSimulating {controller_name}...")
        if verbose:
            print(f"  Start: {waypoints[0]}, Goal: {waypoints[-1]}")
            print(f"  Total waypoints: {len(waypoints)}")
        
        trajectory = []
        controller.reset()
        steps = 0
        
        goal = waypoints[-1]
        stuck_counter = 0
        last_position = (float('inf'), float('inf'))
        last_waypoint_idx = 0
        obstacle_warnings = 0
        vessel_type = type(vessel).__name__
        
        while steps < max_steps:
            steps += 1
            
            # Get current state
            x, y = vessel.get_position()
            heading = vessel.get_heading()
            speed = vessel.get_speed()
            
            # Check for obstacles (grid value > 0.5 means obstacle)
            grid_x, grid_y = int(round(x)), int(round(y))
            if (0 <= grid_x < self.grid_world.width and 
                0 <= grid_y < self.grid_world.height):
                if self.grid_world.grid[grid_y, grid_x] > 0.5:
                    obstacle_warnings += 1
                    if verbose and obstacle_warnings <= 3:
                        print(f"  ⚠ Step {steps}: Vessel in obstacle at ({x:.1f}, {y:.1f})!")
            
            # Check if reached goal
            dist_to_goal = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
            if dist_to_goal < 3.0:
                print(f"  ✓ Reached goal in {steps} steps ({steps * dt:.1f}s)")
                if obstacle_warnings > 0:
                    print(f"  ⚠ Hit obstacles {obstacle_warnings} times during navigation!")
                break
            
            # Check if stuck (not making progress)
            dist_moved = np.sqrt((x - last_position[0])**2 + (y - last_position[1])**2)
            if dist_moved < 0.01:
                stuck_counter += 1
                if stuck_counter > 100:
                    print(f"  ⚠ Vessel stuck at ({x:.1f}, {y:.1f}), stopping simulation")
                    print(f"     Current waypoint index: {getattr(controller, 'current_waypoint_idx', getattr(controller, 'current_segment', 'N/A'))}")
                    break
            else:
                stuck_counter = 0
            last_position = (x, y)
            
            # Get desired heading from controller
            desired_heading = controller.compute_desired_heading((x, y), waypoints)
            
            if desired_heading is None:
                print(f"  ✓ Path complete")
                break
            
            # Log waypoint advancement
            current_wp_idx = getattr(controller, 'current_waypoint_idx', getattr(controller, 'current_segment', -1))
            if verbose and current_wp_idx != last_waypoint_idx and steps % 10 == 0:
                print(f"  Step {steps}: Pos=({x:.1f}, {y:.1f}), Waypoint={current_wp_idx}/{len(waypoints)-1}, Dist to goal={dist_to_goal:.1f}")
                last_waypoint_idx = current_wp_idx
            
            # Get target point for visualization
            target = controller.get_target_point((x, y), waypoints)
            
            # Calculate cross-track error (distance to nearest path segment)
            cross_track_error = 0.0
            if len(waypoints) >= 2:
                min_dist = float('inf')
                for i in range(len(waypoints) - 1):
                    p1 = np.array(waypoints[i])
                    p2 = np.array(waypoints[i + 1])
                    vessel_pos = np.array([x, y])
                    
                    # Project vessel position onto line segment
                    line_vec = p2 - p1
                    line_len_sq = np.dot(line_vec, line_vec)
                    if line_len_sq > 0:
                        t = max(0, min(1, np.dot(vessel_pos - p1, line_vec) / line_len_sq))
                        projection = p1 + t * line_vec
                        dist = np.linalg.norm(vessel_pos - projection)
                        min_dist = min(min_dist, dist)
                cross_track_error = min_dist
            
            # Update vessel based on type
            rudder_angle = 0.0
            rate_of_turn = 0.0
            
            if vessel_type == 'NomotoVessel':
                # PD controller for rudder
                heading_error = desired_heading - heading
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                
                yaw_rate = vessel.r if hasattr(vessel, 'r') else vessel.get_turn_rate()
                
                # Tuning gains for T=3.0, K=0.5
                Kp = 4.0  # Proportional gain
                Kd = 2.5  # Damping term
                
                rudder_command = (Kp * heading_error) - (Kd * yaw_rate)
                vessel.update(dt, rudder_command=rudder_command)
                
                # Get rudder angle and rate of turn from vessel
                rudder_angle = vessel.rudder_angle if hasattr(vessel, 'rudder_angle') else rudder_command
                rate_of_turn = yaw_rate
            elif vessel_type == 'KinematicVessel':
                vessel.update(dt, desired_heading=desired_heading)
                rate_of_turn = vessel.state.turn_rate if hasattr(vessel.state, 'turn_rate') else 0.0
            
            # Record state for animation
            trajectory.append({
                'x': x,
                'y': y,
                'heading': heading,
                'speed': speed,
                'time': steps * dt,
                'target': target,
                'desired_heading': desired_heading,
                'rudder_angle': rudder_angle,
                'rate_of_turn': rate_of_turn,
                'cross_track_error': cross_track_error
            })
        
        # Calculate statistics
        if len(trajectory) > 0:
            # Path length
            path_length = 0
            for i in range(len(trajectory) - 1):
                dx = trajectory[i+1]['x'] - trajectory[i]['x']
                dy = trajectory[i+1]['y'] - trajectory[i]['y']
                path_length += np.sqrt(dx**2 + dy**2)
            
            # Average cross-track error (distance from waypoints)
            cross_track_errors = []
            for state in trajectory[::10]:  # Sample every 10 steps
                min_dist = float('inf')
                for wp in waypoints:
                    dist = np.sqrt((state['x'] - wp[0])**2 + (state['y'] - wp[1])**2)
                    min_dist = min(min_dist, dist)
                cross_track_errors.append(min_dist)
            
            avg_cross_track = np.mean(cross_track_errors) if cross_track_errors else 0
            
            print(f"  Path length: {path_length:.1f} units")
            print(f"  Avg cross-track error: {avg_cross_track:.2f} units")
        
        return trajectory
    
    def animate_comparison(self, waypoints: List[Tuple[float, float]], 
                          trajectories: Dict[str, List[Dict]],
                          interval: int = 10):
        """
        Animate multiple trajectories for comparison.
        
        Args:
            waypoints: A* waypoints
            trajectories: Dict of {controller_name: trajectory}
            interval: Animation interval in milliseconds (default=10)
        """
        print("\nCreating comparison animation...")
        
        # Create figure with subplots
        n_controllers = len(trajectories)
        fig, axes = plt.subplots(1, n_controllers, figsize=(7 * n_controllers, 7))
        
        if n_controllers == 1:
            axes = [axes]
        
        # Setup for each controller
        animations = []
        
        for idx, (name, trajectory) in enumerate(trajectories.items()):
            ax = axes[idx]
            
            # Plot grid
            ax.imshow(self.grid_world.grid, cmap='Blues', origin='lower',
                     vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
            
            # Plot A* waypoints
            wp_x = [w[0] for w in waypoints]
            wp_y = [w[1] for w in waypoints]
            ax.plot(wp_x, wp_y, 'g--', linewidth=2, alpha=0.4, label='A* Path')
            
            # Plot start and goal
            ax.plot(waypoints[0][0], waypoints[0][1], 'go', markersize=12,
                   markeredgecolor='darkgreen', markeredgewidth=2)
            ax.plot(waypoints[-1][0], waypoints[-1][1], 'r*', markersize=18,
                   markeredgecolor='darkred', markeredgewidth=2)
            
            # Vessel shape: Rectangle body + Triangle bow (10% smaller)
            # Body: centered at origin, 5.4x2.25 (90% of 6x2.5)
            vessel_body = Rectangle((-2.7, -1.125), 5.4, 2.25, 
                                   facecolor='black', edgecolor='darkgray',
                                   linewidth=1.5, zorder=10)
            ax.add_patch(vessel_body)
            
            # Bow: triangle at front (right side), 1.8x2.25 (90% of 2x2.5)
            vessel_bow = Polygon([(2.7, -1.125), (2.7, 1.125), (4.5, 0)],
                                facecolor='black', edgecolor='darkgray',
                                linewidth=1.5, zorder=10)
            ax.add_patch(vessel_bow)
            
            # Target point
            target_circle = Circle((0, 0), 0.8, facecolor='yellow',
                                  edgecolor='orange', linewidth=2, zorder=8)
            ax.add_patch(target_circle)
            
            # Target line
            target_line, = ax.plot([], [], 'y--', linewidth=1.5, alpha=0.6, zorder=7)
            
            # Trajectory trail
            trail_line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7,
                                 label='Actual Path', zorder=3)
            
            # Setup
            ax.set_xlabel('X (cells)', fontsize=11)
            ax.set_ylabel('Y (cells)', fontsize=11)
            ax.set_title(f'{name} Controller', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_aspect('equal')
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.2)
            
            # Info text (vessel state)
            info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                              fontsize=9, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                              zorder=12)
            
            # Dynamics text (bottom left - rudder, ROT, CTE)
            dynamics_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                                  fontsize=9, verticalalignment='bottom',
                                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),
                                  zorder=12)
            
            animations.append({
                'ax': ax,
                'trajectory': trajectory,
                'vessel_body': vessel_body,
                'vessel_bow': vessel_bow,
                'target_circle': target_circle,
                'target_line': target_line,
                'trail_line': trail_line,
                'info_text': info_text,
                'dynamics_text': dynamics_text,
                'trail_x': [],
                'trail_y': [],
                'distance_traveled': 0.0
            })
        
        plt.tight_layout()
        
        # Find longest trajectory for animation length
        max_frames = max(len(anim['trajectory']) for anim in animations)
        
        def init():
            """Initialize animation."""
            return []
        
        def update(frame):
            """Update all subplots."""
            artists = []
            
            for anim in animations:
                trajectory = anim['trajectory']
                
                if frame >= len(trajectory):
                    continue
                
                state = trajectory[frame]
                x, y = state['x'], state['y']
                heading = state['heading']
                rudder_angle = state.get('rudder_angle', 0.0)
                rate_of_turn = state.get('rate_of_turn', 0.0)
                cross_track_error = state.get('cross_track_error', 0.0)
                
                # Create rotation transform
                transform = (Affine2D().rotate(heading).translate(x, y) + 
                           anim['ax'].transData)
                
                # Update vessel body and bow with rotation
                anim['vessel_body'].set_transform(transform)
                anim['vessel_bow'].set_transform(transform)
                
                # Update target point
                if state['target']:
                    tx, ty = state['target']
                    anim['target_circle'].set_center((tx, ty))
                    anim['target_line'].set_data([x, tx], [y, ty])
                
                # Update trail and calculate distance traveled
                anim['trail_x'].append(x)
                anim['trail_y'].append(y)
                anim['trail_line'].set_data(anim['trail_x'], anim['trail_y'])
                
                # Calculate distance traveled
                if len(anim['trail_x']) > 1:
                    dx = anim['trail_x'][-1] - anim['trail_x'][-2]
                    dy = anim['trail_y'][-1] - anim['trail_y'][-2]
                    anim['distance_traveled'] += np.sqrt(dx**2 + dy**2)
                
                # Convert heading to standard navigation notation (0° = North, 90° = East, clockwise)
                # Matplotlib uses (0° = East, counterclockwise), so convert: Nav = 90° - Math heading
                heading_360 = (90 - np.degrees(heading)) % 360
                
                # Update info text (top left - position and speed)
                anim['info_text'].set_text(
                    f'Time: {state["time"]:.1f}s\n'
                    f'Pos: ({x:.1f}, {y:.1f})\n'
                    f'Heading: {heading_360:.0f}°\n'
                    f'Speed: {state["speed"]:.2f} units/s\n'
                    f'Distance: {anim["distance_traveled"]:.1f} units'
                )
                
                # Update dynamics text (bottom left - rudder, ROT, CTE)
                anim['dynamics_text'].set_text(
                    f'Rudder: {np.degrees(rudder_angle):+.1f}°\n'
                    f'ROT: {np.degrees(rate_of_turn):+.2f}°/s\n'
                    f'CTE: {cross_track_error:.2f} units'
                )
                
                artists.extend([anim['vessel_body'], anim['vessel_bow'], 
                              anim['target_circle'], anim['target_line'], 
                              anim['trail_line'], anim['info_text'],
                              anim['dynamics_text']])
            
            return artists
        
        # Sample frames for performance
        frame_skip = max(1, max_frames // 500)
        frames = range(0, max_frames, frame_skip)
        
        anim = FuncAnimation(fig, update, init_func=init,
                            frames=frames, interval=interval,
                            blit=False, repeat=False)
        
        plt.show()
        print("\n✓ Animation complete!")
