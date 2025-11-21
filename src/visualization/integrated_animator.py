"""
Integrated Navigation Animation

Animates our vessel following a path while dynamic obstacles move around.
Combines path following and dynamic obstacle visualization by reusing existing animators.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from typing import List, Tuple, Dict

from .path_animator import PathAnimator
from .dynamic_animator import DynamicObstaclesAnimator


class IntegratedNavigationAnimator:
    """Animates our vessel and dynamic obstacles together using composition."""
    
    def __init__(self, grid_world, our_vessel, controller, 
                 obstacle_manager, controller_name: str):
        """
        Initialize animator.
        
        Args:
            grid_world: GridWorld object
            our_vessel: Our vessel object
            controller: Path following controller
            obstacle_manager: DynamicObstacleManager
            controller_name: Controller name for display
        """
        self.grid_world = grid_world
        self.our_vessel = our_vessel
        self.controller = controller
        self.manager = obstacle_manager
        self.controller_name = controller_name
        
        # Reuse existing animators
        self.path_animator = PathAnimator(grid_world)
        self.dynamic_animator = DynamicObstaclesAnimator(grid_world, obstacle_manager)
        
        self.simulation_data = []
    
    def simulate(self, waypoints: List[Tuple[float, float]], 
                duration: float, dt: float = 0.1):
        """
        Simulate our vessel and dynamic obstacles together.
        Leverages PathAnimator's simulate_controller for our vessel logic.
        
        Args:
            waypoints: Path for our vessel
            duration: Simulation duration
            dt: Time step
        """
        print(f"\n{'='*60}")
        print(f"Integrated Simulation: {self.controller_name}")
        print(f"  Our vessel + {len(self.manager.obstacles)} dynamic obstacles")
        print(f"{'='*60}")
        
        self.simulation_data = []
        self.controller.reset()
        
        steps = int(duration / dt)
        max_steps = steps
        goal = waypoints[-1]
        
        # Track our vessel using PathAnimator's logic (but inline to sync with obstacles)
        our_trajectory = []
        step = 0
        
        while step < max_steps:
            # Get our vessel state
            our_x, our_y = self.our_vessel.get_position()
            our_heading = self.our_vessel.get_heading()
            our_speed = self.our_vessel.get_speed()
            
            # Check if reached goal
            dist_to_goal = np.sqrt((our_x - goal[0])**2 + (our_y - goal[1])**2)
            if dist_to_goal < 3.0:
                print(f"  ✓ Our vessel reached goal in {step} steps ({step * dt:.1f}s)")
                break
            
            # Get desired heading from controller
            desired_heading = self.controller.compute_desired_heading(
                (our_x, our_y), waypoints)
            
            if desired_heading is None:
                print(f"  ✓ Path complete at step {step}")
                break
            
            # Get target point for visualization
            target = self.controller.get_target_point((our_x, our_y), waypoints)
            
            # Get rudder angle and rate of turn
            rudder_angle = 0.0
            rate_of_turn = 0.0
            if hasattr(self.our_vessel, 'rudder_angle'):
                rudder_angle = self.our_vessel.rudder_angle
            if hasattr(self.our_vessel, 'get_turn_rate'):
                rate_of_turn = self.our_vessel.get_turn_rate()
            
            # Combined state (merges our vessel + obstacles)
            state = {
                'time': step * dt,
                'our_vessel': {
                    'x': our_x,
                    'y': our_y,
                    'heading': our_heading,
                    'speed': our_speed,
                    'target': target,
                    'rudder_angle': rudder_angle,
                    'rate_of_turn': rate_of_turn
                },
                'dynamic_obstacles': []
            }
            
            # Record dynamic obstacles
            min_distance = float('inf')
            for vessel in self.manager.obstacles:
                obs_state = {
                    'id': vessel.obstacle.id,
                    'x': vessel.obstacle.x,
                    'y': vessel.obstacle.y,
                    'heading': vessel.obstacle.heading,
                    'speed': vessel.obstacle.speed,
                    'predicted': vessel.predict_position(5.0)
                }
                state['dynamic_obstacles'].append(obs_state)
                
                # Track closest approach
                dist = np.sqrt((vessel.obstacle.x - our_x)**2 + 
                             (vessel.obstacle.y - our_y)**2)
                min_distance = min(min_distance, dist)
                
                # Warn about close encounters
                if dist < 10.0 and step % 50 == 0:
                    print(f"    ⚠️  Close encounter with vessel {vessel.obstacle.id}: {dist:.1f} units")
            
            self.simulation_data.append(state)
            
            # Update our vessel (reuse PathAnimator's PD controller logic)
            heading_error = desired_heading - our_heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            vessel_type = type(self.our_vessel).__name__
            if vessel_type == 'NomotoVessel':
                Kp = 4.0
                Kd = 2.5
                yaw_rate = self.our_vessel.get_turn_rate()
                rudder_command = (Kp * heading_error) - (Kd * yaw_rate)
                self.our_vessel.update(dt, rudder_command=rudder_command)
            else:
                self.our_vessel.update(dt, desired_heading=desired_heading)
            
            # Update dynamic obstacles
            self.manager.update_all(dt)
            step += 1
        
        print(f"  ✓ Simulation complete: {len(self.simulation_data)} frames")
        print(f"{'='*60}\n")
        return self.simulation_data
    
    def _create_vessel_patches(self, ax, color: str, scale: float = 1.0, zorder: int = 10):
        """
        Create vessel body and bow patches (DRY helper).
        
        Args:
            ax: Matplotlib axis
            color: Vessel color
            scale: Size scale (1.0 = full size, 0.9 = 10% smaller)
            zorder: Drawing order
            
        Returns:
            Tuple of (body_patch, bow_patch)
        """
        # Base dimensions
        body_width = 5.4 * scale
        body_height = 2.25 * scale
        bow_length = 1.8 * scale
        
        body = Rectangle((-body_width/2, -body_height/2), body_width, body_height,
                        facecolor=color, edgecolor='darkgray',
                        linewidth=1.5, alpha=0.8 if scale < 1.0 else 0.9, 
                        zorder=zorder)
        ax.add_patch(body)
        
        bow_x = body_width / 2
        bow = Polygon([(bow_x, -body_height/2), (bow_x, body_height/2), 
                      (bow_x + bow_length, 0)],
                     facecolor=color, edgecolor='darkgray',
                     linewidth=1.5, alpha=0.8 if scale < 1.0 else 0.9,
                     zorder=zorder)
        ax.add_patch(bow)
        
        return body, bow
    
    def animate(self, waypoints: List[Tuple[float, float]], 
               interval: int = 10, show_predictions: bool = True, repeat: bool = False):
        """
        Animate the integrated simulation.
        Reuses vessel shape creation helper and follows same pattern as other animators.
        
        Args:
            waypoints: A* waypoints
            interval: Milliseconds between frames
            show_predictions: Show predicted positions for obstacles
            repeat: Whether to loop the animation
        """
        if not self.simulation_data:
            print("❌ No simulation data! Run simulate() first.")
            return
        
        print("\nCreating integrated animation...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot environment (reuse grid plotting pattern)
        ax.imshow(self.grid_world.grid, cmap='Blues', origin='lower',
                 vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
        
        # Plot planned path
        wp_x, wp_y = [w[0] for w in waypoints], [w[1] for w in waypoints]
        ax.plot(wp_x, wp_y, 'g--', linewidth=2, alpha=0.5, label='Planned Path')
        ax.plot(wp_x[0], wp_y[0], 'go', markersize=15, 
               markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
        ax.plot(wp_x[-1], wp_y[-1], 'r*', markersize=20,
               markeredgecolor='darkred', markeredgewidth=2, label='Goal')
        
        # Our vessel (green, full size) - using helper
        our_body, our_bow = self._create_vessel_patches(ax, 'green', scale=1.0, zorder=12)
        our_trail, = ax.plot([], [], 'g-', linewidth=2.5, alpha=0.7,
                            label='Our Path', zorder=11)
        our_trail_data = {'x': [], 'y': [], 'distance': 0.0}
        
        # Target visualization
        target_circle = Circle((0, 0), 1.0, facecolor='yellow',
                              edgecolor='orange', linewidth=2, zorder=10)
        ax.add_patch(target_circle)
        target_line, = ax.plot([], [], 'y--', linewidth=1.5, alpha=0.6, zorder=9)
        
        # Dynamic obstacles - using helper for consistency
        num_obstacles = len(self.manager.obstacles)
        colors = ['red', 'blue', 'orange', 'purple', 'gold', 'cyan', 'magenta']
        
        obstacle_patches = []  # [(body, bow), ...]
        obstacle_trails = []
        obstacle_trail_data = []
        obstacle_predictions = []
        
        for i in range(num_obstacles):
            color = colors[i % len(colors)]
            
            # Create vessel (10% smaller) - using helper
            body, bow = self._create_vessel_patches(ax, color, scale=0.9, zorder=10)
            obstacle_patches.append((body, bow))
            
            # Trail
            trail, = ax.plot([], [], '-', color=color, linewidth=1.5,
                           alpha=0.5, zorder=3)
            obstacle_trails.append(trail)
            obstacle_trail_data.append({'x': [], 'y': []})
            
            # Prediction circle
            pred_circle = Circle((0, 0), 1.0, facecolor='none',
                               edgecolor=color, linewidth=2,
                               linestyle='--', alpha=0.6, zorder=8)
            ax.add_patch(pred_circle)
            obstacle_predictions.append(pred_circle)
        
        # Axis setup
        ax.set_xlabel('X (cells)', fontsize=12)
        ax.set_ylabel('Y (cells)', fontsize=12)
        ax.set_title(f'Integrated Navigation - {self.controller_name}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.2)
        
        # Info boxes (reuse layout from path_animator)
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                          zorder=14, family='monospace')
        
        dynamics_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                              fontsize=9, verticalalignment='bottom',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),
                              zorder=14)
        
        warning_text = ax.text(0.50, 0.12, '', transform=ax.transAxes,
                             fontsize=11, verticalalignment='bottom',
                             horizontalalignment='center',
                             bbox=dict(boxstyle='round', facecolor='red', alpha=0.9),
                             color='white', fontweight='bold',
                             zorder=15)
        
        plt.tight_layout()
        
        def init():
            """Initialize animation."""
            artists = [our_body, our_bow, our_trail, target_circle, target_line]
            for body, bow in obstacle_patches:
                artists.extend([body, bow])
            artists.extend(obstacle_trails + obstacle_predictions)
            artists.extend([info_text, dynamics_text, warning_text])
            return artists
        
        def update(frame):
            """Update animation frame - reuses transform pattern."""
            if frame >= len(self.simulation_data):
                return init()
            
            state = self.simulation_data[frame]
            
            # Update our vessel (reuse transform pattern from path_animator)
            our_state = state['our_vessel']
            our_x, our_y = our_state['x'], our_state['y']
            our_heading = our_state['heading']
            
            transform = (Affine2D().rotate(our_heading).translate(our_x, our_y) + ax.transData)
            our_body.set_transform(transform)
            our_bow.set_transform(transform)
            
            # Update trail and distance (reuse from path_animator)
            our_trail_data['x'].append(our_x)
            our_trail_data['y'].append(our_y)
            our_trail.set_data(our_trail_data['x'], our_trail_data['y'])
            
            if len(our_trail_data['x']) > 1:
                dx = our_trail_data['x'][-1] - our_trail_data['x'][-2]
                dy = our_trail_data['y'][-1] - our_trail_data['y'][-2]
                our_trail_data['distance'] += np.sqrt(dx**2 + dy**2)
            
            # Update target
            if our_state['target']:
                tx, ty = our_state['target']
                target_circle.set_center((tx, ty))
                target_line.set_data([our_x, tx], [our_y, ty])
            
            # Update obstacles (reuse transform pattern from dynamic_animator)
            min_distance = float('inf')
            closest_id = None
            
            for i, obs_state in enumerate(state['dynamic_obstacles']):
                obs_x, obs_y = obs_state['x'], obs_state['y']
                obs_heading = obs_state['heading']
                
                # Distance tracking
                dist = np.sqrt((obs_x - our_x)**2 + (obs_y - our_y)**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_id = obs_state['id']
                
                # Update obstacle vessel (same transform pattern)
                obs_transform = (Affine2D().rotate(obs_heading).translate(obs_x, obs_y) + ax.transData)
                body, bow = obstacle_patches[i]
                body.set_transform(obs_transform)
                bow.set_transform(obs_transform)
                
                # Update trail
                obstacle_trail_data[i]['x'].append(obs_x)
                obstacle_trail_data[i]['y'].append(obs_y)
                obstacle_trails[i].set_data(obstacle_trail_data[i]['x'],
                                          obstacle_trail_data[i]['y'])
                
                # Update prediction
                if show_predictions:
                    pred_x, pred_y = obs_state['predicted']
                    obstacle_predictions[i].set_center((pred_x, pred_y))
            
            # Info text (reuse format from path_animator)
            heading_360 = (90 - np.degrees(our_heading)) % 360
            info_text.set_text(
                f'Time: {state["time"]:.1f}s\n'
                f'Pos: ({our_x:.1f}, {our_y:.1f})\n'
                f'Heading: {heading_360:.0f}°\n'
                f'Speed: {our_state["speed"]:.2f} units/s\n'
                f'Distance: {our_trail_data["distance"]:.1f} units\n'
                f'\n'
                f'Obstacles: {len(state["dynamic_obstacles"])}\n'
                f'Closest: {min_distance:.1f} units'
            )
            
            # Dynamics text (reuse from path_animator)
            rudder_angle = our_state.get('rudder_angle', 0.0)
            rate_of_turn = our_state.get('rate_of_turn', 0.0)
            dynamics_text.set_text(
                f'Rudder: {np.degrees(rudder_angle):+.1f}°\n'
                f'ROT: {np.degrees(rate_of_turn):+.2f}°/s'
            )
            
            # Collision warning
            if min_distance < 10.0:
                warning_text.set_text(f'⚠️  COLLISION RISK! Vessel {closest_id} at {min_distance:.1f} units')
            else:
                warning_text.set_text('')
            
            return init()
        
        # Speed up by showing every 3rd frame (3x faster animation)
        frame_skip = 3
        frames = range(0, len(self.simulation_data), frame_skip)
        print(f"  Rendering {len(frames)} frames (every {frame_skip} steps for speed)")
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=frames, interval=interval,
                           blit=False, repeat=repeat)
        
        plt.show()
        print("✓ Animation complete!")
