"""
Navigation with Collision Avoidance

Demonstrates autonomous vessel navigation with active collision avoidance:
- Real-time collision detection
- Automatic avoidance maneuvers following COLREGs-inspired rules
- Dynamic path adjustments
- Collision point tracking

Extends NavigationWithCollisionDetection to add active avoidance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D
from typing import List, Tuple, Dict

from src.environment.grid_world import GridWorld
from src.pathfinding.astar import AStar, create_test_scenario
from src.vessel.vessel_model import NomotoVessel
from src.vessel.path_follower import PurePursuitController, LOSController
from src.environment.dynamic_obstacles import DynamicObstacleManager
from src.environment.collision_detection import CollisionDetector
from src.vessel.collision_avoidance import CollisionAvoidance
from src.visualization.integrated_animator import IntegratedNavigationAnimator


class NavigationWithCollisionAvoidance(IntegratedNavigationAnimator):
    """Extends IntegratedNavigationAnimator with active collision avoidance."""
    
    def __init__(self, grid_world, our_vessel, controller,
                 obstacle_manager, controller_name: str):
        """Initialize with collision detector and avoidance system."""
        super().__init__(grid_world, our_vessel, controller, 
                        obstacle_manager, controller_name)
        
        self.collision_detector = CollisionDetector(
            safe_distance=8.0,
            warning_distance=15.0,
            time_horizon=60.0
        )
        
        self.collision_avoider = CollisionAvoidance(
            safe_distance=8.0,
            warning_distance=15.0,
            avoidance_angle=np.radians(30)
        )
        
        self.collision_data = []
        self.collision_points = []
        self.avoidance_history = []  # Track avoidance actions
    
    def simulate(self, waypoints: List[Tuple[float, float]],
                duration: float, dt: float = 0.1):
        """Simulate with active collision avoidance."""
        print(f"  Collision avoidance enabled:")
        print(f"    Safe distance: {self.collision_detector.safe_distance} units")
        print(f"    Warning distance: {self.collision_detector.warning_distance} units")
        print(f"    Avoidance angle: {np.degrees(self.collision_avoider.avoidance_angle):.0f}°")
        
        print(f"\n{'='*60}")
        print(f"Integrated Simulation with Avoidance: {self.controller_name}")
        print(f"  Our vessel + {len(self.manager.obstacles)} dynamic obstacles")
        print(f"{'='*60}")
        
        self.simulation_data = []
        self.collision_data = []
        self.collision_points = []
        self.avoidance_history = []
        self.controller.reset()
        
        steps = int(duration / dt)
        max_steps = steps
        goal = waypoints[-1]
        
        step = 0
        avoidance_count = 0
        
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
            
            # Collision detection and avoidance
            collision_infos = []
            avoidance_actions = []
            
            for vessel in self.manager.obstacles:
                obs_x = vessel.obstacle.x
                obs_y = vessel.obstacle.y
                obs_heading = vessel.obstacle.heading
                obs_speed = vessel.obstacle.speed
                
                # Assess collision risk
                info = self.collision_detector.assess_collision_risk(
                    pos1=(our_x, our_y),
                    heading1=our_heading,
                    speed1=our_speed,
                    pos2=(obs_x, obs_y),
                    heading2=obs_heading,
                    speed2=obs_speed,
                    vessel1_id=0,
                    vessel2_id=vessel.obstacle.id
                )
                
                # Determine encounter type
                encounter_type = self.collision_detector.determine_encounter_type(
                    info.relative_bearing, our_heading, obs_heading
                )
                
                collision_infos.append({
                    'info': info,
                    'encounter_type': encounter_type,
                    'vessel_id': vessel.obstacle.id,
                    'obs_x': obs_x,
                    'obs_y': obs_y,
                    'obs_heading': obs_heading,
                    'obs_speed': obs_speed
                })
                
                # Determine avoidance action if needed
                avoidance_action = self.collision_avoider.determine_avoidance_action(
                    our_pos=(our_x, our_y),
                    our_heading=our_heading,
                    our_speed=our_speed,
                    collision_info=info,
                    encounter_type=encounter_type,
                    obstacle_pos=(obs_x, obs_y),
                    obstacle_heading=obs_heading
                )
                
                if avoidance_action:
                    avoidance_actions.append(avoidance_action)
                
                # Detect actual collisions
                if info.current_distance < 3.0:
                    collision_point = {
                        'x': our_x,
                        'y': our_y,
                        'time': step * dt,
                        'vessel_id': vessel.obstacle.id,
                        'distance': info.current_distance
                    }
                    if not any(abs(cp['x'] - our_x) < 1.0 and abs(cp['y'] - our_y) < 1.0 
                             for cp in self.collision_points):
                        self.collision_points.append(collision_point)
            
            # Apply avoidance maneuvers
            if avoidance_actions:
                desired_heading, our_speed = self.collision_avoider.apply_avoidance(
                    desired_heading, our_speed, avoidance_actions
                )
                avoidance_count += 1
                
                # Record avoidance action
                if step % 10 == 0:  # Log every 10 steps
                    self.avoidance_history.append({
                        'time': step * dt,
                        'action': avoidance_actions[0],
                        'position': (our_x, our_y)
                    })
            
            # Record state
            state = {
                'time': step * dt,
                'our_vessel': {
                    'x': our_x,
                    'y': our_y,
                    'heading': our_heading,
                    'speed': our_speed,
                    'target': target,
                    'rudder_angle': rudder_angle,
                    'rate_of_turn': rate_of_turn,
                    'avoiding': len(avoidance_actions) > 0
                },
                'dynamic_obstacles': []
            }
            
            # Record dynamic obstacles
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
            
            self.simulation_data.append(state)
            self.collision_data.append(collision_infos)
            
            # Update our vessel with avoidance-modified heading
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
            
            # Update speed if modified by avoidance
            if hasattr(self.our_vessel, 'speed'):
                self.our_vessel.speed = our_speed
            
            # Update dynamic obstacles
            self.manager.update_all(dt)
            step += 1
        
        print(f"  ✓ Simulation complete: {len(self.simulation_data)} frames")
        print(f"  ⚠️  Avoidance maneuvers: {avoidance_count} steps")
        
        if self.collision_points:
            print(f"  💥 COLLISIONS DETECTED: {len(self.collision_points)}")
            for cp in self.collision_points:
                print(f"    - t={cp['time']:.1f}s: Vessel {cp['vessel_id']} at ({cp['x']:.1f}, {cp['y']:.1f}), dist={cp['distance']:.1f}")
        else:
            print(f"  ✓ NO COLLISIONS - Avoidance successful!")
        
        print(f"{'='*60}\n")
        return self.simulation_data
    
    def animate(self, waypoints: List[Tuple[float, float]], 
               interval: int = 5, show_predictions: bool = True, repeat: bool = False):
        """Animate with collision avoidance visualization."""
        if not self.simulation_data or not self.collision_data:
            print("❌ No simulation data! Run simulate() first.")
            return
        
        print("\nCreating animation with collision avoidance...")
        
        # Create figure with info panel
        fig = plt.figure(figsize=(18, 12))
        
        # Main plot
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        
        # Info panel
        ax_info = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
        ax_info.axis('off')
        
        # Plot environment
        ax_main.imshow(self.grid_world.grid, cmap='Blues', origin='lower',
                      vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
        
        # Plot planned path
        wp_x, wp_y = [w[0] for w in waypoints], [w[1] for w in waypoints]
        ax_main.plot(wp_x, wp_y, 'g--', linewidth=2, alpha=0.5, label='Planned Path')
        ax_main.plot(wp_x[0], wp_y[0], 'go', markersize=15,
                    markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
        ax_main.plot(wp_x[-1], wp_y[-1], 'r*', markersize=20,
                    markeredgecolor='darkred', markeredgewidth=2, label='Goal')
        
        # Our vessel - reuse parent's helper
        our_body, our_bow = self._create_vessel_patches(ax_main, 'green', scale=1.0, zorder=12)
        our_trail, = ax_main.plot([], [], 'g-', linewidth=2.5, alpha=0.7,
                                 label='Actual Path', zorder=11)
        our_trail_data = {'x': [], 'y': [], 'distance': 0.0}
        
        # Target
        target_circle = Circle((0, 0), 1.0, facecolor='yellow',
                              edgecolor='orange', linewidth=2, zorder=10)
        ax_main.add_patch(target_circle)
        target_line, = ax_main.plot([], [], 'y--', linewidth=1.5, alpha=0.6, zorder=9)
        
        # Dynamic obstacles
        num_obstacles = len(self.manager.obstacles)
        colors = ['red', 'blue', 'orange', 'purple', 'gold', 'cyan', 'magenta']
        
        obstacle_patches = []
        obstacle_trails = []
        obstacle_trail_data = []
        cpa_lines = []
        
        # Collision points
        collision_markers = []
        for cp in self.collision_points:
            marker = ax_main.plot(cp['x'], cp['y'], 'rX', markersize=20, 
                                 markeredgewidth=3, label='Collision!' if not collision_markers else '',
                                 zorder=15)[0]
            collision_markers.append(marker)
            coll_circle = Circle((cp['x'], cp['y']), 3.0, facecolor='red',
                               edgecolor='darkred', linewidth=2, alpha=0.3, zorder=14)
            ax_main.add_patch(coll_circle)
        
        for i in range(num_obstacles):
            color = colors[i % len(colors)]
            
            body, bow = self._create_vessel_patches(ax_main, color, scale=0.9, zorder=10)
            obstacle_patches.append((body, bow))
            
            trail, = ax_main.plot([], [], '-', color=color, linewidth=1.5,
                                alpha=0.5, zorder=3)
            obstacle_trails.append(trail)
            obstacle_trail_data.append({'x': [], 'y': []})
            
            cpa_line, = ax_main.plot([], [], 'r--', linewidth=2, alpha=0.7, zorder=9)
            cpa_lines.append(cpa_line)
        
        # Axis setup
        ax_main.set_xlabel('X (cells)', fontsize=12)
        ax_main.set_ylabel('Y (cells)', fontsize=12)
        ax_main.set_title(f'Navigation with Collision Avoidance - {self.controller_name}',
                         fontsize=14, fontweight='bold')
        ax_main.legend(loc='upper left', fontsize=9)
        ax_main.set_aspect('equal')
        ax_main.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.2)
        
        # Info texts
        status_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                                  fontsize=10, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                  zorder=14, family='monospace')
        
        dynamics_text = ax_main.text(0.02, 0.02, '', transform=ax_main.transAxes,
                                    fontsize=9, verticalalignment='bottom',
                                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),
                                    zorder=14)
        
        avoidance_text = ax_info.text(0.05, 0.95, '', verticalalignment='top',
                                     fontsize=9, family='monospace',
                                     bbox=dict(boxstyle='round', facecolor='lightyellow',
                                             edgecolor='orange', linewidth=2))
        
        plt.tight_layout()
        
        def init():
            """Initialize animation."""
            artists = [our_body, our_bow, our_trail, target_circle, target_line]
            for body, bow in obstacle_patches:
                artists.extend([body, bow])
            artists.extend(obstacle_trails + cpa_lines + [status_text, dynamics_text, avoidance_text])
            return artists
        
        def update(frame):
            """Update animation frame."""
            if frame >= len(self.simulation_data):
                return init()
            
            state = self.simulation_data[frame]
            collision_infos = self.collision_data[frame]
            
            # Update our vessel
            our_state = state['our_vessel']
            our_x, our_y = our_state['x'], our_state['y']
            our_heading = our_state['heading']
            
            transform = (Affine2D().rotate(our_heading).translate(our_x, our_y) + ax_main.transData)
            our_body.set_transform(transform)
            our_bow.set_transform(transform)
            
            # Change vessel color if avoiding
            if our_state.get('avoiding', False):
                our_body.set_facecolor('yellow')
                our_bow.set_facecolor('yellow')
            else:
                our_body.set_facecolor('green')
                our_bow.set_facecolor('green')
            
            # Update trail
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
            
            # Update obstacles
            for i, obs_state in enumerate(state['dynamic_obstacles']):
                obs_x, obs_y = obs_state['x'], obs_state['y']
                obs_heading = obs_state['heading']
                
                obs_transform = (Affine2D().rotate(obs_heading).translate(obs_x, obs_y) + ax_main.transData)
                body, bow = obstacle_patches[i]
                body.set_transform(obs_transform)
                bow.set_transform(obs_transform)
                
                obstacle_trail_data[i]['x'].append(obs_x)
                obstacle_trail_data[i]['y'].append(obs_y)
                obstacle_trails[i].set_data(obstacle_trail_data[i]['x'],
                                          obstacle_trail_data[i]['y'])
            
            # Update collision info
            info_lines = ['COLLISION AVOIDANCE\n' + '='*30 + '\n\n']
            highest_risk = 0
            
            if our_state.get('avoiding', False):
                info_lines.append('🚨 AVOIDING!\n\n')
            
            for coll_info in collision_infos:
                info = coll_info['info']
                encounter = coll_info['encounter_type']
                vessel_id = coll_info['vessel_id']
                
                risk_level = info.risk_level
                highest_risk = max(highest_risk, risk_level)
                
                risk_names = ['SAFE', 'LOW', 'MEDIUM', 'HIGH']
                
                info_lines.append(f'Vessel {vessel_id}:\n')
                info_lines.append(f'  Type: {encounter.upper()}\n')
                info_lines.append(f'  Current: {info.current_distance:.1f}u\n')
                info_lines.append(f'  CPA: {info.cpa_distance:.1f}u\n')
                info_lines.append(f'  TCPA: {info.tcpa:.1f}s\n')
                info_lines.append(f'  Risk: {risk_names[risk_level]}\n')
                info_lines.append('\n')
                
                # Draw CPA lines for high-risk
                if risk_level >= 2 and info.tcpa > 0:
                    vel1_x = our_state['speed'] * np.cos(our_heading)
                    vel1_y = our_state['speed'] * np.sin(our_heading)
                    vel2_x = coll_info['obs_speed'] * np.cos(coll_info['obs_heading'])
                    vel2_y = coll_info['obs_speed'] * np.sin(coll_info['obs_heading'])
                    
                    cpa_our_x = our_x + vel1_x * info.tcpa
                    cpa_our_y = our_y + vel1_y * info.tcpa
                    cpa_obs_x = coll_info['obs_x'] + vel2_x * info.tcpa
                    cpa_obs_y = coll_info['obs_y'] + vel2_y * info.tcpa
                    
                    cpa_lines[vessel_id].set_data([cpa_our_x, cpa_obs_x],
                                                 [cpa_our_y, cpa_obs_y])
                else:
                    cpa_lines[vessel_id].set_data([], [])
            
            info_lines.append(f'\nCollisions: {len(self.collision_points)}\n')
            info_lines.append(f'Avoidance actions: {len(self.avoidance_history)}')
            
            avoidance_text.set_text(''.join(info_lines))
            
            # Color by risk
            risk_bg_colors = ['lightgreen', 'lightyellow', 'orange', 'red']
            avoidance_text.get_bbox_patch().set_facecolor(risk_bg_colors[highest_risk])
            
            # Status text
            heading_360 = (90 - np.degrees(our_heading)) % 360
            status_text.set_text(
                f'Time: {state["time"]:.1f}s\n'
                f'Pos: ({our_x:.1f}, {our_y:.1f})\n'
                f'Heading: {heading_360:.0f}°\n'
                f'Speed: {our_state["speed"]:.2f} units/s\n'
                f'Distance: {our_trail_data["distance"]:.1f} units\n'
                f'{"⚠️ AVOIDING" if our_state.get("avoiding") else ""}'
            )
            
            # Dynamics text
            rudder_angle = our_state.get('rudder_angle', 0.0)
            rate_of_turn = our_state.get('rate_of_turn', 0.0)
            dynamics_text.set_text(
                f'Rudder: {np.degrees(rudder_angle):+.1f}°\n'
                f'ROT: {np.degrees(rate_of_turn):+.2f}°/s'
            )
            
            return init()
        
        # Speed up animation
        frame_skip = 3
        frames = range(0, len(self.simulation_data), frame_skip)
        print(f"  Rendering {len(frames)} frames (every {frame_skip} steps)")
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=frames, interval=interval,
                           blit=False, repeat=repeat)
        
        plt.show()
        print("✓ Animation complete!")


def main():
    """Main demo."""
    print("=" * 70)
    print("NAVIGATION WITH COLLISION AVOIDANCE")
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
    
    # Obstacle 0: Crossing from west
    vessel1 = manager.add_obstacle(x=5, y=50, heading=0, speed=1.5, behavior='straight')
    print(f"  Added {vessel1.obstacle}")
    
    # Obstacle 1: Moving north (collision course!)
    vessel2 = manager.add_obstacle(x=40, y=10, heading=np.pi/2, speed=1.2, behavior='straight')
    print(f"  Added {vessel2.obstacle}")
    
    # Obstacle 2: Circular patrol
    vessel3 = manager.add_obstacle(x=50, y=50, heading=0, speed=1.0, behavior='circular')
    vessel3.set_circular_path(center=(50, 50), radius=10.0)
    print(f"  Added {vessel3.obstacle} (circular)")
    
    # Create our vessel
    print("\n" + "=" * 70)
    print("CREATING OUR VESSEL")
    print("=" * 70)
    
    our_vessel = NomotoVessel(x=float(start[0]), y=float(start[1]),
                             heading=0.0, speed=0.5, K=0.5, T=3.0)
    
    controller = LOSController(lookahead_distance=8.0, path_tolerance=3.0)
    controller_name = "LOS"
    
    print(f"  Using {controller_name} controller")
    
    # Simulate
    print("\n" + "=" * 70)
    print("SIMULATING WITH COLLISION AVOIDANCE...")
    print("=" * 70)
    
    sim = NavigationWithCollisionAvoidance(grid, our_vessel, controller,
                                          manager, controller_name)
    sim.simulate(waypoints, duration=120.0, dt=0.1)
    
    # Animate
    print("\n" + "=" * 70)
    print("ANIMATION STARTING...")
    print("=" * 70)
    print("\nAnimation features:")
    print("  🟢 Green vessel      = Our vessel following path")
    print("  🟡 Yellow vessel     = Our vessel AVOIDING collision")
    print("  📊 Right panel: Real-time collision & avoidance info")
    print("  🔴 Red dashed lines: CPA points")
    print("  ❌ Red X markers: Collision points (if any)")
    print("\nWatch for avoidance maneuvers!")
    print("=" * 70)
    
    sim.animate(waypoints, interval=5)
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("\n1. Vessel actively avoids collisions (turns yellow)")
    print("2. COLREGs-inspired rules applied:")
    print("   - Head-on: Both turn starboard (right)")
    print("   - Crossing from starboard: Give way (turn right)")
    print("   - Crossing from port: Stand on (maintain course)")
    print("3. Avoidance maneuvers shown in real-time")
    print("4. Collision count displayed")
    print(f"5. Final result: {'✓ SUCCESS!' if not sim.collision_points else '⚠️ COLLISIONS OCCURRED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
