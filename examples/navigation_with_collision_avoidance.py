"""
Navigation with Collision Avoidance

Demonstrates autonomous vessel navigation with active collision avoidance:
- Real-time collision detection
- Automatic avoidance maneuvers following COLREGs-inspired rules
- Dynamic path adjustments
- Collision point tracking

Extends IntegratedNavigationAnimator to add active collision avoidance.
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
            safe_distance=10.0,  # Balanced for land and vessel avoidance
            warning_distance=18.0,  # Proportional warning distance
            avoidance_angle=np.radians(30),
            grid_world=grid_world
        )
        
        self.collision_data = []
        self.collision_points = []
        self.avoidance_history = []  # Track avoidance actions
        self.grounding_points = []  # Track grounding incidents
    
    def simulate(self, waypoints: List[Tuple[float, float]],
                duration: float, dt: float = 0.1):
        """Simulate with active collision avoidance."""
        self._print_simulation_header()
        
        self.simulation_data = []
        self.collision_data = []
        self.collision_points = []
        self.avoidance_history = []
        self.grounding_points = []
        self.controller.reset()
        
        steps = int(duration / dt)
        goal = waypoints[-1]
        
        step = 0
        avoidance_count = 0
        
        # Logging summary variables
        last_maneuver = None
        emergency_count = 0
        maneuver_changes = []
        
        while step < steps:
            # 1. Get current state
            our_state = self._get_our_vessel_state()
            
            # 2. Check goal
            dist_to_goal = np.sqrt((our_state['x'] - goal[0])**2 + (our_state['y'] - goal[1])**2)
            if dist_to_goal < 3.0:
                print(f"  ✓ Our vessel reached goal in {step} steps ({step * dt:.1f}s)")
                break
            
            # 3. Get navigation command
            desired_heading = self.controller.compute_desired_heading(
                (our_state['x'], our_state['y']), waypoints)
            
            if desired_heading is None:
                print(f"  ✓ Path complete at step {step}")
                break
            
            target = self.controller.get_target_point((our_state['x'], our_state['y']), waypoints)
            
            # Start with desired speed (maintain current)
            desired_speed = our_state['speed']
            
            # 4. Collision Detection & Avoidance
            collision_infos, avoidance_actions = self._process_collisions(
                our_state, step * dt, desired_heading)
            
            # 5. Apply Avoidance
            current_avoidance_reason = None
            if avoidance_actions:
                desired_heading, desired_speed = self.collision_avoider.apply_avoidance(
                    desired_heading, desired_speed, avoidance_actions
                )
                
                # Get primary action (list is sorted by apply_avoidance)
                primary_action = avoidance_actions[0]
                current_avoidance_reason = primary_action.reason
                avoidance_count += 1
                
                if step % 10 == 0:
                    self.avoidance_history.append({
                        'time': step * dt,
                        'action': primary_action,
                        'position': (our_state['x'], our_state['y'])
                    })
                    
                    # Track significant events for summary
                    if primary_action.type == 'emergency':
                        emergency_count += 1
                    
                    # Log only when maneuver type changes or during emergencies
                    current_maneuver = f"{primary_action.type}:{primary_action.reason[:30]}"
                    if current_maneuver != last_maneuver:
                        if primary_action.priority >= 4 or primary_action.type == 'emergency':
                            maneuver_changes.append(f"t={step*dt:.0f}s: {primary_action.reason[:50]}")
                        last_maneuver = current_maneuver
            
            # 6. Check for grounding
            gx, gy = int(our_state['x']), int(our_state['y'])
            if (gx >= 0 and gx < self.grid_world.width and 
                gy >= 0 and gy < self.grid_world.height):
                if self.grid_world.grid[gy, gx] > 0.5:
                    # Vessel is on land!
                    if not self.grounding_points or \
                       (step * dt - self.grounding_points[-1]['time']) > 1.0:
                        self.grounding_points.append({
                            'x': our_state['x'],
                            'y': our_state['y'],
                            'time': step * dt
                        })
            
            # 7. Record State
            self._record_simulation_step(step * dt, our_state, target, 
                                       collision_infos, avoidance_actions, current_avoidance_reason)
            
            # 8. Update Vessel & Obstacles
            self._update_vessel_control(dt, desired_heading, desired_speed)
            self.manager.update_all(dt)
            step += 1
        
        self._print_simulation_footer(avoidance_count, emergency_count, maneuver_changes)
        return self.simulation_data

    def _print_simulation_header(self):
        print(f"  Collision avoidance enabled:")
        print(f"    Safe distance: {self.collision_detector.safe_distance} units")
        print(f"    Warning distance: {self.collision_detector.warning_distance} units")
        print(f"    Avoidance angle: {np.degrees(self.collision_avoider.avoidance_angle):.0f}°")
        print(f"\n{'='*60}")
        print(f"Integrated Simulation with Avoidance: {self.controller_name}")
        print(f"  Our vessel + {len(self.manager.obstacles)} dynamic obstacles")
        print(f"{'='*60}")

    def _print_simulation_footer(self, avoidance_count, emergency_count, maneuver_changes):
        print(f"\n{'='*60}")
        print("SIMULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration: {len(self.simulation_data) * 0.1:.1f}s ({len(self.simulation_data)} frames)")
        print(f"Avoidance: {avoidance_count} steps ({avoidance_count*100//len(self.simulation_data)}% of time)")
        print(f"Emergency maneuvers: {emergency_count}")
        print(f"Collisions: {len(self.collision_points)}")
        print(f"Grounding incidents: {len(self.grounding_points)}")
        
        if maneuver_changes:
            print(f"\nKey Events ({len(maneuver_changes)} significant maneuvers):")
            # Show first 5 and last 5 if more than 10
            if len(maneuver_changes) > 10:
                for event in maneuver_changes[:5]:
                    print(f"  • {event}")
                print(f"  ... ({len(maneuver_changes) - 10} more events) ...")
                for event in maneuver_changes[-5:]:
                    print(f"  • {event}")
            else:
                for event in maneuver_changes:
                    print(f"  • {event}")
        
        if self.collision_points:
            print(f"\n💥 COLLISIONS: {len(self.collision_points)}")
            for cp in self.collision_points:
                print(f"  - t={cp['time']:.0f}s: Vessel {cp['vessel_id']} @ ({cp['x']:.0f},{cp['y']:.0f})")
        else:
            print(f"\n✓ NO COLLISIONS")
        
        if self.grounding_points:
            print(f"\n⚠️ GROUNDING: {len(self.grounding_points)} incidents")
            for gp in self.grounding_points:
                print(f"  - t={gp['time']:.0f}s: Vessel ran aground @ ({gp['x']:.0f},{gp['y']:.0f})")
        else:
            print(f"\n✓ NO GROUNDING")
        
        print(f"{'='*60}\n")

    def _get_our_vessel_state(self):
        x, y = self.our_vessel.get_position()
        return {
            'x': x, 'y': y,
            'heading': self.our_vessel.get_heading(),
            'speed': self.our_vessel.get_speed(),
            'rudder_angle': getattr(self.our_vessel, 'rudder_angle', 0.0),
            'rate_of_turn': getattr(self.our_vessel, 'get_turn_rate', lambda: 0.0)()
        }

    def _process_collisions(self, our_state, time, desired_heading=None):
        collision_infos = []
        avoidance_actions = []
        
        # Check Static Obstacles (Land) first
        static_action = self.collision_avoider.check_static_obstacles(
            pos=(our_state['x'], our_state['y']),
            heading=our_state['heading'],
            speed=our_state['speed']
        )
        if static_action:
            avoidance_actions.append(static_action)
        
        # Check each dynamic obstacle and determine avoidance per vessel
        for vessel in self.manager.obstacles:
            # Assess collision risk
            info = self.collision_detector.assess_collision_risk(
                pos1=(our_state['x'], our_state['y']),
                heading1=our_state['heading'],
                speed1=our_state['speed'],
                pos2=(vessel.obstacle.x, vessel.obstacle.y),
                heading2=vessel.obstacle.heading,
                speed2=vessel.obstacle.speed,
                vessel1_id=0,
                vessel2_id=vessel.obstacle.id
            )
            
            # Determine encounter type
            encounter_type = self.collision_detector.determine_encounter_type(
                info.relative_bearing, our_state['heading'], vessel.obstacle.heading
            )
            
            collision_infos.append({
                'info': info,
                'encounter_type': encounter_type,
                'vessel_id': vessel.obstacle.id,
                'obs_x': vessel.obstacle.x,
                'obs_y': vessel.obstacle.y,
                'obs_heading': vessel.obstacle.heading,
                'obs_speed': vessel.obstacle.speed
            })
            
            # Get avoidance action for this specific vessel
            if info.is_collision_risk:
                action = self.collision_avoider.determine_avoidance_action(
                    our_pos=(our_state['x'], our_state['y']),
                    our_heading=our_state['heading'],
                    our_speed=our_state['speed'],
                    collision_info=info,
                    encounter_type=encounter_type,
                    obstacle_pos=(vessel.obstacle.x, vessel.obstacle.y),
                    obstacle_heading=vessel.obstacle.heading
                )
                
                if action:
                    avoidance_actions.append(action)
            
            # Detect actual collisions
            if info.current_distance < 3.0:
                self._record_collision(our_state, time, vessel.obstacle.id, info.current_distance)
                
        return collision_infos, avoidance_actions

    def _record_collision(self, our_state, time, vessel_id, distance):
        collision_point = {
            'x': our_state['x'],
            'y': our_state['y'],
            'time': time,
            'vessel_id': vessel_id,
            'distance': distance
        }
        if not any(abs(cp['x'] - our_state['x']) < 1.0 and abs(cp['y'] - our_state['y']) < 1.0 
                 for cp in self.collision_points):
            self.collision_points.append(collision_point)

    def _record_simulation_step(self, time, our_state, target, collision_infos, avoidance_actions, avoidance_reason=None):
        state = {
            'time': time,
            'our_vessel': {
                **our_state,
                'target': target,
                'avoiding': len(avoidance_actions) > 0,
                'avoidance_reason': avoidance_reason
            },
            'dynamic_obstacles': []
        }
        
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

    def _update_vessel_control(self, dt, desired_heading, desired_speed):
        # Update heading with PD controller
        heading_error = desired_heading - self.our_vessel.get_heading()
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        vessel_type = type(self.our_vessel).__name__
        if vessel_type == 'NomotoVessel':
            Kp = 4.0
            Kd = 2.5
            yaw_rate = self.our_vessel.get_turn_rate()
            rudder_command = (Kp * heading_error) - (Kd * yaw_rate)
            # Pass desired_speed to vessel update for proper dynamics integration
            self.our_vessel.update(dt, rudder_command=rudder_command)
            # Update speed after dynamics (simpler approach for now)
            if hasattr(self.our_vessel, 'speed'):
                self.our_vessel.speed = desired_speed
        else:
            self.our_vessel.update(dt, desired_heading=desired_heading)
            if hasattr(self.our_vessel, 'speed'):
                self.our_vessel.speed = desired_speed
    
    def animate(self, waypoints: List[Tuple[float, float]], 
               interval: int = 5, show_predictions: bool = True, repeat: bool = False):
        """Animate with collision avoidance visualization."""
        if not self.simulation_data or not self.collision_data:
            print("❌ No simulation data! Run simulate() first.")
            return
        
        print("\nCreating animation with collision avoidance...")
        
        fig, ax_main, ax_info, visuals, state_data = self._setup_animation_visuals(waypoints)
        
        def init():
            return self._init_animation(visuals)
        
        def update(frame):
            return self._update_animation(frame, ax_main, visuals, state_data)
        
        # Speed up animation
        frame_skip = 3
        frames = range(0, len(self.simulation_data), frame_skip)
        print(f"  Rendering {len(frames)} frames (every {frame_skip} steps)")
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=frames, interval=interval,
                           blit=False, repeat=repeat)
        
        plt.show()
        print("✓ Animation complete!")

    def _setup_animation_visuals(self, waypoints):
        # Create figure with info panel
        fig = plt.figure(figsize=(18, 12))
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
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
        
        # Our vessel
        our_body, our_bow = self._create_vessel_patches(ax_main, 'green', scale=1.0, zorder=12)
        our_trail, = ax_main.plot([], [], 'g-', linewidth=2.5, alpha=0.7,
                                 label='Actual Path', zorder=11)
        
        # Target
        target_circle = Circle((0, 0), 1.0, facecolor='yellow',
                              edgecolor='orange', linewidth=2, zorder=10)
        ax_main.add_patch(target_circle)
        target_line, = ax_main.plot([], [], 'y--', linewidth=1.5, alpha=0.6, zorder=9)
        
        # Dynamic obstacles
        obstacle_patches = []
        obstacle_trails = []
        cpa_lines = []
        colors = ['red', 'blue', 'orange', 'purple', 'gold', 'cyan', 'magenta']
        
        for i in range(len(self.manager.obstacles)):
            color = colors[i % len(colors)]
            body, bow = self._create_vessel_patches(ax_main, color, scale=0.9, zorder=10)
            obstacle_patches.append((body, bow))
            trail, = ax_main.plot([], [], '-', color=color, linewidth=1.5, alpha=0.5, zorder=3)
            obstacle_trails.append(trail)
            cpa_line, = ax_main.plot([], [], 'r--', linewidth=2, alpha=0.7, zorder=9)
            cpa_lines.append(cpa_line)
            
        # Collision points
        for cp in self.collision_points:
            ax_main.plot(cp['x'], cp['y'], 'rX', markersize=20, markeredgewidth=3, zorder=15)
            ax_main.add_patch(Circle((cp['x'], cp['y']), 3.0, facecolor='red',
                                   edgecolor='darkred', linewidth=2, alpha=0.3, zorder=14))
            
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
        
        visuals = {
            'our_body': our_body, 'our_bow': our_bow, 'our_trail': our_trail,
            'target_circle': target_circle, 'target_line': target_line,
            'obstacle_patches': obstacle_patches, 'obstacle_trails': obstacle_trails,
            'cpa_lines': cpa_lines,
            'status_text': status_text, 'dynamics_text': dynamics_text,
            'avoidance_text': avoidance_text
        }
        
        state_data = {
            'our_trail': {'x': [], 'y': [], 'distance': 0.0},
            'obstacle_trails': [{'x': [], 'y': []} for _ in range(len(self.manager.obstacles))]
        }
        
        return fig, ax_main, ax_info, visuals, state_data

    def _init_animation(self, visuals):
        artists = [visuals['our_body'], visuals['our_bow'], visuals['our_trail'],
                  visuals['target_circle'], visuals['target_line']]
        for body, bow in visuals['obstacle_patches']:
            artists.extend([body, bow])
        artists.extend(visuals['obstacle_trails'])
        artists.extend(visuals['cpa_lines'])
        artists.extend([visuals['status_text'], visuals['dynamics_text'], visuals['avoidance_text']])
        return artists

    def _update_animation(self, frame, ax_main, visuals, state_data):
        if frame >= len(self.simulation_data):
            return self._init_animation(visuals)
        
        state = self.simulation_data[frame]
        collision_infos = self.collision_data[frame]
        our_state = state['our_vessel']
        
        # Update our vessel
        self._update_vessel_visuals(our_state, visuals, state_data, ax_main)
        
        # Update obstacles
        self._update_obstacles_visuals(state['dynamic_obstacles'], visuals, state_data, ax_main)
        
        # Update info panels
        self._update_info_panels(our_state, collision_infos, visuals, state_data)
        
        return self._init_animation(visuals)

    def _update_vessel_visuals(self, our_state, visuals, state_data, ax_main):
        x, y, heading = our_state['x'], our_state['y'], our_state['heading']
        
        # Transform
        transform = (Affine2D().rotate(heading).translate(x, y) + ax_main.transData)
        visuals['our_body'].set_transform(transform)
        visuals['our_bow'].set_transform(transform)
        
        # Color
        color = 'yellow' if our_state.get('avoiding', False) else 'green'
        visuals['our_body'].set_facecolor(color)
        visuals['our_bow'].set_facecolor(color)
        
        # Trail
        trail = state_data['our_trail']
        trail['x'].append(x)
        trail['y'].append(y)
        visuals['our_trail'].set_data(trail['x'], trail['y'])
        
        if len(trail['x']) > 1:
            dx = trail['x'][-1] - trail['x'][-2]
            dy = trail['y'][-1] - trail['y'][-2]
            trail['distance'] += np.sqrt(dx**2 + dy**2)
            
        # Target
        if our_state['target']:
            tx, ty = our_state['target']
            visuals['target_circle'].set_center((tx, ty))
            visuals['target_line'].set_data([x, tx], [y, ty])

    def _update_obstacles_visuals(self, obstacles, visuals, state_data, ax_main):
        for i, obs in enumerate(obstacles):
            # Transform
            transform = (Affine2D().rotate(obs['heading']).translate(obs['x'], obs['y']) + ax_main.transData)
            body, bow = visuals['obstacle_patches'][i]
            body.set_transform(transform)
            bow.set_transform(transform)
            
            # Trail
            trail = state_data['obstacle_trails'][i]
            trail['x'].append(obs['x'])
            trail['y'].append(obs['y'])
            visuals['obstacle_trails'][i].set_data(trail['x'], trail['y'])

    def _update_info_panels(self, our_state, collision_infos, visuals, state_data):
        # Avoidance Text
        info_lines = ['COLLISION AVOIDANCE\n' + '='*30 + '\n\n']
        if our_state.get('avoiding', False):
            info_lines.append('🚨 AVOIDING!\n')
            if our_state.get('avoidance_reason'):
                reason = our_state['avoidance_reason']
                # Split long lines
                if len(reason) > 35:
                    reason = reason[:35] + '...'
                info_lines.append(f"{reason}\n\n")
            else:
                info_lines.append('\n')
            
        highest_risk = 0
        risk_names = ['SAFE', 'LOW', 'MEDIUM', 'HIGH']
        
        for i, coll_info in enumerate(collision_infos):
            info = coll_info['info']
            risk_level = info.risk_level
            highest_risk = max(highest_risk, risk_level)
            
            info_lines.append(f'Vessel {coll_info["vessel_id"]}:\n')
            info_lines.append(f'  Type: {coll_info["encounter_type"].upper()}\n')
            info_lines.append(f'  CPA: {info.cpa_distance:.1f}u, TCPA: {info.tcpa:.1f}s\n')
            info_lines.append(f'  Risk: {risk_names[risk_level]}\n\n')
            
            # Update CPA lines
            if risk_level >= 2 and info.tcpa > 0:
                # Calculate CPA points (simplified for visualization)
                # ... (omitted for brevity, using current pos + vel * tcpa)
                vel1_x = our_state['speed'] * np.cos(our_state['heading'])
                vel1_y = our_state['speed'] * np.sin(our_state['heading'])
                vel2_x = coll_info['obs_speed'] * np.cos(coll_info['obs_heading'])
                vel2_y = coll_info['obs_speed'] * np.sin(coll_info['obs_heading'])
                
                cpa_our_x = our_state['x'] + vel1_x * info.tcpa
                cpa_our_y = our_state['y'] + vel1_y * info.tcpa
                cpa_obs_x = coll_info['obs_x'] + vel2_x * info.tcpa
                cpa_obs_y = coll_info['obs_y'] + vel2_y * info.tcpa
                
                visuals['cpa_lines'][i].set_data([cpa_our_x, cpa_obs_x], [cpa_our_y, cpa_obs_y])
            else:
                visuals['cpa_lines'][i].set_data([], [])
                
        info_lines.append(f'Collisions: {len(self.collision_points)}\n')
        info_lines.append(f'Avoidance actions: {len(self.avoidance_history)}')
        
        visuals['avoidance_text'].set_text(''.join(info_lines))
        risk_bg_colors = ['lightgreen', 'lightyellow', 'orange', 'red']
        visuals['avoidance_text'].get_bbox_patch().set_facecolor(risk_bg_colors[highest_risk])
        
        # Status Text
        heading_360 = (90 - np.degrees(our_state['heading'])) % 360
        visuals['status_text'].set_text(
            f'Time: {our_state.get("time", 0):.1f}s\n' # Time might not be in our_state if not passed, but it is in state
            f'Pos: ({our_state["x"]:.1f}, {our_state["y"]:.1f})\n'
            f'Heading: {heading_360:.0f}°\n'
            f'Speed: {our_state["speed"]:.2f} units/s\n'
            f'Distance: {state_data["our_trail"]["distance"]:.1f} units\n'
            f'{"⚠️ AVOIDING" if our_state.get("avoiding") else ""}'
        )
        
        # Dynamics Text
        visuals['dynamics_text'].set_text(
            f'Rudder: {np.degrees(our_state.get("rudder_angle", 0)):+.1f}°\n'
            f'ROT: {np.degrees(our_state.get("rate_of_turn", 0)):+.2f}°/s'
        )


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
    
    # Obstacle 0: Crossing from West to East (South area)
    # Will cross our path as we leave the start area
    vessel1 = manager.add_obstacle(x=30, y=30, heading=0.0, speed=0.1, behavior='straight')
    print(f"  Added {vessel1.obstacle} (Crossing)")
    
    # Obstacle 1: Head-on in the narrow channel
    # Coming down from North-East towards South-West
    vessel2 = manager.add_obstacle(x=55, y=100, heading=np.radians(280), speed=0.2, behavior='straight')
    print(f"  Added {vessel2.obstacle} (Head-on in channel)")
    
    # Removed Obstacle 2 (waypoint patrol) to reduce complexity and avoid narrow passages
    
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
    # Increased duration to ensure vessel reaches goal
    sim.simulate(waypoints, duration=500.0, dt=0.1)
    
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
