"""
Navigation with Collision Detection

Shows our vessel with real-time collision detection:
- CPA/TCPA calculations for all dynamic obstacles
- Risk assessment
- Encounter type identification
- Visual collision warnings

Extends IntegratedNavigationAnimator with collision detection overlay.
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
from src.visualization.integrated_animator import IntegratedNavigationAnimator


class NavigationWithCollisionDetection(IntegratedNavigationAnimator):
    """Extends IntegratedNavigationAnimator with collision detection."""
    
    def __init__(self, grid_world, our_vessel, controller,
                 obstacle_manager, controller_name: str):
        """Initialize with collision detector."""
        super().__init__(grid_world, our_vessel, controller, 
                        obstacle_manager, controller_name)
        self.collision_detector = CollisionDetector(
            safe_distance=8.0,
            warning_distance=15.0,
            time_horizon=60.0
        )
        self.collision_data = []
        self.collision_points = []  # Track actual collision points
    
    def simulate(self, waypoints: List[Tuple[float, float]],
                duration: float, dt: float = 0.1):
        """Simulate with collision detection - extends parent simulate."""
        print(f"  Collision detection enabled:")
        print(f"    Safe distance: {self.collision_detector.safe_distance} units")
        print(f"    Warning distance: {self.collision_detector.warning_distance} units")
        
        # Use parent's simulate method
        super().simulate(waypoints, duration, dt)
        
        # Enhance simulation data with collision info
        collision_events = []
        self.collision_data = []
        
        for state in self.simulation_data:
            our_state = state['our_vessel']
            our_x, our_y = our_state['x'], our_state['y']
            our_heading, our_speed = our_state['heading'], our_state['speed']
            
            # Calculate collision info for each obstacle
            collision_infos = []
            
            for obs_state in state['dynamic_obstacles']:
                obs_x, obs_y = obs_state['x'], obs_state['y']
                obs_heading = obs_state['heading']
                obs_speed = obs_state['speed']
                
                # Assess collision risk
                info = self.collision_detector.assess_collision_risk(
                    pos1=(our_x, our_y),
                    heading1=our_heading,
                    speed1=our_speed,
                    pos2=(obs_x, obs_y),
                    heading2=obs_heading,
                    speed2=obs_speed,
                    vessel1_id=0,
                    vessel2_id=obs_state['id']
                )
                
                # Determine encounter type
                encounter_type = self.collision_detector.determine_encounter_type(
                    info.relative_bearing, our_heading, obs_heading
                )
                
                collision_infos.append({
                    'info': info,
                    'encounter_type': encounter_type,
                    'vessel_id': obs_state['id'],
                    'obs_x': obs_x,
                    'obs_y': obs_y,
                    'obs_heading': obs_heading,
                    'obs_speed': obs_speed
                })
                
                # Log high-risk encounters and detect actual collisions
                if info.risk_level == 3:
                    event_desc = (f"HIGH RISK with vessel {obs_state['id']} "
                                f"(CPA={info.cpa_distance:.1f} in {info.tcpa:.1f}s)")
                    if event_desc not in collision_events:
                        collision_events.append(event_desc)
                
                # Detect actual collision (within 3 units)
                if info.current_distance < 3.0:
                    collision_point = {
                        'x': our_x,
                        'y': our_y,
                        'time': state['time'],
                        'vessel_id': obs_state['id'],
                        'distance': info.current_distance
                    }
                    # Only add if not already recorded at this position
                    if not any(abs(cp['x'] - our_x) < 1.0 and abs(cp['y'] - our_y) < 1.0 
                             for cp in self.collision_points):
                        self.collision_points.append(collision_point)
            
            self.collision_data.append(collision_infos)
        
        if collision_events:
            print(f"\n  ⚠️  High-risk events detected: {len(collision_events)}")
            for event in collision_events[:3]:  # Show first 3
                print(f"    - {event}")
        
        if self.collision_points:
            print(f"\n  💥 COLLISIONS DETECTED: {len(self.collision_points)}")
            for cp in self.collision_points:
                print(f"    - t={cp['time']:.1f}s: Vessel {cp['vessel_id']} at ({cp['x']:.1f}, {cp['y']:.1f}), dist={cp['distance']:.1f}")
        
        return self.simulation_data
    
    def animate(self, waypoints: List[Tuple[float, float]], 
               interval: int = 5, show_predictions: bool = True, repeat: bool = False):
        """
        Animate with collision detection overlay.
        Extends parent animator with collision info panel.
        """
        if not self.simulation_data or not self.collision_data:
            print("❌ No simulation data! Run simulate() first.")
            return
        
        print("\nCreating animation with collision detection overlay...")
        
        # Create figure with collision info panel
        fig = plt.figure(figsize=(18, 12))
        
        # Main plot (reuse parent's setup)
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        
        # Collision info panel
        ax_info = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
        ax_info.axis('off')
        
        # Plot environment using parent's pattern
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
                                 label='Our Path', zorder=11)
        our_trail_data = {'x': [], 'y': []}
        
        # Target
        target_circle = Circle((0, 0), 1.0, facecolor='yellow',
                              edgecolor='orange', linewidth=2, zorder=10)
        ax_main.add_patch(target_circle)
        target_line, = ax_main.plot([], [], 'y--', linewidth=1.5, alpha=0.6, zorder=9)
        
        # Dynamic obstacles - reuse parent's helper
        num_obstacles = len(self.manager.obstacles)
        colors = ['red', 'blue', 'orange', 'purple', 'gold', 'cyan', 'magenta']
        
        obstacle_patches = []
        obstacle_trails = []
        obstacle_trail_data = []
        cpa_lines = []  # Collision-specific: show CPA points
        
        # Collision points visualization
        collision_markers = []
        for cp in self.collision_points:
            marker = ax_main.plot(cp['x'], cp['y'], 'rX', markersize=20, 
                                 markeredgewidth=3, label='Collision!' if not collision_markers else '',
                                 zorder=15)[0]
            collision_markers.append(marker)
            # Add collision circle
            coll_circle = Circle((cp['x'], cp['y']), 3.0, facecolor='red',
                               edgecolor='darkred', linewidth=2, alpha=0.3, zorder=14)
            ax_main.add_patch(coll_circle)
        
        for i in range(num_obstacles):
            color = colors[i % len(colors)]
            
            # Reuse parent's vessel creation helper
            body, bow = self._create_vessel_patches(ax_main, color, scale=0.9, zorder=10)
            obstacle_patches.append((body, bow))
            
            # Trail
            trail, = ax_main.plot([], [], '-', color=color, linewidth=1.5,
                                alpha=0.5, zorder=3)
            obstacle_trails.append(trail)
            obstacle_trail_data.append({'x': [], 'y': []})
            
            # CPA line (collision detection specific)
            cpa_line, = ax_main.plot([], [], 'r--', linewidth=2, alpha=0.7, zorder=9)
            cpa_lines.append(cpa_line)
        
        # Axis setup
        ax_main.set_xlabel('X (cells)', fontsize=12)
        ax_main.set_ylabel('Y (cells)', fontsize=12)
        ax_main.set_title(f'Navigation with Collision Detection - {self.controller_name}',
                         fontsize=14, fontweight='bold')
        ax_main.legend(loc='upper left', fontsize=9)
        ax_main.set_aspect('equal')
        ax_main.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.2)
        
        # Status text (reuse parent's format)
        status_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                                  fontsize=10, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                  zorder=14, family='monospace')
        
        # Collision info panel (unique to this class)
        collision_text = ax_info.text(0.05, 0.95, '', verticalalignment='top',
                                     fontsize=9, family='monospace',
                                     bbox=dict(boxstyle='round', facecolor='lightyellow',
                                             edgecolor='orange', linewidth=2))
        
        plt.tight_layout()
        
        def init():
            """Initialize animation."""
            artists = [our_body, our_bow, our_trail, target_circle, target_line]
            for body, bow in obstacle_patches:
                artists.extend([body, bow])
            artists.extend(obstacle_trails + cpa_lines + [status_text, collision_text])
            return artists
        
        def update(frame):
            """Update animation frame."""
            if frame >= len(self.simulation_data):
                return init()
            
            state = self.simulation_data[frame]
            collision_infos = self.collision_data[frame]
            
            # Update our vessel (reuse parent's transform pattern)
            our_state = state['our_vessel']
            our_x, our_y = our_state['x'], our_state['y']
            our_heading = our_state['heading']
            
            transform = (Affine2D().rotate(our_heading).translate(our_x, our_y) + ax_main.transData)
            our_body.set_transform(transform)
            our_bow.set_transform(transform)
            
            # Update trail (reuse parent's pattern)
            our_trail_data['x'].append(our_x)
            our_trail_data['y'].append(our_y)
            our_trail.set_data(our_trail_data['x'], our_trail_data['y'])
            
            # Update target
            if our_state['target']:
                tx, ty = our_state['target']
                target_circle.set_center((tx, ty))
                target_line.set_data([our_x, tx], [our_y, ty])
            
            # Update obstacles (reuse parent's transform pattern)
            for i, obs_state in enumerate(state['dynamic_obstacles']):
                obs_x, obs_y = obs_state['x'], obs_state['y']
                obs_heading = obs_state['heading']
                
                obs_transform = (Affine2D().rotate(obs_heading).translate(obs_x, obs_y) + ax_main.transData)
                body, bow = obstacle_patches[i]
                body.set_transform(obs_transform)
                bow.set_transform(obs_transform)
                
                # Update trail
                obstacle_trail_data[i]['x'].append(obs_x)
                obstacle_trail_data[i]['y'].append(obs_y)
                obstacle_trails[i].set_data(obstacle_trail_data[i]['x'],
                                          obstacle_trail_data[i]['y'])
            
            # Update collision-specific visualizations
            info_lines = ['COLLISION DETECTION\n' + '='*30 + '\n\n']
            highest_risk = 0
            
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
                info_lines.append(f'  Bearing: {np.degrees(info.relative_bearing):.0f}°\n')
                info_lines.append('\n')
                
                # Draw CPA lines for high-risk encounters
                if risk_level >= 2 and info.tcpa > 0:
                    # Calculate CPA positions
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
            
            collision_text.set_text(''.join(info_lines))
            
            # Color panel by risk level
            risk_bg_colors = ['lightgreen', 'lightyellow', 'orange', 'red']
            collision_text.get_bbox_patch().set_facecolor(risk_bg_colors[highest_risk])
            
            # Status text (reuse parent's format)
            heading_360 = (90 - np.degrees(our_heading)) % 360
            status_text.set_text(
                f'Time: {state["time"]:.1f}s\n'
                f'Pos: ({our_x:.1f}, {our_y:.1f})\n'
                f'Heading: {heading_360:.0f}°\n'
                f'Speed: {our_state["speed"]:.2f} units/s'
            )
            
            return init()
        
        # Speed up animation like parent
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
    print("NAVIGATION WITH COLLISION DETECTION")
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
    vessel1 = manager.add_obstacle(x=5, y=50, heading=0, speed=0.4, behavior='straight')
    print(f"  Added {vessel1.obstacle}")
    
    # Obstacle 1: Moving north (collision course!)
    vessel2 = manager.add_obstacle(x=40, y=10, heading=np.pi/2, speed=0.3, behavior='straight')
    print(f"  Added {vessel2.obstacle}")
    
    # Obstacle 2: Circular patrol
    vessel3 = manager.add_obstacle(x=50, y=50, heading=0, speed=0.2, behavior='circular')
    vessel3.set_circular_path(center=(50, 50), radius=7.0)
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
    print("SIMULATING WITH COLLISION DETECTION...")
    print("=" * 70)
    
    sim = NavigationWithCollisionDetection(grid, our_vessel, controller,
                                          manager, controller_name)
    sim.simulate(waypoints, duration=120.0, dt=0.1)
    
    # Animate
    print("\n" + "=" * 70)
    print("ANIMATION STARTING...")
    print("=" * 70)
    print("\nAnimation features:")
    print("  📊 Right panel: Real-time collision info")
    print("  - CPA distance and time for each vessel")
    print("  - Risk level (SAFE/LOW/MEDIUM/HIGH)")
    print("  - Encounter type for COLREGs")
    print("  - Relative bearing")
    print("\n  🔴 Red dashed lines: CPA points (where closest approach occurs)")
    print("  📈 Panel color changes with risk level")
    print("\nWatch the collision info panel!")
    print("=" * 70)
    
    sim.animate(waypoints, interval=5)
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("\n1. CPA/TCPA calculated in real-time for all vessels")
    print("2. Risk levels change as vessels approach")
    print("3. Encounter types identified (head-on, crossing, etc.)")
    print("4. Red lines show where CPA will occur")
    print("5. Our vessel STILL doesn't avoid - just monitors!")
    print("\nNext step: Implement collision avoidance maneuvers!")
    print("=" * 70)


if __name__ == "__main__":
    main()