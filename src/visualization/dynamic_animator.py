"""
Dynamic Obstacles Animation

Animates dynamic obstacles (moving vessels) through the environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge, Circle, FancyArrow, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from typing import List, Dict


class DynamicObstaclesAnimator:
    """Animates dynamic obstacles moving through the environment."""
    
    def __init__(self, grid_world, obstacle_manager, figsize=(14, 12)):
        """
        Initialize animator.
        
        Args:
            grid_world: GridWorld object
            obstacle_manager: DynamicObstacleManager object
            figsize: Figure size
        """
        self.grid_world = grid_world
        self.manager = obstacle_manager
        self.figsize = figsize
        self.simulation_data = []
    
    def simulate(self, duration: float, dt: float = 0.1):
        """
        Simulate the dynamic obstacles for a duration.
        
        Args:
            duration: Simulation duration in seconds
            dt: Time step
        """
        print(f"Simulating {len(self.manager.obstacles)} vessels for {duration}s...")
        
        self.simulation_data = []
        steps = int(duration / dt)
        
        for step in range(steps):
            # Record current state
            state = {
                'time': step * dt,
                'vessels': []
            }
            
            for vessel in self.manager.obstacles:
                vessel_state = {
                    'id': vessel.obstacle.id,
                    'x': vessel.obstacle.x,
                    'y': vessel.obstacle.y,
                    'heading': vessel.obstacle.heading,
                    'speed': vessel.obstacle.speed,
                    'behavior': vessel.behavior
                }
                state['vessels'].append(vessel_state)
            
            self.simulation_data.append(state)
            
            # Update all vessels
            self.manager.update_all(dt)
        
        print(f"  ✓ Simulation complete: {len(self.simulation_data)} frames")
    
    def animate(self, interval: int = 20, show_trails: bool = True,
                show_predictions: bool = True, repeat: bool = True):
        """
        Animate the simulation with realistic vessel shapes.
        
        Args:
            interval: Milliseconds between frames
            show_trails: Show trajectory trails
            show_predictions: Show predicted future positions
            repeat: Whether to loop the animation
        """
        if not self.simulation_data:
            print("❌ No simulation data! Run simulate() first.")
            return
        
        print("\nCreating animation...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot grid
        ax.imshow(self.grid_world.grid, cmap='Blues', origin='lower',
                 vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
        
        # Grid setup
        ax.set_xlabel('X (cells)', fontsize=12)
        ax.set_ylabel('Y (cells)', fontsize=12)
        ax.set_title('Dynamic Obstacles - Moving Vessels', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.2)
        
        # Color map for different vessels (more distinct colors)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'gold', 'cyan', 'magenta']
        
        # Initialize vessel patches and trails
        vessel_bodies = []
        vessel_bows = []
        trail_lines = []
        prediction_circles = []
        trails_data = []
        
        for i, vessel in enumerate(self.manager.obstacles):
            color = colors[i % len(colors)]
            
            # Vessel shape: Rectangle body + Triangle bow (20% smaller than main vessels)
            # Body: 4.86x2.025 (90% of 5.4x2.25)
            vessel_body = Rectangle((-2.43, -1.0125), 4.86, 2.025, 
                                   facecolor=color, edgecolor='darkgray',
                                   linewidth=1.5, alpha=0.8, zorder=10)
            ax.add_patch(vessel_body)
            vessel_bodies.append(vessel_body)
            
            # Bow: triangle at front, 1.62x2.025 (90% of 1.8x2.25)
            vessel_bow = Polygon([(2.43, -1.0125), (2.43, 1.0125), (4.05, 0)],
                                facecolor=color, edgecolor='darkgray',
                                linewidth=1.5, alpha=0.8, zorder=10)
            ax.add_patch(vessel_bow)
            vessel_bows.append(vessel_bow)
            
            # Trail
            trail_line, = ax.plot([], [], '-', color=color, linewidth=1.5,
                                 alpha=0.4, zorder=3)
            trail_lines.append(trail_line)
            trails_data.append({'x': [], 'y': []})
            
            # Prediction circle
            pred_circle = Circle((0, 0), 1.0, facecolor='none',
                                edgecolor=color, linewidth=2,
                                linestyle='--', alpha=0.6, zorder=8)
            ax.add_patch(pred_circle)
            prediction_circles.append(pred_circle)
        
        # Info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                          zorder=12, family='monospace')
        
        # Legend
        legend_elements = []
        for i, vessel in enumerate(self.manager.obstacles):
            color = colors[i % len(colors)]
            behavior = vessel.behavior.capitalize()
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, markersize=10,
                      label=f'Vessel {i+1} ({behavior})')
            )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        def init():
            """Initialize animation."""
            return vessel_bodies + vessel_bows + trail_lines + prediction_circles + [info_text]
        
        def update(frame):
            """Update animation frame."""
            if frame >= len(self.simulation_data):
                return vessel_bodies + vessel_bows + trail_lines + prediction_circles + [info_text]
            
            state = self.simulation_data[frame]
            
            info_lines = [f'Time: {state["time"]:.1f}s\n']
            
            for i, vessel_state in enumerate(state['vessels']):
                x = vessel_state['x']
                y = vessel_state['y']
                heading = vessel_state['heading']
                
                # Create rotation transform for vessel
                transform = (Affine2D().rotate(heading).translate(x, y) + ax.transData)
                
                # Update vessel body and bow
                vessel_bodies[i].set_transform(transform)
                vessel_bows[i].set_transform(transform)
                
                # Update trail
                if show_trails:
                    trails_data[i]['x'].append(x)
                    trails_data[i]['y'].append(y)
                    trail_lines[i].set_data(trails_data[i]['x'], trails_data[i]['y'])
                
                # Update prediction (5 seconds ahead)
                if show_predictions:
                    vessel_obj = self.manager.obstacles[i]
                    pred_x, pred_y = vessel_obj.predict_position(5.0)
                    prediction_circles[i].set_center((pred_x, pred_y))
                else:
                    prediction_circles[i].set_radius(0)  # Hide
                
                # Convert heading to navigation notation
                heading_360 = (90 - np.degrees(heading)) % 360
                
                # Update info text
                info_lines.append(
                    f'V{i+1}: ({x:5.1f},{y:5.1f}) '
                    f'{heading_360:5.0f}° '
                    f'{vessel_state["speed"]:.1f}u/s\n'
                )
            
            info_text.set_text(''.join(info_lines))
            
            return vessel_bodies + vessel_bows + trail_lines + prediction_circles + [info_text]
        
        # Sample frames for performance
        frame_skip = max(1, len(self.simulation_data) // 500)
        frames = range(0, len(self.simulation_data), frame_skip)
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=frames, interval=interval,
                           blit=False, repeat=repeat)
        
        plt.show()
        print("\n✓ Animation complete!")
