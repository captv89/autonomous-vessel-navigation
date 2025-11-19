"""
A* Animation - Visualize the A* search process step by step.

Shows:
- Open set (nodes to be explored) in green
- Closed set (already explored) in red
- Current node being processed in yellow
- Final path in bright yellow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import sys
import os

# Import the AStar class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.pathfinding.astar import AStar


class AStarAnimator:
    """
    Animator for A* pathfinding using the AStar class.
    Records each step of the search for animation.
    """
    
    def __init__(self, grid_world, allow_diagonal: bool = True):
        """
        Initialize animator.
        
        Args:
            grid_world: GridWorld object
            allow_diagonal: Whether to allow diagonal movement
        """
        self.grid_world = grid_world
        self.astar = AStar(grid_world, allow_diagonal=allow_diagonal)
        
        # Animation data
        self.frames = []  # Store each step for animation
        self.final_path = None
    
    def _record_frame(self, open_positions: set, closed_set: set, current_pos: Optional[Tuple[int, int]]):
        """
        Callback function to record each search step.
        
        Args:
            open_positions: Set of positions in open set
            closed_set: Set of positions already explored
            current_pos: Current position being processed
        """
        self.frames.append({
            'open_set': open_positions,
            'closed_set': closed_set,
            'current': current_pos
        })
    
    def find_path_animated(self, start: Tuple[int, int], goal: Tuple[int, int],
                          max_frames: int = 1000) -> Optional[List[Tuple[int, int]]]:
        """
        Find path and record each step for animation.
        
        Args:
            start: Start position
            goal: Goal position
            max_frames: Maximum number of frames to record (prevents infinite loops)
            
        Returns:
            Final path or None
        """
        print(f"Running A* search from {start} to {goal}...")
        print("Recording animation frames...")
        
        # Clear previous frames
        self.frames = []
        self.final_path = None
        
        # Use AStar to find path with callback to record frames
        self.final_path = self.astar.find_path(start, goal, verbose=True, 
                                              step_callback=self._record_frame)
        
        if self.final_path:
            print(f"✓ Path found!")
            print(f"  Frames recorded: {len(self.frames)}")
            print(f"  Path length: {len(self.final_path)} waypoints")
        
        return self.final_path
    
    def animate(self, start: Tuple[int, int], goal: Tuple[int, int],
                interval: int = 50, save_file: Optional[str] = None):
        """
        Create and display animation of A* search.
        
        Args:
            start: Start position
            goal: Goal position
            interval: Milliseconds between frames
            save_file: Optional filename to save animation (e.g., 'astar.gif')
        """
        if len(self.frames) == 0:
            print("No frames to animate. Run find_path_animated first!")
            return
        
        print(f"\nCreating animation with {len(self.frames)} frames...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Display grid (obstacles)
        im = ax.imshow(self.grid_world.grid, cmap='Blues', origin='lower',
                      vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
        
        # Initialize empty collections for open/closed sets
        open_scatter = ax.scatter([], [], c='lightgreen', s=30, alpha=0.6, label='Open Set')
        closed_scatter = ax.scatter([], [], c='salmon', s=20, alpha=0.6, label='Closed Set')
        current_scatter = ax.scatter([], [], c='yellow', s=100, marker='o', 
                                    edgecolors='orange', linewidths=2, label='Current')
        
        # Plot start and goal
        ax.plot(start[0], start[1], 'go', markersize=15, label='Start',
               markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal',
               markeredgecolor='darkred', markeredgewidth=2)
        
        # Path line (will be drawn at the end)
        path_line, = ax.plot([], [], 'y-', linewidth=3, alpha=0.8, label='Path')
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.grid_world.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_world.height, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Labels and title
        ax.set_xlabel('X (cells)', fontsize=12)
        ax.set_ylabel('Y (cells)', fontsize=12)
        title = ax.set_title('A* Search Animation - Frame 0', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal')
        
        # Frame counter text
        frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def init():
            """Initialize animation."""
            open_scatter.set_offsets(np.empty((0, 2)))
            closed_scatter.set_offsets(np.empty((0, 2)))
            current_scatter.set_offsets(np.empty((0, 2)))
            path_line.set_data([], [])
            return open_scatter, closed_scatter, current_scatter, path_line, title, frame_text
        
        def update(frame_num):
            """Update animation for each frame."""
            # Check if we've finished searching
            if frame_num >= len(self.frames):
                # Show final path
                if self.final_path:
                    path_x = [p[0] for p in self.final_path]
                    path_y = [p[1] for p in self.final_path]
                    path_line.set_data(path_x, path_y)
                    title.set_text(f'A* Search Complete - Path Found! ({len(self.final_path)} waypoints)')
                    frame_text.set_text(f'Search Complete\nPath Length: {len(self.final_path)}')
                else:
                    title.set_text('A* Search Complete - No Path Found')
                    frame_text.set_text('Search Complete\nNo Path Found')
                return open_scatter, closed_scatter, current_scatter, path_line, title, frame_text
            
            frame = self.frames[frame_num]
            
            # Update open set
            if frame['open_set']:
                open_positions = np.array(list(frame['open_set']))
                open_scatter.set_offsets(open_positions)
            else:
                open_scatter.set_offsets(np.empty((0, 2)))
            
            # Update closed set
            if frame['closed_set']:
                closed_positions = np.array(list(frame['closed_set']))
                closed_scatter.set_offsets(closed_positions)
            else:
                closed_scatter.set_offsets(np.empty((0, 2)))
            
            # Update current node
            if frame['current']:
                current_scatter.set_offsets([frame['current']])
            else:
                current_scatter.set_offsets(np.empty((0, 2)))
            
            # Update title and info
            title.set_text('A* Search Animation')
            frame_text.set_text(f'Frame: {frame_num + 1}/{len(self.frames)}\n'
                              f'Open: {len(frame["open_set"])}\n'
                              f'Closed: {len(frame["closed_set"])}')
            
            return open_scatter, closed_scatter, current_scatter, path_line, title, frame_text
        
        # Add extra frames at the end to show the final result
        total_frames = len(self.frames) + 30  # Hold final result for 30 frames
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=total_frames, interval=interval,
                           blit=True, repeat=False)
        
        if save_file:
            print(f"Saving animation to {save_file}...")
            anim.save(save_file, writer='pillow', fps=20)
            print(f"✓ Animation saved!")
        
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Animation complete!")


def demo_animation():
    """Demonstrate the A* animation."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.environment.grid_world import GridWorld
    
    print("=" * 70)
    print("A* SEARCH ANIMATION DEMO")
    print("=" * 70)
    
    # Create a moderately sized grid for clear visualization
    grid = GridWorld(width=40, height=40, cell_size=10.0)
    
    print("\nCreating scenario with obstacles...")
    # Add some interesting obstacles
    grid.add_obstacle(15, 10, width=3, height=15)
    grid.add_circular_obstacle(25, 25, radius=4)
    grid.add_obstacle(8, 25, width=10, height=3)
    grid.add_circular_obstacle(30, 10, radius=3)
    
    # Define start and goal
    start = (5, 5)
    goal = (35, 35)
    
    print(f"\nStart: {start}")
    print(f"Goal:  {goal}")
    
    # Create animator and find path
    animator = AStarAnimator(grid, allow_diagonal=True)
    path = animator.find_path_animated(start, goal)
    
    if path:
        print("\n" + "=" * 70)
        print("Starting animation...")
        print("=" * 70)
        print("\nWatch how A* explores the grid:")
        print("  🟢 Light green = Open set (nodes to explore)")
        print("  🔴 Red/salmon  = Closed set (already explored)")
        print("  🟡 Yellow dot  = Current node being processed")
        print("  🟢 Green       = Start position")
        print("  🔴 Red star    = Goal position")
        print("  🟡 Yellow line = Final path (shown at the end)")
        print("\nClose the window when done watching.")
        print("=" * 70)
        
        # Animate with 50ms between frames (adjust for faster/slower)
            animator.animate(start, goal, interval=30)    else:
        print("\n❌ Could not find path to animate.")


if __name__ == "__main__":
    demo_animation()