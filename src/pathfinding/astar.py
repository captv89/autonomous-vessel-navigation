"""
A* Pathfinding Algorithm

Finds the shortest path between two points while avoiding obstacles.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import heapq
from dataclasses import dataclass, field


def simplify_path(waypoints: List[Tuple[float, float]], 
                  min_distance: float = 5.0,
                  lookahead_buffer: float = 1.2) -> List[Tuple[float, float]]:
    """
    Simplify path by removing intermediate waypoints that are too close together.
    This removes grid artifacts and creates a smoother path for controllers.
    
    Args:
        waypoints: Original dense waypoint list from A*
        min_distance: Minimum distance between kept waypoints (default=5.0)
        lookahead_buffer: Reserved for future use (default=1.2)
        
    Returns:
        Simplified waypoint list with only key turning points
        
    Example:
        # A* returns 85 waypoints for a 105-unit path (too dense)
        simplified = simplify_path(waypoints, min_distance=8.0)
        # Returns ~13 waypoints - much better for path following controllers
    """
    if len(waypoints) <= 2:
        return waypoints
    
    # Always keep start
    simplified = [waypoints[0]]
    
    for i in range(1, len(waypoints) - 1):
        prev_wp = simplified[-1]
        curr_wp = waypoints[i]
        goal = waypoints[-1]
        
        dist_to_prev = np.hypot(curr_wp[0] - prev_wp[0], curr_wp[1] - prev_wp[1])
        dist_to_goal = np.hypot(goal[0] - curr_wp[0], goal[1] - curr_wp[1])
        
        # Only keep the waypoint if it is sufficiently far from the previous
        # AND sufficiently far from the goal (to avoid clustering at the end)
        if dist_to_prev >= min_distance and dist_to_goal >= min_distance:
            simplified.append(curr_wp)
            
    simplified.append(waypoints[-1])  # Always keep goal
    return simplified


@dataclass(order=True)
class Node:
    """
    Represents a node in the A* search.
    
    Attributes:
        f_cost: Total cost (g + h)
        position: (x, y) coordinates (not used in comparison)
        g_cost: Cost from start to this node
        h_cost: Heuristic cost from this node to goal
        parent: Parent node for path reconstruction
    """
    f_cost: float
    position: Tuple[int, int] = field(compare=False)
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)


class AStar:
    """
    A* pathfinding algorithm implementation.
    """
    
    def __init__(self, grid_world, allow_diagonal: bool = True, safety_margin_cells: int = 2):
        """
        Initialize A* pathfinder.
        
        Args:
            grid_world: GridWorld object containing the environment
            allow_diagonal: Whether to allow diagonal movement
            safety_margin_cells: Safety buffer around obstacles in cells (default=2).
                               Set to 0 to disable safety buffer.
                               Example: For 10m cells and 20m vessel, use 2 cells buffer.
        """
        self.grid_world = grid_world
        self.allow_diagonal = allow_diagonal
        self.safety_margin_cells = safety_margin_cells
        
        # Get planning grid with safety buffer
        # This inflated grid is used for pathfinding to avoid obstacles
        # The original grid is still used for actual collision detection
        if safety_margin_cells > 0:
            print(f"A* using safety margin: {safety_margin_cells} cells ({safety_margin_cells * grid_world.cell_size:.1f}m)")
            self.planning_grid = grid_world.get_planning_grid(safety_margin_cells)
        else:
            print("A* using no safety margin (not recommended)")
            self.planning_grid = grid_world.grid
        
        # Costs for movement
        self.straight_cost = 1.0
        self.diagonal_cost = np.sqrt(2)  # ~1.414
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate heuristic cost (estimated distance) between two positions.
        Using Euclidean distance for accuracy.
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Estimated distance
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Euclidean distance
        return np.sqrt(dx*dx + dy*dy)
    
    def get_movement_cost(self, current: Tuple[int, int], neighbor: Tuple[int, int]) -> float:
        """
        Calculate the cost of moving from current to neighbor.
        
        Args:
            current: Current position
            neighbor: Neighbor position
            
        Returns:
            Movement cost
        """
        dx = abs(neighbor[0] - current[0])
        dy = abs(neighbor[1] - current[1])
        
        # Diagonal movement
        if dx == 1 and dy == 1:
            return self.diagonal_cost
        
        # Straight movement
        return self.straight_cost
    
    def get_valid_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get valid navigable neighbors using the planning grid (with safety buffer).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            List of valid neighbor positions (x, y)
        """
        neighbors = []
        
        # Define movement directions
        if self.allow_diagonal:
            # 8-directional movement
            directions = [
                (-1, -1), (0, -1), (1, -1),
                (-1,  0),          (1,  0),
                (-1,  1), (0,  1), (1,  1)
            ]
        else:
            # 4-directional movement
            directions = [
                          (0, -1),
                (-1,  0),          (1,  0),
                          (0,  1)
            ]
        
        # Check each direction
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if nx < 0 or nx >= self.grid_world.width or ny < 0 or ny >= self.grid_world.height:
                continue
            
            # Check if navigable in planning grid (respects safety buffer)
            if self.planning_grid[ny, nx] > 0.5:
                continue
            
            neighbors.append((nx, ny))
        
        return neighbors
    
    def reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to goal by following parent nodes.
        
        Args:
            node: The goal node
            
        Returns:
            List of (x, y) positions from start to goal
        """
        path = []
        current = node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        # Reverse to get path from start to goal
        path.reverse()
        return path
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  verbose: bool = False, 
                  step_callback: Optional[callable] = None) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to goal using A*.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            verbose: Print search progress
            step_callback: Optional callback function called at each step with
                         (open_set_positions, closed_set, current_position)
            
        Returns:
            List of (x, y) positions representing the path, or None if no path exists
        """
        # Validate start and goal against planning grid (with safety buffer)
        if (start[0] < 0 or start[0] >= self.grid_world.width or
            start[1] < 0 or start[1] >= self.grid_world.height or
            self.planning_grid[start[1], start[0]] > 0.5):
            print(f"❌ Start position {start} is not valid or too close to obstacles!")
            return None
        
        if (goal[0] < 0 or goal[0] >= self.grid_world.width or
            goal[1] < 0 or goal[1] >= self.grid_world.height or
            self.planning_grid[goal[1], goal[0]] > 0.5):
            print(f"❌ Goal position {goal} is not valid or too close to obstacles!")
            return None
        
        if verbose:
            print(f"Finding path from {start} to {goal}...")
        
        # Initialize
        start_node = Node(
            f_cost=0,
            position=start,
            g_cost=0,
            h_cost=self.heuristic(start, goal),
            parent=None
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        # Priority queue (min-heap) for open set
        open_set = []
        heapq.heappush(open_set, start_node)
        
        # Track visited nodes and their g_costs
        closed_set = set()
        g_costs: Dict[Tuple[int, int], float] = {start: 0}
        
        # Track open set positions for callbacks
        open_positions = {start}
        
        nodes_explored = 0
        
        while open_set:
            # Call callback before processing (for animation)
            if step_callback:
                step_callback(open_positions.copy(), closed_set.copy(), 
                            open_set[0].position if open_set else None)
            
            # Get node with lowest f_cost
            current_node = heapq.heappop(open_set)
            current_pos = current_node.position
            open_positions.discard(current_pos)
            
            # Skip if we've already processed this position with a better cost
            if current_pos in closed_set:
                continue
            
            nodes_explored += 1
            
            # Check if we reached the goal
            if current_pos == goal:
                path = self.reconstruct_path(current_node)
                if verbose:
                    print(f"✓ Path found! Length: {len(path)} cells")
                    print(f"  Nodes explored: {nodes_explored}")
                    print(f"  Path cost: {current_node.g_cost:.2f}")
                return path
            
            # Mark as visited
            closed_set.add(current_pos)
            
            # Explore neighbors (using planning grid with safety buffer)
            neighbors = self.get_valid_neighbors(current_pos[0], current_pos[1])
            
            for neighbor_pos in neighbors:
                # Skip if already processed
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate costs
                movement_cost = self.get_movement_cost(current_pos, neighbor_pos)
                tentative_g_cost = current_node.g_cost + movement_cost
                
                # Skip if we've found a better path to this neighbor
                if neighbor_pos in g_costs and tentative_g_cost >= g_costs[neighbor_pos]:
                    continue
                
                # This is the best path to this neighbor so far
                g_costs[neighbor_pos] = tentative_g_cost
                h_cost = self.heuristic(neighbor_pos, goal)
                f_cost = tentative_g_cost + h_cost
                
                neighbor_node = Node(
                    f_cost=f_cost,
                    position=neighbor_pos,
                    g_cost=tentative_g_cost,
                    h_cost=h_cost,
                    parent=current_node
                )
                
                heapq.heappush(open_set, neighbor_node)
                open_positions.add(neighbor_pos)
        
        # No path found
        if verbose:
            print(f"❌ No path found from {start} to {goal}")
            print(f"  Nodes explored: {nodes_explored}")
        
        return None


def create_test_scenario():
    """
    Create a test scenario with grid world and waypoints.
    
    Returns:
        Tuple of (grid_world, start, goal)
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.environment.grid_world import GridWorld
    
    # Create a moderately sized grid for clear visualization
    grid = GridWorld(width=100, height=100, cell_size=10.0)
    
    print("\nCreating scenario with obstacles...")
    
    # Large island on the left
    grid.add_circular_obstacle(center_x=20, center_y=50, radius=12)
    
    # Small islands
    grid.add_circular_obstacle(center_x=45, center_y=25, radius=5)
    grid.add_circular_obstacle(center_x=60, center_y=70, radius=6)
    grid.add_circular_obstacle(center_x=75, center_y=40, radius=4)
    
    # Shallow water zones
    grid.add_obstacle(x=35, y=55, width=15, height=8)
    grid.add_obstacle(x=50, y=10, width=10, height=12)
    
    # Restricted zone
    grid.add_obstacle(x=10, y=10, width=12, height=12)
    
    # Reef area
    grid.add_circular_obstacle(center_x=85, center_y=20, radius=7)
    
    # Define start and goal
    start = (10, 90)
    goal = (60, 6)
    
    return grid, start, goal


def demo_astar(animate: bool = False):
    """
    Demonstrate A* pathfinding.
    
    Args:
        animate: If True, show animation. If False, show static plot.
    """
    print("=" * 70)
    print("A* PATHFINDING DEMO")
    print("=" * 70)
    
    # Create test scenario
    grid, start, goal = create_test_scenario()
    
    print(f"\nStart: {start}")
    print(f"Goal:  {goal}")
    
    if animate:
        # Use animator for animation
        from src.visualization.astar_animator import AStarAnimator
        
        print("\n" + "=" * 70)
        print("Running A* with animation...")
        print("=" * 70)
        
        animator = AStarAnimator(grid, allow_diagonal=True)
        path = animator.find_path_animated(start, goal)
        
        if path:
            print("\nStarting animation...")
            print("Watch how A* explores the grid:")
            print("  🟢 Light green = Open set (nodes to explore)")
            print("  🔴 Red/salmon  = Closed set (already explored)")
            print("  🟡 Yellow dot  = Current node being processed")
            print("  🟢 Green       = Start position")
            print("  🔴 Red star    = Goal position")
            print("  🟡 Yellow line = Final path (shown at the end)")
            print("\nClose the window when done watching.")
            animator.animate(start, goal, interval=30)
    else:
        # Use AStar for static plot
        from src.visualization.plotter import Plotter
        
        print("\n" + "=" * 70)
        print("Running A* with static visualization...")
        print("=" * 70)
        
        astar = AStar(grid, allow_diagonal=True)
        path = astar.find_path(start, goal, verbose=True)
        
        print("\nDisplaying result...")
        plotter = Plotter(figsize=(14, 14))
        title = "A* Pathfinding Result"
        if path:
            title += f" - Path Found ({len(path)} waypoints)"
        else:
            title += " - No Path Found"
        plotter.plot_grid(grid, title=title, start=start, goal=goal, path=path, show=True)
    
    print("\n✓ Demo complete!")


def test_astar():
    """Test the A* algorithm."""
    print("Testing A* Algorithm")
    print("=" * 60)
    
    # Use the shared test scenario
    grid, start, goal = create_test_scenario()
    
    # Create A* pathfinder
    astar = AStar(grid, allow_diagonal=True)
    
    print(f"\nFinding path from {start} to {goal}...")
    print("-" * 60)
    
    path = astar.find_path(start, goal, verbose=True)
    
    if path:
        print(f"\n✓ Success! Path has {len(path)} waypoints")
        print(f"  First few waypoints: {path[:5]}")
        print(f"  Last few waypoints: {path[-5:]}")
    else:
        print("\n❌ Failed to find path")
    
    print("=" * 60)
    
    # Visualize the result
    from src.visualization.plotter import Plotter
    
    print("\nDisplaying visualization...")
    plotter = Plotter(figsize=(14, 14))
    title = "A* Test Result"
    if path:
        title += f" - Path Found ({len(path)} waypoints)"
    else:
        title += " - No Path Found"
    plotter.plot_grid(grid, title=title, start=start, goal=goal, path=path, show=True)


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--animate':
        demo_astar(animate=True)
    elif len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_astar()
    else:
        # Default: show static plot
        demo_astar(animate=False)