"""
A* Pathfinding Algorithm

Finds the shortest path between two points while avoiding obstacles.
Includes advanced post-processing for smooth vessel trajectories.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import heapq
from dataclasses import dataclass, field


def bresenham_line(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Bresenham's line algorithm - returns all grid cells intersected by a line.
    
    Args:
        start: Starting grid position (x, y)
        end: Ending grid position (x, y)
        
    Returns:
        List of all grid cells the line passes through
    """
    x0, y0 = start
    x1, y1 = end
    
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    err = dx - dy
    
    while True:
        cells.append((x, y))
        
        if x == x1 and y == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
            
    return cells


def has_line_of_sight(start: Tuple[int, int], end: Tuple[int, int], 
                      grid: np.ndarray, width: int, height: int) -> bool:
    """
    Check if there's a clear line of sight between two points.
    Uses Bresenham's algorithm to check all cells along the line.
    
    Args:
        start: Starting position (x, y)
        end: Ending position (x, y)
        grid: Obstacle grid (values > 0.5 are obstacles)
        width: Grid width
        height: Grid height
        
    Returns:
        True if line is clear, False if it intersects obstacles
    """
    cells = bresenham_line(start, end)
    
    for x, y in cells:
        # Check bounds
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        # Check collision
        if grid[y, x] > 0.5:
            return False
            
    return True


def string_pulling(path: List[Tuple[int, int]], grid: np.ndarray, 
                   width: int, height: int) -> List[Tuple[float, float]]:
    """
    String Pulling algorithm - optimizes path by removing unnecessary waypoints.
    
    'Pulls the string tight' to create straight-line segments wherever possible.
    If point A can see point C directly (no obstacles), point B is removed.
    
    This dramatically reduces waypoint count while maintaining safety.
    
    Args:
        path: Original waypoint list from A*
        grid: Planning grid (with safety buffer applied)
        width: Grid width
        height: Grid height
        
    Returns:
        Optimized path with minimal waypoints
        
    Example:
        Original: A -> B -> C -> D -> E (5 waypoints)
        If A can see E directly: A -> E (2 waypoints)
    """
    if len(path) < 3:
        return [(float(x), float(y)) for x, y in path]
    
    optimized = [path[0]]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # Greedily find the furthest visible waypoint
        next_idx = current_idx + 1
        
        for i in range(len(path) - 1, current_idx + 1, -1):
            if has_line_of_sight(path[current_idx], path[i], grid, width, height):
                next_idx = i
                break
        
        optimized.append(path[next_idx])
        current_idx = next_idx
    
    # Convert to float tuples
    return [(float(x), float(y)) for x, y in optimized]


def inject_intermediate_waypoints(waypoints: List[Tuple[float, float]], 
                                  max_segment_length: float = 15.0) -> List[Tuple[float, float]]:
    """
    Inject intermediate waypoints on long straight segments.
    
    While straight lines are optimal for distance, very long segments can cause
    controllers to accumulate cross-track error. Intermediate waypoints help
    maintain tighter path following.
    
    Args:
        waypoints: Optimized waypoint list
        max_segment_length: Maximum allowed distance between waypoints
        
    Returns:
        Waypoint list with injected intermediate points
    """
    if len(waypoints) < 2:
        return waypoints
    
    result = [waypoints[0]]
    
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i])
        end = np.array(waypoints[i + 1])
        
        segment_length = np.linalg.norm(end - start)
        
        if segment_length > max_segment_length:
            # Calculate number of subdivisions needed
            num_segments = int(np.ceil(segment_length / max_segment_length))
            
            # Inject intermediate points
            for j in range(1, num_segments):
                fraction = j / num_segments
                intermediate = start + (end - start) * fraction
                result.append(tuple(intermediate))
        
        result.append(waypoints[i + 1])
    
    return result


def simplify_path(waypoints: List[Tuple[float, float]], 
                  grid: Optional[np.ndarray] = None,
                  grid_width: Optional[int] = None,
                  grid_height: Optional[int] = None,
                  use_string_pulling: bool = True,
                  max_segment_length: float = 15.0) -> List[Tuple[float, float]]:
    """
    Advanced path simplification using string pulling and optional waypoint injection.
    
    PRIMARY METHOD: String Pulling (Line-of-Sight Optimization)
    - Creates straight-line segments wherever possible
    - Guarantees no obstacle collisions
    - Optimal for vessel navigation
    
    Args:
        waypoints: Original dense waypoint list from A*
        grid: Planning grid (with safety buffer) for line-of-sight checks
        grid_width: Grid width in cells
        grid_height: Grid height in cells
        use_string_pulling: Use string pulling if grid info available (default=True)
        max_segment_length: Max length before injecting intermediate waypoints (default=15.0)
        
    Returns:
        Optimized waypoint list with minimal turning points
        
    Example:
        # With string pulling (optimal):
        path = astar.find_path(start, goal)
        simplified = simplify_path(path, grid=astar.planning_grid, 
                                  grid_width=100, grid_height=100)
        # 85 waypoints -> 4-6 waypoints (straight lines only)
        
        # Without grid (fallback to geometric simplification):
        simplified = simplify_path(path)
        # 85 waypoints -> 8-12 waypoints
    """
    if len(waypoints) <= 2:
        return waypoints
    
    # Convert to integer grid coordinates for string pulling
    int_waypoints = [(int(round(x)), int(round(y))) for x, y in waypoints]
    
    # PRIMARY: String Pulling (if grid available)
    if use_string_pulling and grid is not None and grid_width and grid_height:
        optimized = string_pulling(int_waypoints, grid, grid_width, grid_height)
        
        # Optional: Inject intermediate waypoints on very long segments
        # This helps controllers maintain cross-track accuracy
        if max_segment_length > 0:
            optimized = inject_intermediate_waypoints(optimized, max_segment_length)
        
        return optimized
    
    # FALLBACK: Geometric simplification (RDP + angle filtering)
    # Used when grid is not available or string pulling is disabled
    else:
        return _geometric_simplification(waypoints)


def _geometric_simplification(waypoints: List[Tuple[float, float]],
                              min_distance: float = 5.0,
                              angle_threshold: float = 10.0,
                              epsilon: float = 2.0) -> List[Tuple[float, float]]:
    """
    Fallback geometric simplification (used when grid is unavailable).
    Combines distance filtering, angle filtering, and Ramer-Douglas-Peucker.
    
    Args:
        waypoints: Original waypoint list
        min_distance: Minimum distance between waypoints
        angle_threshold: Angle threshold for collinear detection (degrees)
        epsilon: RDP tolerance
        
    Returns:
        Simplified waypoint list
    """
    if len(waypoints) <= 2:
        return waypoints
    
    # Step 1: Distance-based filtering - remove waypoints too close together
    distance_filtered = [waypoints[0]]  # Always keep start
    
    for i in range(1, len(waypoints) - 1):
        curr = waypoints[i]
        prev = distance_filtered[-1]
        goal = waypoints[-1]
        
        dist_to_prev = np.hypot(curr[0] - prev[0], curr[1] - prev[1])
        dist_to_goal = np.hypot(goal[0] - curr[0], goal[1] - curr[1])
        
        # Keep if far enough from previous AND not too close to goal
        if dist_to_prev >= min_distance and dist_to_goal >= min_distance:
            distance_filtered.append(curr)
    
    distance_filtered.append(waypoints[-1])  # Always keep goal
    
    # Step 2: Angle-based filtering - remove collinear points
    angle_filtered = [distance_filtered[0]]  # Keep start
    
    for i in range(1, len(distance_filtered) - 1):
        prev = angle_filtered[-1]
        curr = distance_filtered[i]
        next_pt = distance_filtered[i + 1]
        
        # Vectors
        v1 = np.array([curr[0] - prev[0], curr[1] - prev[1]])
        v2 = np.array([next_pt[0] - curr[0], next_pt[1] - curr[1]])
        
        # Normalize
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        if mag1 > 0.1 and mag2 > 0.1:
            v1_norm = v1 / mag1
            v2_norm = v2 / mag2
            
            # Angle between vectors
            dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_product))
            
            # Keep if significant direction change
            if angle > angle_threshold:
                angle_filtered.append(curr)
        else:
            # Keep very short segments (might be important)
            angle_filtered.append(curr)
    
    angle_filtered.append(distance_filtered[-1])  # Keep goal
    
    # Step 3: Ramer-Douglas-Peucker - geometrically optimal simplification
    def rdp(points: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
        """
        Ramer-Douglas-Peucker algorithm for line simplification.
        Recursively removes points that deviate less than epsilon from the line.
        """
        if len(points) <= 2:
            return points
        
        # Find point with maximum distance from line between start and end
        start = np.array(points[0])
        end = np.array(points[-1])
        
        max_dist = 0
        max_idx = 0
        
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 0.001:  # Degenerate case
            return [points[0], points[-1]]
        
        line_unit = line_vec / line_len
        
        for i in range(1, len(points) - 1):
            point = np.array(points[i])
            
            # Point-to-line distance
            point_vec = point - start
            proj_length = np.dot(point_vec, line_unit)
            
            # Clamp projection to line segment
            proj_length = np.clip(proj_length, 0, line_len)
            proj_point = start + proj_length * line_unit
            
            dist = np.linalg.norm(point - proj_point)
            
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If max distance is greater than epsilon, recursively simplify
        if max_dist > eps:
            # Recursively simplify both segments
            left = rdp(points[:max_idx + 1], eps)
            right = rdp(points[max_idx:], eps)
            
            # Combine results (remove duplicate middle point)
            return left[:-1] + right
        else:
            # All points are close enough to the line, keep only endpoints
            return [points[0], points[-1]]
    
    # Apply RDP if we still have many points
    if len(angle_filtered) > 4:
        rdp_simplified = rdp(angle_filtered, epsilon)
    else:
        rdp_simplified = angle_filtered
    
    return rdp_simplified


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
                  step_callback: Optional[callable] = None,
                  simplify: bool = True,
                  max_segment_length: float = 15.0) -> Optional[List[Tuple[float, float]]]:
        """
        Find the shortest path from start to goal using A*.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            verbose: Print search progress
            step_callback: Optional callback function called at each step with
                         (open_set_positions, closed_set, current_position)
            simplify: Apply string pulling optimization (default=True)
            max_segment_length: Max segment length for waypoint injection (default=15.0)
            
        Returns:
            List of (x, y) waypoints representing the optimized path, or None if no path exists
            By default returns simplified path with string pulling applied.
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
                
                # Apply string pulling optimization by default
                if simplify:
                    if verbose:
                        print(f"  Applying string pulling optimization...")
                    optimized_path = string_pulling(path, self.planning_grid, 
                                                   self.grid_world.width, self.grid_world.height)
                    if verbose:
                        print(f"  Optimized to {len(optimized_path)} waypoints")
                    
                    # Optionally inject intermediate waypoints on long segments
                    if max_segment_length > 0:
                        optimized_path = inject_intermediate_waypoints(optimized_path, max_segment_length)
                        if verbose:
                            print(f"  Final path: {len(optimized_path)} waypoints (with injection)")
                    
                    return optimized_path
                else:
                    # Return raw path as float tuples
                    return [(float(x), float(y)) for x, y in path]
            
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
    Create a realistic complex scenario with coastlines, islands, and narrow passages.
    
    Returns:
        Tuple of (grid_world, start, goal)
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.environment.grid_world import GridWorld
    
    # Create a large grid
    grid = GridWorld(width=100, height=100, cell_size=10.0)
    
    print("\nCreating realistic coastal scenario...")
    
    # 1. Mainland Coastline (West side)
    # Irregular shape using overlapping rectangles
    grid.add_obstacle(x=0, y=0, width=15, height=100)  # Base coast
    grid.add_obstacle(x=15, y=20, width=10, height=15) # Peninsula 1
    grid.add_obstacle(x=15, y=60, width=15, height=20) # Peninsula 2 (Headland)
    grid.add_circular_obstacle(center_x=25, center_y=70, radius=8) # Rounded tip
    
    # 2. Large Island (North-East)
    grid.add_circular_obstacle(center_x=75, center_y=75, radius=15)
    grid.add_obstacle(x=70, y=60, width=10, height=15) # Southern extension
    
    # 3. Archipelago / Reefs (Center-East)
    # A cluster of small islands creating a "minefield" of obstacles
    islands = [
        (50, 40, 4), (60, 35, 3), (55, 25, 5), 
        (70, 30, 4), (65, 45, 3), (80, 20, 6)
    ]
    for x, y, r in islands:
        grid.add_circular_obstacle(center_x=x, center_y=y, radius=r)
        
    # 4. Breakwater / Artificial Structure (South-East)
    # Protecting a bay area
    grid.add_obstacle(x=60, y=5, width=30, height=2)   # Horizontal part
    grid.add_obstacle(x=60, y=5, width=2, height=10)   # Vertical tip
    
    # 5. Narrow Channel (The "Strait")
    # Between the Headland (West) and a central island
    grid.add_circular_obstacle(center_x=45, center_y=65, radius=6)
    
    # 6. Scattered Rocks (Random small obstacles)
    rocks = [(35, 85), (40, 90), (30, 80), (90, 50), (95, 45)]
    for x, y in rocks:
        grid.add_circular_obstacle(center_x=x, center_y=y, radius=2)

    # Define start and goal
    # Start in the South-West (open water)
    start = (25, 10)
    
    # Goal in the North-East (behind the large island, requiring navigation)
    goal = (90, 90)
    
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
            # You can customize the figure size: figsize=(width, height) in inches
            # Example: figsize=(16, 16) for a larger window
            animator.animate(start, goal, interval=30, figsize=(10, 8))
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