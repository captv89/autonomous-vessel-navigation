"""
Reinforcement Learning Environment for Autonomous Vessel Navigation

Gymnasium-compatible environment that wraps the classical navigation system.
Designed for training RL agents to navigate through static and dynamic obstacles.

Learning Progression:
1. Phase 1: Static obstacles only
2. Phase 2: Add dynamic obstacles (CURRENT)
3. Phase 3: Multi-vessel scenarios
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.transforms import Affine2D

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.environment.grid_world import GridWorld
from src.environment.dynamic_obstacles import DynamicObstacleManager, MovingVessel
from src.environment.collision_detection import CollisionDetector
from src.vessel.vessel_model import NomotoVessel


class VesselNavigationEnv(gym.Env):
    """
    Gymnasium environment for vessel navigation with RL.
    
    PHASE 2: Static + Dynamic Obstacles
    
    OBSERVATION SPACE (Hybrid approach - best of both worlds):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    [0]     Goal distance (normalized by max distance)
    [1]     Goal angle relative to heading (-1 to 1, normalized from -π to π)
    [2]     Current speed (normalized)
    [3]     Current turn rate (normalized)
    [4]     Current rudder angle (normalized)
    [5-20]  16-ray LIDAR distances (0-1, like robot sensors)
    [21-36] 16-ray obstacle types (0=free, 0.5=static, 1.0=dynamic)
    [37-40] Closest dynamic obstacle: [distance, bearing, CPA, TCPA] (normalized)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Total: 41 values
    
    Why this design?
    - Goal info: Essential for navigation
    - Vessel state: Physics-aware decisions (includes rudder for dynamics awareness)
    - LIDAR: Like a robot's sensors - simple, effective
    - Dynamic obstacle info: CPA/TCPA for collision avoidance decisions
    
    ACTION SPACE (Discrete - easier to learn):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    0: Hard left  (-30°)
    1: Soft left  (-15°)
    2: Straight   (0°)
    3: Soft right (+15°)
    4: Hard right (+30°)
    5: Slow down  (0.5x speed)
    6: Speed up   (1.5x speed)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Total: 7 discrete actions
    
    REWARD FUNCTION (Phase 2 - Dynamic Obstacles):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    +1000   Reach goal
    -1000   Collision with static obstacle
    -500    Collision with dynamic obstacle
    -100    Out of bounds
    -1      Per timestep (encourages efficiency)
    +10     Progress toward goal (shaped reward)
    -10     Moving away from goal
    -50     Near-miss with dynamic obstacle (CPA < warning_distance)
    +20     Safe avoidance (passed obstacle with safe CPA)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, 
                 grid_size: int = 100,
                 cell_size: float = 10.0,
                 max_steps: int = 1000,
                 goal_threshold: float = 5.0,
                 num_lidar_rays: int = 16,
                 lidar_range: float = 50.0,
                 # Phase 2: Dynamic obstacle parameters
                 num_dynamic_obstacles: int = 2,
                 dynamic_obstacle_speed_range: Tuple[float, float] = (3.0, 7.0),
                 safe_distance: float = 8.0,
                 warning_distance: float = 15.0,
                 render_mode: Optional[str] = None):
        """
        Initialize the RL navigation environment with dynamic obstacles.
        
        Args:
            grid_size: Size of grid (grid_size x grid_size)
            cell_size: Size of each cell in meters
            max_steps: Maximum steps per episode
            goal_threshold: Distance to goal to consider "reached"
            num_lidar_rays: Number of LIDAR rays for sensing
            lidar_range: Maximum LIDAR range in meters
            num_dynamic_obstacles: Number of dynamic obstacles to spawn
            dynamic_obstacle_speed_range: (min_speed, max_speed) for obstacles
            safe_distance: CPA below this is a collision
            warning_distance: CPA below this triggers near-miss penalty
            render_mode: 'human' for visualization, None for training
        """
        super().__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        self.num_lidar_rays = num_lidar_rays
        self.lidar_range = lidar_range
        self.render_mode = render_mode
        
        # Phase 2: Dynamic obstacle parameters
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.dynamic_obstacle_speed_range = dynamic_obstacle_speed_range
        self.safe_distance = safe_distance
        self.warning_distance = warning_distance
        
        # Physics parameters (matching Nomoto model from classical system)
        self.vessel_max_speed = 10.0  # m/s
        self.vessel_nominal_speed = 5.0  # m/s
        self.K = 0.3  # Nomoto gain
        self.T = 5.0  # Nomoto time constant
        
        # Create environment components (initialized in reset)
        self.grid_world = None
        self.vessel = None
        self.obstacle_manager = None
        self.collision_detector = CollisionDetector(
            safe_distance=safe_distance,
            warning_distance=warning_distance,
            time_horizon=60.0
        )
        
        # State tracking
        self.current_step = 0
        self.goal_position = None
        self.start_position = None
        self.previous_distance_to_goal = None
        
        # Episode statistics
        self.total_reward = 0
        self.static_collisions = 0
        self.dynamic_collisions = 0
        self.near_misses = 0
        self.safe_avoidances = 0
        self.reached_goal = False
        
        # Track obstacles for avoidance rewards
        self._tracked_obstacles = {}  # {obstacle_id: {'passed': bool, 'min_cpa': float}}
        
        # Define action space: 7 discrete actions
        self.action_space = spaces.Discrete(7)
        
        # Action mapping
        self.action_to_rudder = {
            0: -np.radians(30),  # Hard left
            1: -np.radians(15),  # Soft left
            2: 0.0,               # Straight
            3: np.radians(15),    # Soft right
            4: np.radians(30),    # Hard right
            5: 0.0,               # Slow down (rudder straight)
            6: 0.0,               # Speed up (rudder straight)
        }
        
        self.action_to_speed_factor = {
            0: 1.0,   # Normal speed
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 1.0,
            5: 0.5,   # Slow down
            6: 1.5,   # Speed up
        }
        
        # Define observation space: 41 values (Phase 2)
        # [0-1]: goal info, [2-4]: vessel state, [5-20]: lidar distances, 
        # [21-36]: obstacle types, [37-40]: closest dynamic obstacle info
        obs_dim = 2 + 3 + num_lidar_rays + num_lidar_rays + 4
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Rendering
        self.fig = None
        self.ax = None
        self._trajectory = []  # Store vessel trajectory for rendering
        
        print(f"✓ Created VesselNavigationEnv (Phase 2: Dynamic Obstacles)")
        print(f"  Grid: {grid_size}x{grid_size} cells ({grid_size * cell_size}m x {grid_size * cell_size}m)")
        print(f"  Observation space: {obs_dim} values")
        print(f"  Action space: {self.action_space.n} discrete actions")
        print(f"  LIDAR: {num_lidar_rays} rays, range {lidar_range}m")
        print(f"  Dynamic obstacles: {num_dynamic_obstacles} (speed: {dynamic_obstacle_speed_range})")
        print(f"  Safe/Warning distance: {safe_distance}/{warning_distance}m")
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options:
                - 'obstacles': List of static obstacle dicts
                - 'start': Start position tuple
                - 'goal': Goal position tuple
                - 'dynamic_obstacles': List of dynamic obstacle configs
                - 'num_dynamic_obstacles': Override default count
            
        Returns:
            observation, info dict
        """
        super().reset(seed=seed)
        
        # Reset tracking
        self.current_step = 0
        self.total_reward = 0
        self.static_collisions = 0
        self.dynamic_collisions = 0
        self.near_misses = 0
        self.safe_avoidances = 0
        self.reached_goal = False
        self._tracked_obstacles = {}
        self._trajectory = []
        
        # Create fresh grid world
        self.grid_world = GridWorld(self.grid_size, self.grid_size, self.cell_size)
        
        # Add static obstacles
        if options and 'obstacles' in options:
            for obs in options['obstacles']:
                if obs['type'] == 'rectangle':
                    self.grid_world.add_obstacle(obs['x'], obs['y'], obs['w'], obs['h'])
                elif obs['type'] == 'circle':
                    self.grid_world.add_circular_obstacle(obs['x'], obs['y'], obs['r'])
        else:
            self._add_default_obstacles()
        
        # Set start and goal positions
        if options and 'start' in options:
            start_cell = options['start']
        else:
            start_cell = self.grid_world.get_random_free_position()
        
        if options and 'goal' in options:
            goal_cell = options['goal']
        else:
            goal_cell = self._find_distant_goal(start_cell)
        
        # Convert to world coordinates (center of cell)
        self.start_position = (
            start_cell[0] * self.cell_size + self.cell_size / 2,
            start_cell[1] * self.cell_size + self.cell_size / 2
        )
        self.goal_position = (
            goal_cell[0] * self.cell_size + self.cell_size / 2,
            goal_cell[1] * self.cell_size + self.cell_size / 2
        )
        
        # Create vessel at start position with initial heading toward goal
        initial_heading = np.arctan2(
            self.goal_position[1] - self.start_position[1],
            self.goal_position[0] - self.start_position[0]
        )
        
        self.vessel = NomotoVessel(
            x=self.start_position[0],
            y=self.start_position[1],
            heading=initial_heading,
            speed=self.vessel_nominal_speed,
            max_speed=self.vessel_max_speed,
            K=self.K,
            T=self.T,
            rudder_rate=0  # IMO standard rudder rate
        )
        
        # Phase 2: Create dynamic obstacles
        self._setup_dynamic_obstacles(options)
        
        # Initialize distance tracking
        self.previous_distance_to_goal = self._distance_to_goal()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _setup_dynamic_obstacles(self, options: Optional[Dict] = None):
        """Setup dynamic obstacle manager with obstacles."""
        self.obstacle_manager = DynamicObstacleManager()
        
        # Determine number of obstacles
        num_obs = self.num_dynamic_obstacles
        if options and 'num_dynamic_obstacles' in options:
            num_obs = options['num_dynamic_obstacles']
        
        # Use provided dynamic obstacles or generate random ones
        if options and 'dynamic_obstacles' in options:
            for obs_config in options['dynamic_obstacles']:
                vessel = self.obstacle_manager.add_obstacle(
                    x=obs_config['x'],
                    y=obs_config['y'],
                    heading=obs_config['heading'],
                    speed=obs_config.get('speed', 5.0),
                    behavior=obs_config.get('behavior', 'straight')
                )
                # Track for avoidance rewards
                self._tracked_obstacles[vessel.obstacle.id] = {
                    'passed': False,
                    'min_cpa': float('inf'),
                    'had_risk': False
                }
        else:
            # Generate random dynamic obstacles
            self._generate_random_dynamic_obstacles(num_obs)
    
    def _generate_random_dynamic_obstacles(self, num_obstacles: int):
        """Generate random dynamic obstacles that will cross the vessel's path."""
        vessel_pos = self.vessel.get_position()
        
        for _ in range(num_obstacles):
            # Try to place obstacle in a position that creates an interesting encounter
            for attempt in range(10):  # Max 10 attempts per obstacle
                # Random position at edge of scenario or in open water
                spawn_type = np.random.choice(['edge', 'crossing', 'overtaking'])
                
                if spawn_type == 'edge':
                    # Spawn at random edge
                    edge = np.random.randint(4)
                    world_size = self.grid_size * self.cell_size
                    if edge == 0:  # Top
                        x = np.random.uniform(0.2, 0.8) * world_size
                        y = world_size - 10
                        heading = np.random.uniform(-np.pi, -np.pi/2)  # Heading south
                    elif edge == 1:  # Right
                        x = world_size - 10
                        y = np.random.uniform(0.2, 0.8) * world_size
                        heading = np.random.uniform(np.pi/2, 3*np.pi/2)  # Heading west
                    elif edge == 2:  # Bottom
                        x = np.random.uniform(0.2, 0.8) * world_size
                        y = 10
                        heading = np.random.uniform(0, np.pi)  # Heading north
                    else:  # Left
                        x = 10
                        y = np.random.uniform(0.2, 0.8) * world_size
                        heading = np.random.uniform(-np.pi/2, np.pi/2)  # Heading east
                
                elif spawn_type == 'crossing':
                    # Spawn to create crossing situation
                    # Position offset from vessel's path
                    offset_dist = np.random.uniform(30, 60)
                    offset_angle = np.random.choice([np.pi/2, -np.pi/2])  # Port or starboard
                    vessel_heading = self.vessel.get_heading()
                    
                    x = vessel_pos[0] + offset_dist * np.cos(vessel_heading + offset_angle)
                    y = vessel_pos[1] + offset_dist * np.sin(vessel_heading + offset_angle)
                    
                    # Head toward vessel's projected position
                    heading = vessel_heading - offset_angle + np.random.uniform(-0.3, 0.3)
                
                else:  # overtaking
                    # Spawn behind vessel
                    behind_dist = np.random.uniform(40, 80)
                    vessel_heading = self.vessel.get_heading()
                    
                    x = vessel_pos[0] - behind_dist * np.cos(vessel_heading)
                    y = vessel_pos[1] - behind_dist * np.sin(vessel_heading)
                    
                    # Same heading but faster
                    heading = vessel_heading + np.random.uniform(-0.2, 0.2)
                
                # Check if position is valid (not on obstacle, in bounds)
                grid_x = int(x / self.cell_size)
                grid_y = int(y / self.cell_size)
                
                if (0 < grid_x < self.grid_size - 1 and 
                    0 < grid_y < self.grid_size - 1 and
                    self.grid_world.grid[grid_y, grid_x] < 0.5):
                    
                    # Random speed within range
                    speed = np.random.uniform(*self.dynamic_obstacle_speed_range)
                    
                    # Random behavior
                    behavior = np.random.choice(['straight', 'straight', 'waypoint'])
                    
                    vessel = self.obstacle_manager.add_obstacle(
                        x=x, y=y,
                        heading=heading,
                        speed=speed,
                        behavior=behavior
                    )
                    
                    # If waypoint behavior, add some waypoints
                    if behavior == 'waypoint':
                        waypoints = self._generate_waypoints(x, y, heading, speed)
                        vessel.set_waypoints(waypoints)
                    
                    # Track for avoidance rewards
                    self._tracked_obstacles[vessel.obstacle.id] = {
                        'passed': False,
                        'min_cpa': float('inf'),
                        'had_risk': False
                    }
                    break
    
    def _generate_waypoints(self, x: float, y: float, heading: float, speed: float) -> List[Tuple[float, float]]:
        """Generate waypoints for a dynamic obstacle."""
        waypoints = []
        current_x, current_y = x, y
        current_heading = heading
        world_size = self.grid_size * self.cell_size
        
        for _ in range(3):
            # Project forward with slight heading variation
            dist = np.random.uniform(50, 100)
            current_heading += np.random.uniform(-0.3, 0.3)
            
            next_x = current_x + dist * np.cos(current_heading)
            next_y = current_y + dist * np.sin(current_heading)
            
            # Clamp to world bounds
            next_x = np.clip(next_x, 10, world_size - 10)
            next_y = np.clip(next_y, 10, world_size - 10)
            
            waypoints.append((next_x, next_y))
            current_x, current_y = next_x, next_y
        
        return waypoints
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action index (0-6)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to control commands
        rudder_command = self.action_to_rudder[action]
        speed_factor = self.action_to_speed_factor[action]
        desired_speed = self.vessel_nominal_speed * speed_factor
        
        # Update vessel physics
        dt = 0.1
        self.vessel.update(dt, rudder_command=rudder_command, desired_speed=desired_speed)
        
        # Phase 2: Update dynamic obstacles
        self.obstacle_manager.update_all(dt)
        
        # Store trajectory for rendering
        pos = self.vessel.get_position()
        self._trajectory.append(pos)
        
        # Check termination conditions and calculate reward
        terminated = False
        reward = 0.0
        
        # 1. Check goal reached
        distance_to_goal = self._distance_to_goal()
        if distance_to_goal < self.goal_threshold:
            reward += 1000.0
            terminated = True
            self.reached_goal = True
        
        # 2. Check collision with static obstacles
        if self._check_static_collision():
            reward -= 1000.0
            terminated = True
            self.static_collisions += 1
        
        # 3. Phase 2: Check collision with dynamic obstacles
        dynamic_collision, near_miss, safe_avoidance = self._check_dynamic_obstacles()
        if dynamic_collision:
            reward -= 500.0
            terminated = True
            self.dynamic_collisions += 1
        
        if near_miss:
            reward -= 50.0
            self.near_misses += 1
        
        if safe_avoidance:
            reward += 20.0
            self.safe_avoidances += 1
        
        # 4. Check out of bounds
        if self._is_out_of_bounds():
            reward -= 100.0
            terminated = True
        
        # 5. Time penalty (encourages efficiency)
        reward -= 1.0
        
        # 6. Progress reward (shaped reward)
        progress = self.previous_distance_to_goal - distance_to_goal
        if progress > 0:
            reward += 10.0  # Moving toward goal
        else:
            reward -= 10.0  # Moving away
        
        self.previous_distance_to_goal = distance_to_goal
        
        # Check truncation (max steps)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # Update stats
        self.total_reward += reward
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _check_dynamic_obstacles(self) -> Tuple[bool, bool, bool]:
        """
        Check interactions with dynamic obstacles.
        
        Returns:
            (collision, near_miss, safe_avoidance)
        """
        collision = False
        near_miss = False
        safe_avoidance = False
        
        vessel_pos = self.vessel.get_position()
        vessel_heading = self.vessel.get_heading()
        vessel_speed = self.vessel.get_speed()
        
        for moving_vessel in self.obstacle_manager.obstacles:
            obs_id = moving_vessel.obstacle.id
            obs_pos = moving_vessel.get_position()
            obs_heading = moving_vessel.get_heading()
            obs_speed = moving_vessel.obstacle.speed
            
            # Assess collision risk using classical system's detector
            info = self.collision_detector.assess_collision_risk(
                pos1=vessel_pos,
                heading1=vessel_heading,
                speed1=vessel_speed,
                pos2=obs_pos,
                heading2=obs_heading,
                speed2=obs_speed,
                vessel1_id=0,
                vessel2_id=obs_id
            )
            
            # Track minimum CPA for this obstacle
            tracking = self._tracked_obstacles.get(obs_id, {
                'passed': False,
                'min_cpa': float('inf'),
                'had_risk': False
            })
            
            # Update minimum CPA
            if info.cpa_distance < tracking['min_cpa']:
                tracking['min_cpa'] = info.cpa_distance
            
            # Check for collision (current distance < safe_distance)
            if info.current_distance < self.safe_distance:
                collision = True
            
            # Check for near-miss (CPA within warning but outside safe)
            elif info.is_collision_risk and info.cpa_distance < self.warning_distance:
                tracking['had_risk'] = True
                if info.cpa_distance < self.safe_distance * 1.5:
                    near_miss = True
            
            # Check if obstacle has passed (TCPA < 0 or very large)
            if not tracking['passed'] and (info.tcpa < 0 or info.tcpa > 120):
                tracking['passed'] = True
                
                # Reward safe avoidance: had risk but maintained safe distance
                if tracking['had_risk'] and tracking['min_cpa'] >= self.safe_distance:
                    safe_avoidance = True
            
            self._tracked_obstacles[obs_id] = tracking
        
        return collision, near_miss, safe_avoidance
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (what the agent sees).
        
        Returns:
            41-dimensional observation vector (Phase 2)
        """
        pos = self.vessel.get_position()
        heading = self.vessel.get_heading()
        speed = self.vessel.get_speed()
        turn_rate = self.vessel.get_turn_rate()
        rudder_angle = self.vessel.get_rudder_angle()
        
        # 1. Goal information (2 values)
        dx_goal = self.goal_position[0] - pos[0]
        dy_goal = self.goal_position[1] - pos[1]
        distance_to_goal = np.sqrt(dx_goal**2 + dy_goal**2)
        
        max_distance = np.sqrt(2) * self.grid_size * self.cell_size
        normalized_distance = distance_to_goal / max_distance
        
        angle_to_goal = np.arctan2(dy_goal, dx_goal) - heading
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))
        normalized_angle = angle_to_goal / np.pi
        
        # 2. Vessel state (3 values)
        normalized_speed = speed / self.vessel_max_speed
        normalized_turn_rate = np.clip(turn_rate / 0.5, -1, 1)
        normalized_rudder = rudder_angle / np.radians(35)  # Max rudder is 35°
        
        # 3. LIDAR sensor readings (32 values: 16 distances + 16 types)
        lidar_distances, lidar_types = self._get_lidar_readings()
        
        # 4. Phase 2: Closest dynamic obstacle info (4 values)
        closest_obs_info = self._get_closest_dynamic_obstacle_info()
        
        # Combine all observations
        observation = np.concatenate([
            [normalized_distance, normalized_angle],
            [normalized_speed, normalized_turn_rate, normalized_rudder],
            lidar_distances,
            lidar_types,
            closest_obs_info
        ]).astype(np.float32)
        
        return observation
    
    def _get_closest_dynamic_obstacle_info(self) -> np.ndarray:
        """
        Get information about the closest/most threatening dynamic obstacle.
        
        Returns:
            [distance, bearing, CPA, TCPA] - all normalized to [-1, 1]
        """
        if len(self.obstacle_manager.obstacles) == 0:
            return np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)  # No obstacles
        
        vessel_pos = self.vessel.get_position()
        vessel_heading = self.vessel.get_heading()
        vessel_speed = self.vessel.get_speed()
        
        closest_distance = float('inf')
        closest_info = None
        
        for moving_vessel in self.obstacle_manager.obstacles:
            obs_pos = moving_vessel.get_position()
            obs_heading = moving_vessel.get_heading()
            obs_speed = moving_vessel.obstacle.speed
            
            info = self.collision_detector.assess_collision_risk(
                pos1=vessel_pos,
                heading1=vessel_heading,
                speed1=vessel_speed,
                pos2=obs_pos,
                heading2=obs_heading,
                speed2=obs_speed
            )
            
            # Prioritize by collision risk (CPA), then by distance
            threat_score = info.cpa_distance if info.tcpa > 0 else info.current_distance
            
            if threat_score < closest_distance:
                closest_distance = threat_score
                closest_info = info
        
        if closest_info is None:
            return np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        
        # Normalize values
        max_dist = self.lidar_range * 2
        normalized_dist = np.clip(closest_info.current_distance / max_dist, 0, 1)
        normalized_bearing = closest_info.relative_bearing / np.pi
        normalized_cpa = np.clip(closest_info.cpa_distance / max_dist, 0, 1)
        
        # Normalize TCPA (0-60s mapped to 0-1)
        if closest_info.tcpa < 0 or closest_info.tcpa > 60:
            normalized_tcpa = 1.0  # Not approaching
        else:
            normalized_tcpa = closest_info.tcpa / 60.0
        
        return np.array([normalized_dist, normalized_bearing, normalized_cpa, normalized_tcpa], 
                       dtype=np.float32)
    
    def _get_lidar_readings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate LIDAR sensor - cast rays in all directions.
        Detects both static and dynamic obstacles.
        
        Returns:
            distances: Array of normalized distances (0-1)
            types: Array of obstacle types (0=free, 0.5=static, 1.0=dynamic)
        """
        pos = self.vessel.get_position()
        heading = self.vessel.get_heading()
        
        distances = np.ones(self.num_lidar_rays, dtype=np.float32)
        types = np.zeros(self.num_lidar_rays, dtype=np.float32)
        
        for i in range(self.num_lidar_rays):
            angle = heading + (2 * np.pi * i / self.num_lidar_rays)
            
            # Cast ray for static and dynamic obstacles
            distance, obs_type = self._cast_ray(pos, angle, self.lidar_range)
            
            distances[i] = distance / self.lidar_range
            types[i] = obs_type
        
        return distances, types
    
    def _cast_ray(self, start_pos: Tuple[float, float], 
                  angle: float, max_range: float) -> Tuple[float, float]:
        """
        Cast a single ray and find first obstacle.
        Checks both static obstacles and dynamic vessels.
        
        Args:
            start_pos: Starting position (x, y)
            angle: Ray angle in radians
            max_range: Maximum ray length
            
        Returns:
            distance: Distance to obstacle (or max_range)
            obs_type: 0=free, 0.5=static, 1.0=dynamic
        """
        num_steps = 50
        step_size = max_range / num_steps
        
        dx = np.cos(angle) * step_size
        dy = np.sin(angle) * step_size
        
        closest_dist = max_range
        closest_type = 0.0
        
        # Check against static obstacles
        for step in range(1, num_steps + 1):
            x = start_pos[0] + dx * step
            y = start_pos[1] + dy * step
            
            grid_x = int(x / self.cell_size)
            grid_y = int(y / self.cell_size)
            
            # Check bounds
            if (grid_x < 0 or grid_x >= self.grid_size or 
                grid_y < 0 or grid_y >= self.grid_size):
                dist = step * step_size
                if dist < closest_dist:
                    closest_dist = dist
                    closest_type = 0.5  # Boundary treated as static
                break
            
            # Check static obstacle
            if self.grid_world.grid[grid_y, grid_x] > 0.5:
                dist = step * step_size
                if dist < closest_dist:
                    closest_dist = dist
                    closest_type = 0.5  # Static obstacle
                break
        
        # Phase 2: Check against dynamic obstacles
        for moving_vessel in self.obstacle_manager.obstacles:
            obs_pos = moving_vessel.get_position()
            
            # Simple point-to-line distance check
            # Vector from start to obstacle
            to_obs_x = obs_pos[0] - start_pos[0]
            to_obs_y = obs_pos[1] - start_pos[1]
            
            # Project onto ray direction
            ray_dir_x = np.cos(angle)
            ray_dir_y = np.sin(angle)
            
            proj_dist = to_obs_x * ray_dir_x + to_obs_y * ray_dir_y
            
            if proj_dist > 0 and proj_dist < max_range:
                # Perpendicular distance to ray
                perp_dist = abs(to_obs_x * ray_dir_y - to_obs_y * ray_dir_x)
                
                # Vessel detection radius (approximate vessel size)
                vessel_radius = 3.0
                
                if perp_dist < vessel_radius:
                    # Ray hits dynamic obstacle
                    hit_dist = proj_dist - np.sqrt(max(0, vessel_radius**2 - perp_dist**2))
                    if hit_dist > 0 and hit_dist < closest_dist:
                        closest_dist = hit_dist
                        closest_type = 1.0  # Dynamic obstacle
        
        return closest_dist, closest_type
    
    def _check_static_collision(self) -> bool:
        """Check if vessel collided with static obstacle."""
        pos = self.vessel.get_position()
        grid_x = int(pos[0] / self.cell_size)
        grid_y = int(pos[1] / self.cell_size)
        
        if (grid_x < 0 or grid_x >= self.grid_size or 
            grid_y < 0 or grid_y >= self.grid_size):
            return False
        
        return self.grid_world.grid[grid_y, grid_x] > 0.5
    
    def _is_out_of_bounds(self) -> bool:
        """Check if vessel is out of bounds."""
        pos = self.vessel.get_position()
        world_size = self.grid_size * self.cell_size
        return (pos[0] < 0 or pos[0] > world_size or 
                pos[1] < 0 or pos[1] > world_size)
    
    def _distance_to_goal(self) -> float:
        """Calculate Euclidean distance to goal."""
        pos = self.vessel.get_position()
        dx = self.goal_position[0] - pos[0]
        dy = self.goal_position[1] - pos[1]
        return np.sqrt(dx**2 + dy**2)
    
    def _add_default_obstacles(self):
        """Add default random obstacles for training variety."""
        num_obstacles = np.random.randint(3, 6)
        
        for _ in range(num_obstacles):
            if np.random.rand() < 0.5:
                x = np.random.randint(10, self.grid_size - 20)
                y = np.random.randint(10, self.grid_size - 20)
                w = np.random.randint(5, 15)
                h = np.random.randint(5, 15)
                self.grid_world.add_obstacle(x, y, w, h)
            else:
                x = np.random.randint(15, self.grid_size - 15)
                y = np.random.randint(15, self.grid_size - 15)
                r = np.random.randint(3, 10)
                self.grid_world.add_circular_obstacle(x, y, r)
    
    def _find_distant_goal(self, start_cell: Tuple[int, int]) -> Tuple[int, int]:
        """Find a goal position far from start."""
        free_cells = np.argwhere(self.grid_world.grid < 0.5)
        
        if len(free_cells) == 0:
            raise ValueError("No free cells for goal!")
        
        max_dist = 0
        goal_cell = start_cell
        
        for cell_yx in free_cells:
            cell_xy = (cell_yx[1], cell_yx[0])
            dist = np.sqrt((cell_xy[0] - start_cell[0])**2 + 
                          (cell_xy[1] - start_cell[1])**2)
            if dist > max_dist:
                max_dist = dist
                goal_cell = cell_xy
        
        return goal_cell
    
    def _get_info(self) -> Dict:
        """Get additional info about current state."""
        return {
            'step': self.current_step,
            'distance_to_goal': self._distance_to_goal(),
            'total_reward': self.total_reward,
            'vessel_position': self.vessel.get_position(),
            'vessel_heading': np.degrees(self.vessel.get_heading()),
            'vessel_speed': self.vessel.get_speed(),
            'static_collisions': self.static_collisions,
            'dynamic_collisions': self.dynamic_collisions,
            'near_misses': self.near_misses,
            'safe_avoidances': self.safe_avoidances,
            'reached_goal': self.reached_goal,
            'num_dynamic_obstacles': len(self.obstacle_manager.obstacles),
        }
    
    def render(self):
        """Render the environment (for debugging/visualization)."""
        if self.render_mode != 'human':
            return
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            plt.ion()
        
        self.ax.clear()
        
        world_size = self.grid_size * self.cell_size
        self.ax.set_xlim(0, world_size)
        self.ax.set_ylim(0, world_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw static obstacles
        obstacle_cells = np.argwhere(self.grid_world.grid > 0.5)
        for cell_yx in obstacle_cells:
            rect = Rectangle(
                (cell_yx[1] * self.cell_size, cell_yx[0] * self.cell_size),
                self.cell_size, self.cell_size,
                facecolor='gray', edgecolor='black', alpha=0.7
            )
            self.ax.add_patch(rect)
        
        # Draw goal
        goal_circle = Circle(self.goal_position, self.goal_threshold,
                           facecolor='green', alpha=0.3, edgecolor='darkgreen', linewidth=2)
        self.ax.add_patch(goal_circle)
        self.ax.plot(self.goal_position[0], self.goal_position[1], 
                    'g*', markersize=20, label='Goal')
        
        # Draw trajectory
        if len(self._trajectory) > 1:
            traj_x = [p[0] for p in self._trajectory]
            traj_y = [p[1] for p in self._trajectory]
            self.ax.plot(traj_x, traj_y, 'b-', linewidth=1.5, alpha=0.5, label='Path')
        
        # Draw our vessel
        self._draw_vessel(self.vessel, color='blue', label='Our Vessel')
        
        # Phase 2: Draw dynamic obstacles
        colors = ['red', 'orange', 'purple', 'magenta', 'cyan']
        for i, moving_vessel in enumerate(self.obstacle_manager.obstacles):
            color = colors[i % len(colors)]
            self._draw_vessel_simple(moving_vessel, color=color)
            
            # Draw predicted position
            future_pos = moving_vessel.predict_position(5.0)
            self.ax.plot([moving_vessel.obstacle.x, future_pos[0]],
                        [moving_vessel.obstacle.y, future_pos[1]],
                        '--', color=color, alpha=0.5, linewidth=1)
            self.ax.plot(future_pos[0], future_pos[1], 'o', color=color, 
                        alpha=0.3, markersize=8)
        
        # Draw LIDAR rays
        pos = self.vessel.get_position()
        heading = self.vessel.get_heading()
        lidar_distances, lidar_types = self._get_lidar_readings()
        
        for i in range(self.num_lidar_rays):
            angle = heading + (2 * np.pi * i / self.num_lidar_rays)
            distance = lidar_distances[i] * self.lidar_range
            end_x = pos[0] + distance * np.cos(angle)
            end_y = pos[1] + distance * np.sin(angle)
            
            # Color by type: green=free, gray=static, red=dynamic
            if lidar_types[i] > 0.75:
                color = 'red'
                alpha = 0.4
            elif lidar_types[i] > 0.25:
                color = 'gray'
                alpha = 0.2
            else:
                color = 'green'
                alpha = 0.1
            
            self.ax.plot([pos[0], end_x], [pos[1], end_y], 
                        '-', color=color, alpha=alpha, linewidth=0.5)
        
        # Info text
        info = self._get_info()
        info_text = (f"Step: {info['step']}/{self.max_steps}\n"
                    f"Distance to goal: {info['distance_to_goal']:.1f}m\n"
                    f"Reward: {info['total_reward']:.1f}\n"
                    f"Speed: {info['vessel_speed']:.1f} m/s\n"
                    f"Heading: {info['vessel_heading']:.1f}°\n"
                    f"─────────────────\n"
                    f"Dynamic obs: {info['num_dynamic_obstacles']}\n"
                    f"Static collisions: {info['static_collisions']}\n"
                    f"Dynamic collisions: {info['dynamic_collisions']}\n"
                    f"Near misses: {info['near_misses']}\n"
                    f"Safe avoidances: {info['safe_avoidances']}")
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    family='monospace')
        
        self.ax.legend(loc='upper right')
        self.ax.set_title('Vessel Navigation RL Environment (Phase 2: Dynamic Obstacles)')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        
        plt.pause(0.01)
        plt.draw()
    
    def _draw_vessel(self, vessel, color='blue', label=None):
        """Draw a vessel with triangle shape."""
        pos = vessel.get_position()
        heading = vessel.get_heading()
        
        vessel_size = 4.0
        triangle_points = np.array([
            [vessel_size, 0],
            [-vessel_size/2, vessel_size/2],
            [-vessel_size/2, -vessel_size/2]
        ])
        
        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])
        triangle_rotated = triangle_points @ rotation.T
        triangle_world = triangle_rotated + np.array(pos)
        
        vessel_patch = Polygon(triangle_world, facecolor=color, 
                              edgecolor='dark' + color if color != 'blue' else 'darkblue',
                              linewidth=2, alpha=0.8, label=label)
        self.ax.add_patch(vessel_patch)
    
    def _draw_vessel_simple(self, moving_vessel, color='red'):
        """Draw a moving vessel."""
        pos = moving_vessel.get_position()
        heading = moving_vessel.get_heading()
        
        vessel_size = 3.0
        triangle_points = np.array([
            [vessel_size, 0],
            [-vessel_size/2, vessel_size/2],
            [-vessel_size/2, -vessel_size/2]
        ])
        
        rotation = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])
        triangle_rotated = triangle_points @ rotation.T
        triangle_world = triangle_rotated + np.array(pos)
        
        vessel_patch = Polygon(triangle_world, facecolor=color, 
                              edgecolor='black', linewidth=1.5, alpha=0.7)
        self.ax.add_patch(vessel_patch)
        
        # Add ID label
        self.ax.text(pos[0], pos[1] + 5, f'V{moving_vessel.obstacle.id}',
                    fontsize=8, ha='center', color=color)
    
    def close(self):
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# ============================================================================
# SCENARIO GENERATORS
# ============================================================================

def create_crossing_scenario(env: VesselNavigationEnv) -> Dict:
    """
    Create a crossing situation scenario for training.
    
    Returns:
        Options dict for env.reset()
    """
    grid_size = env.grid_size
    cell_size = env.cell_size
    world_size = grid_size * cell_size
    
    return {
        'start': (10, 10),
        'goal': (grid_size - 10, grid_size - 10),
        'obstacles': [
            {'type': 'circle', 'x': 30, 'y': 50, 'r': 5},
            {'type': 'circle', 'x': 70, 'y': 40, 'r': 5},
        ],
        'dynamic_obstacles': [
            {
                'x': world_size * 0.7,
                'y': world_size * 0.2,
                'heading': np.pi * 0.75,  # Northwest
                'speed': 5.0,
                'behavior': 'straight'
            },
            {
                'x': world_size * 0.2,
                'y': world_size * 0.7,
                'heading': -np.pi * 0.25,  # Southeast
                'speed': 4.0,
                'behavior': 'straight'
            }
        ]
    }


def create_head_on_scenario(env: VesselNavigationEnv) -> Dict:
    """Create a head-on encounter scenario."""
    grid_size = env.grid_size
    cell_size = env.cell_size
    world_size = grid_size * cell_size
    
    return {
        'start': (10, grid_size // 2),
        'goal': (grid_size - 10, grid_size // 2),
        'obstacles': [],
        'dynamic_obstacles': [
            {
                'x': world_size * 0.8,
                'y': world_size * 0.5 + 10,
                'heading': np.pi,  # West (toward us)
                'speed': 5.0,
                'behavior': 'straight'
            }
        ]
    }


def create_overtaking_scenario(env: VesselNavigationEnv) -> Dict:
    """Create an overtaking scenario."""
    grid_size = env.grid_size
    cell_size = env.cell_size
    world_size = grid_size * cell_size
    
    return {
        'start': (10, grid_size // 2),
        'goal': (grid_size - 10, grid_size // 2),
        'obstacles': [],
        'dynamic_obstacles': [
            {
                'x': world_size * 0.4,
                'y': world_size * 0.5,
                'heading': 0,  # Same direction (east)
                'speed': 3.0,  # Slower than us
                'behavior': 'straight'
            }
        ]
    }


# ============================================================================
# TEST / DEMO CODE
# ============================================================================

def test_environment():
    """Test the RL environment with random actions."""
    print("=" * 70)
    print("TESTING RL NAVIGATION ENVIRONMENT (Phase 2: Dynamic Obstacles)")
    print("=" * 70)
    
    # Create environment
    env = VesselNavigationEnv(
        grid_size=100,
        max_steps=500,
        num_dynamic_obstacles=2,
        render_mode='human'
    )
    
    print("\n✓ Environment created successfully!")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test different scenarios
    scenarios = [
        ("Random", None),
        ("Crossing", create_crossing_scenario(env)),
        ("Head-on", create_head_on_scenario(env)),
    ]
    
    for scenario_name, options in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*70}")
        
        obs, info = env.reset(options=options)
        print(f"Start: ({info['vessel_position'][0]:.0f}, {info['vessel_position'][1]:.0f})")
        print(f"Goal: ({env.goal_position[0]:.0f}, {env.goal_position[1]:.0f})")
        print(f"Initial distance: {info['distance_to_goal']:.1f}m")
        print(f"Dynamic obstacles: {info['num_dynamic_obstacles']}")
        
        done = False
        step = 0
        max_demo_steps = 150
        
        while not done and step < max_demo_steps:
            # Take random action (or could use simple heuristic)
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Render every 3 steps
            if step % 3 == 0:
                env.render()
            
            step += 1
        
        print(f"\nScenario finished:")
        print(f"  Steps: {step}")
        print(f"  Total reward: {info['total_reward']:.1f}")
        print(f"  Final distance: {info['distance_to_goal']:.1f}m")
        print(f"  Reached goal: {info['reached_goal']}")
        print(f"  Static collisions: {info['static_collisions']}")
        print(f"  Dynamic collisions: {info['dynamic_collisions']}")
        print(f"  Near misses: {info['near_misses']}")
        print(f"  Safe avoidances: {info['safe_avoidances']}")
    
    env.close()
    print(f"\n{'='*70}")
    print("✓ Environment test complete!")
    print(f"{'='*70}")


def test_observation_space():
    """Test and verify observation space dimensions."""
    print("=" * 70)
    print("TESTING OBSERVATION SPACE")
    print("=" * 70)
    
    env = VesselNavigationEnv(grid_size=50, num_dynamic_obstacles=1)
    obs, info = env.reset()
    
    print(f"\nObservation shape: {obs.shape}")
    print(f"Expected: ({env.observation_space.shape[0]},)")
    
    print("\nObservation breakdown:")
    idx = 0
    print(f"  [{idx}:{idx+2}] Goal info: {obs[idx:idx+2]}")
    idx += 2
    print(f"  [{idx}:{idx+3}] Vessel state: {obs[idx:idx+3]}")
    idx += 3
    print(f"  [{idx}:{idx+16}] LIDAR distances: {obs[idx:idx+16][:4]}... (16 total)")
    idx += 16
    print(f"  [{idx}:{idx+16}] LIDAR types: {obs[idx:idx+16][:4]}... (16 total)")
    idx += 16
    print(f"  [{idx}:{idx+4}] Closest obstacle: {obs[idx:idx+4]}")
    
    print(f"\nAll values in [-1, 1]: {np.all(obs >= -1) and np.all(obs <= 1)}")
    
    env.close()


if __name__ == "__main__":
    # Run observation space test first
    test_observation_space()
    
    print("\n" + "="*70 + "\n")
    
    # Then run full environment test
    test_environment()
