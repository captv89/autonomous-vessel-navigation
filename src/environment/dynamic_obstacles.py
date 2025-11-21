"""
Dynamic Obstacles - Represents moving objects in the environment.

Includes:
- Moving vessels with predictable trajectories
- Collision detection
- Future position prediction
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DynamicObstacle:
    """
    Represents a moving obstacle (another vessel).
    
    Attributes:
        id: Unique identifier
        x: X position
        y: Y position
        heading: Heading in radians
        speed: Speed in units/s
        length: Vessel length (for collision detection)
        width: Vessel width (for collision detection)
        trajectory: Planned waypoints (if known)
    """
    id: int
    x: float
    y: float
    heading: float
    speed: float
    length: float = 3.0
    width: float = 1.5
    trajectory: Optional[List[Tuple[float, float]]] = None
    
    def __repr__(self):
        return (f"DynamicObstacle(id={self.id}, pos=({self.x:.1f},{self.y:.1f}), "
                f"heading={np.degrees(self.heading):.0f}°, speed={self.speed:.1f})")


class MovingVessel:
    """
    A vessel that moves according to a predefined behavior.
    
    Can move in different patterns:
    - Straight line
    - Following waypoints
    - Circular pattern
    - Random walk
    """
    
    def __init__(self, obstacle_id: int, x: float, y: float, 
                 heading: float, speed: float,
                 length: float = 3.0, width: float = 1.5,
                 behavior: str = 'straight'):
        """
        Initialize moving vessel.
        
        Args:
            obstacle_id: Unique ID
            x: Initial X position
            y: Initial Y position
            heading: Initial heading (radians)
            speed: Speed (units/s)
            length: Vessel length
            width: Vessel width
            behavior: Movement behavior ('straight', 'waypoint', 'circular')
        """
        self.obstacle = DynamicObstacle(
            id=obstacle_id,
            x=x, y=y,
            heading=heading,
            speed=speed,
            length=length,
            width=width
        )
        self.behavior = behavior
        self.waypoints = []
        self.current_waypoint_idx = 0
        
        # For circular behavior
        self.circle_center = None
        self.circle_radius = None
        self.angular_velocity = None
        
    def set_waypoints(self, waypoints: List[Tuple[float, float]]):
        """Set waypoints for 'waypoint' behavior."""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
    
    def set_circular_path(self, center: Tuple[float, float], radius: float):
        """Set circular path parameters."""
        self.circle_center = center
        self.circle_radius = radius
        # Angular velocity = speed / radius
        self.angular_velocity = self.obstacle.speed / radius if radius > 0 else 0
    
    def update(self, dt: float):
        """
        Update vessel position based on behavior.
        
        Args:
            dt: Time step in seconds
        """
        if self.behavior == 'straight':
            self._update_straight(dt)
        elif self.behavior == 'waypoint':
            self._update_waypoint(dt)
        elif self.behavior == 'circular':
            self._update_circular(dt)
    
    def _update_straight(self, dt: float):
        """Update position moving straight."""
        self.obstacle.x += self.obstacle.speed * np.cos(self.obstacle.heading) * dt
        self.obstacle.y += self.obstacle.speed * np.sin(self.obstacle.heading) * dt
    
    def _update_waypoint(self, dt: float):
        """Update position following waypoints."""
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints):
            # No waypoints or reached end, move straight
            self._update_straight(dt)
            return
        
        target = self.waypoints[self.current_waypoint_idx]
        
        # Calculate distance to target
        dx = target[0] - self.obstacle.x
        dy = target[1] - self.obstacle.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Check if reached waypoint
        if dist < 2.0:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.waypoints):
                return
            target = self.waypoints[self.current_waypoint_idx]
            dx = target[0] - self.obstacle.x
            dy = target[1] - self.obstacle.y
        
        # Update heading toward target
        desired_heading = np.arctan2(dy, dx)
        
        # Smooth heading change
        heading_diff = desired_heading - self.obstacle.heading
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        
        # Limit turn rate (realistic)
        max_turn = 0.1 * dt  # Max turn per timestep
        turn = np.clip(heading_diff, -max_turn, max_turn)
        self.obstacle.heading += turn
        
        # Normalize heading
        self.obstacle.heading = np.arctan2(np.sin(self.obstacle.heading),
                                          np.cos(self.obstacle.heading))
        
        # Update position
        self._update_straight(dt)
    
    def _update_circular(self, dt: float):
        """Update position moving in a circle."""
        if self.circle_center is None:
            self._update_straight(dt)
            return
        
        # Update angle
        angle_change = self.angular_velocity * dt
        
        # Current angle from center
        dx = self.obstacle.x - self.circle_center[0]
        dy = self.obstacle.y - self.circle_center[1]
        current_angle = np.arctan2(dy, dx)
        
        # New angle
        new_angle = current_angle + angle_change
        
        # New position
        self.obstacle.x = self.circle_center[0] + self.circle_radius * np.cos(new_angle)
        self.obstacle.y = self.circle_center[1] + self.circle_radius * np.sin(new_angle)
        
        # Heading is tangent to circle (perpendicular to radius)
        self.obstacle.heading = new_angle + np.pi / 2
        
        # Normalize heading
        self.obstacle.heading = np.arctan2(np.sin(self.obstacle.heading),
                                          np.cos(self.obstacle.heading))
    
    def predict_position(self, time_ahead: float) -> Tuple[float, float]:
        """
        Predict future position.
        
        Args:
            time_ahead: Time in seconds to predict ahead
            
        Returns:
            Predicted (x, y) position
        """
        if self.behavior == 'straight':
            # Simple linear prediction
            future_x = self.obstacle.x + self.obstacle.speed * np.cos(self.obstacle.heading) * time_ahead
            future_y = self.obstacle.y + self.obstacle.speed * np.sin(self.obstacle.heading) * time_ahead
            return (future_x, future_y)
        
        elif self.behavior == 'circular' and self.circle_center is not None:
            # Circular prediction
            angle_change = self.angular_velocity * time_ahead
            dx = self.obstacle.x - self.circle_center[0]
            dy = self.obstacle.y - self.circle_center[1]
            current_angle = np.arctan2(dy, dx)
            new_angle = current_angle + angle_change
            
            future_x = self.circle_center[0] + self.circle_radius * np.cos(new_angle)
            future_y = self.circle_center[1] + self.circle_radius * np.sin(new_angle)
            return (future_x, future_y)
        
        else:
            # For waypoint behavior, use simple linear prediction
            # (more sophisticated prediction would follow waypoints)
            future_x = self.obstacle.x + self.obstacle.speed * np.cos(self.obstacle.heading) * time_ahead
            future_y = self.obstacle.y + self.obstacle.speed * np.sin(self.obstacle.heading) * time_ahead
            return (future_x, future_y)
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position."""
        return (self.obstacle.x, self.obstacle.y)
    
    def get_heading(self) -> float:
        """Get current heading."""
        return self.obstacle.heading


class DynamicObstacleManager:
    """
    Manages multiple dynamic obstacles.
    """
    
    def __init__(self):
        """Initialize manager."""
        self.obstacles: List[MovingVessel] = []
        self.next_id = 0
    
    def add_obstacle(self, x: float, y: float, heading: float, 
                    speed: float, behavior: str = 'straight',
                    length: float = 3.0, width: float = 1.5) -> MovingVessel:
        """
        Add a dynamic obstacle.
        
        Args:
            x: Initial X position
            y: Initial Y position
            heading: Initial heading (radians)
            speed: Speed
            behavior: Movement behavior
            length: Vessel length
            width: Vessel width
            
        Returns:
            Created MovingVessel object
        """
        vessel = MovingVessel(
            obstacle_id=self.next_id,
            x=x, y=y,
            heading=heading,
            speed=speed,
            length=length,
            width=width,
            behavior=behavior
        )
        self.obstacles.append(vessel)
        self.next_id += 1
        return vessel
    
    def update_all(self, dt: float):
        """Update all obstacles."""
        for obstacle in self.obstacles:
            obstacle.update(dt)
    
    def get_all_positions(self) -> List[Tuple[float, float]]:
        """Get positions of all obstacles."""
        return [obs.get_position() for obs in self.obstacles]
    
    def clear(self):
        """Remove all obstacles."""
        self.obstacles.clear()
        self.next_id = 0
    
    def __len__(self):
        return len(self.obstacles)
    
    def __repr__(self):
        return f"DynamicObstacleManager({len(self.obstacles)} obstacles)"


def test_dynamic_obstacles():
    """Test dynamic obstacles."""
    print("=" * 70)
    print("DYNAMIC OBSTACLES TEST")
    print("=" * 70)
    
    # Create manager
    manager = DynamicObstacleManager()
    
    # Add straight-moving vessel
    print("\n1. Creating straight-moving vessel...")
    vessel1 = manager.add_obstacle(x=0, y=0, heading=np.radians(45), 
                                   speed=1.0, behavior='straight')
    print(f"   {vessel1.obstacle}")
    
    # Add circular-moving vessel
    print("\n2. Creating circular-moving vessel...")
    vessel2 = manager.add_obstacle(x=10, y=10, heading=0, 
                                   speed=1.5, behavior='circular')
    vessel2.set_circular_path(center=(10, 10), radius=5.0)
    print(f"   {vessel2.obstacle}")
    print(f"   Circle: center=(10,10), radius=5.0")
    
    # Add waypoint-following vessel
    print("\n3. Creating waypoint-following vessel...")
    vessel3 = manager.add_obstacle(x=0, y=20, heading=0,
                                   speed=1.0, behavior='waypoint')
    vessel3.set_waypoints([(10, 20), (20, 25), (30, 20)])
    print(f"   {vessel3.obstacle}")
    print(f"   Waypoints: {vessel3.waypoints}")
    
    print(f"\n{manager}")
    
    # Simulate for 10 seconds
    print("\n" + "=" * 70)
    print("Simulating 10 seconds of movement...")
    print("=" * 70)
    
    dt = 0.1
    for t in range(0, 100, 10):  # Print every second
        manager.update_all(dt)
        time = t * dt
        
        print(f"\nTime: {time:.1f}s")
        for i, vessel in enumerate(manager.obstacles, 1):
            pos = vessel.get_position()
            print(f"  Vessel {i}: pos=({pos[0]:.1f}, {pos[1]:.1f}), "
                  f"heading={np.degrees(vessel.get_heading()):.0f}°")
            
            # Show prediction
            future_pos = vessel.predict_position(5.0)
            print(f"    → Predicted (5s ahead): ({future_pos[0]:.1f}, {future_pos[1]:.1f})")
    
    print("\n" + "=" * 70)
    print("✓ Dynamic obstacles test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_dynamic_obstacles()