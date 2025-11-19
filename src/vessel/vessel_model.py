"""
Vessel Physics Models

Implements different levels of vessel dynamics:
1. Kinematic model - Simple position and heading
2. Nomoto model - Realistic turning dynamics
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VesselState:
    """
    Represents the complete state of a vessel.
    
    Attributes:
        x: X position (meters or grid cells)
        y: Y position (meters or grid cells)
        heading: Heading angle in radians (0 = East, π/2 = North)
        speed: Forward speed (m/s or cells/s)
        turn_rate: Rate of turn (rad/s)
    """
    x: float
    y: float
    heading: float
    speed: float
    turn_rate: float = 0.0
    
    def __repr__(self):
        return (f"VesselState(pos=({self.x:.2f}, {self.y:.2f}), "
                f"heading={np.degrees(self.heading):.1f}°, "
                f"speed={self.speed:.2f})")


class KinematicVessel:
    """
    Simple kinematic vessel model.
    
    The vessel responds instantly to control commands (no momentum).
    Good for: Initial testing and simple simulations.
    """
    
    def __init__(self, 
                 x: float = 0.0, 
                 y: float = 0.0, 
                 heading: float = 0.0,
                 speed: float = 5.0,
                 max_speed: float = 10.0,
                 max_turn_rate: float = 0.1):
        """
        Initialize kinematic vessel.
        
        Args:
            x: Initial X position
            y: Initial Y position
            heading: Initial heading in radians
            speed: Initial speed
            max_speed: Maximum speed
            max_turn_rate: Maximum turn rate (rad/s)
        """
        self.state = VesselState(x, y, heading, speed)
        self.max_speed = max_speed
        self.max_turn_rate = max_turn_rate
        
        print(f"Created KinematicVessel at ({x}, {y})")
        print(f"  Max speed: {max_speed} units/s")
        print(f"  Max turn rate: {np.degrees(max_turn_rate):.1f}°/s")
    
    def update(self, dt: float, desired_heading: Optional[float] = None,
               desired_speed: Optional[float] = None):
        """
        Update vessel state (simple kinematic model).
        
        Args:
            dt: Time step in seconds
            desired_heading: Desired heading (rad), or None to maintain current
            desired_speed: Desired speed, or None to maintain current
        """
        # Update speed (instant response in kinematic model)
        if desired_speed is not None:
            self.state.speed = np.clip(desired_speed, 0, self.max_speed)
        
        # Update heading
        if desired_heading is not None:
            # Calculate shortest angular difference
            heading_error = desired_heading - self.state.heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # Apply turn rate limit
            max_turn = self.max_turn_rate * dt
            turn = np.clip(heading_error, -max_turn, max_turn)
            self.state.heading += turn
            
            # Normalize heading to [-π, π]
            self.state.heading = np.arctan2(np.sin(self.state.heading), 
                                           np.cos(self.state.heading))
            
            self.state.turn_rate = turn / dt if dt > 0 else 0
        
        # Update position based on current heading and speed
        self.state.x += self.state.speed * np.cos(self.state.heading) * dt
        self.state.y += self.state.speed * np.sin(self.state.heading) * dt
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position."""
        return (self.state.x, self.state.y)
    
    def get_heading(self) -> float:
        """Get current heading in radians."""
        return self.state.heading
    
    def get_speed(self) -> float:
        """Get current speed."""
        return self.state.speed
    
    def __repr__(self):
        return f"KinematicVessel({self.state})"


class NomotoVessel:
    """
    Vessel with Nomoto first-order turning dynamics.
    
    More realistic - the vessel doesn't turn instantly.
    The rudder commands affect the rate of turn with a time delay.
    
    Model: T * dψ/dt + ψ = K * δ
    Where:
        ψ = yaw rate (turn rate)
        δ = rudder angle
        K = gain constant
        T = time constant
    """
    
    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0, 
                 heading: float = 0.0,
                 speed: float = 5.0,
                 max_speed: float = 10.0,
                 K: float = 0.3,
                 T: float = 5.0,
                 max_rudder: float = 0.6):
        """
        Initialize Nomoto vessel.
        
        Args:
            x: Initial X position
            y: Initial Y position
            heading: Initial heading in radians
            speed: Initial speed
            max_speed: Maximum speed
            K: Nomoto gain constant (responsiveness)
            T: Nomoto time constant (how quickly vessel responds)
            max_rudder: Maximum rudder angle (radians)
        """
        self.state = VesselState(x, y, heading, speed, turn_rate=0.0)
        self.max_speed = max_speed
        self.K = K
        self.T = T
        self.max_rudder = max_rudder
        
        # Current rudder angle
        self.rudder_angle = 0.0
        
        print(f"Created NomotoVessel at ({x}, {y})")
        print(f"  Max speed: {max_speed} units/s")
        print(f"  K (gain): {K}")
        print(f"  T (time constant): {T}s")
        print(f"  Max rudder: {np.degrees(max_rudder):.1f}°")
    
    def update(self, dt: float, rudder_command: float = 0.0,
               desired_speed: Optional[float] = None):
        """
        Update vessel state using Nomoto dynamics.
        
        Args:
            dt: Time step in seconds
            rudder_command: Desired rudder angle (-max_rudder to +max_rudder)
            desired_speed: Desired speed, or None to maintain current
        """
        # Update speed (simplified - instant response)
        if desired_speed is not None:
            self.state.speed = np.clip(desired_speed, 0, self.max_speed)
        
        # Limit rudder command
        rudder_command = np.clip(rudder_command, -self.max_rudder, self.max_rudder)
        self.rudder_angle = rudder_command
        
        # Nomoto first-order model: T * dψ/dt + ψ = K * δ
        # Rearranged: dψ/dt = (K * δ - ψ) / T
        d_turn_rate = (self.K * rudder_command - self.state.turn_rate) / self.T
        
        # Update turn rate
        self.state.turn_rate += d_turn_rate * dt
        
        # Update heading
        self.state.heading += self.state.turn_rate * dt
        
        # Normalize heading to [-π, π]
        self.state.heading = np.arctan2(np.sin(self.state.heading),
                                       np.cos(self.state.heading))
        
        # Update position
        self.state.x += self.state.speed * np.cos(self.state.heading) * dt
        self.state.y += self.state.speed * np.sin(self.state.heading) * dt
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position."""
        return (self.state.x, self.state.y)
    
    def get_heading(self) -> float:
        """Get current heading in radians."""
        return self.state.heading
    
    def get_speed(self) -> float:
        """Get current speed."""
        return self.state.speed
    
    def get_turn_rate(self) -> float:
        """Get current turn rate in rad/s."""
        return self.state.turn_rate
    
    def __repr__(self):
        return f"NomotoVessel({self.state})"


def test_vessel_models():
    """Test both vessel models."""
    print("=" * 70)
    print("VESSEL PHYSICS MODELS TEST")
    print("=" * 70)
    
    # Test 1: Kinematic vessel
    print("\n1. Testing Kinematic Vessel")
    print("-" * 70)
    
    vessel_kin = KinematicVessel(x=0, y=0, heading=0, speed=5.0, max_turn_rate=0.1)
    print(f"Initial state: {vessel_kin.state}")
    
    # Simulate for 10 seconds with a turn command
    print("\nSimulating 10 seconds with turn to 45°...")
    dt = 0.1
    desired_heading = np.radians(45)
    
    for i in range(100):  # 10 seconds at 0.1s timestep
        vessel_kin.update(dt, desired_heading=desired_heading)
    
    print(f"Final state: {vessel_kin.state}")
    print(f"Distance traveled: {np.sqrt(vessel_kin.state.x**2 + vessel_kin.state.y**2):.2f}")
    
    # Test 2: Nomoto vessel
    print("\n" + "=" * 70)
    print("2. Testing Nomoto Vessel")
    print("-" * 70)
    
    vessel_nomoto = NomotoVessel(x=0, y=0, heading=0, speed=5.0, K=0.3, T=5.0)
    print(f"Initial state: {vessel_nomoto.state}")
    
    # Simulate with rudder command
    print("\nSimulating 20 seconds with 30° rudder command...")
    rudder_cmd = np.radians(30)
    
    for i in range(200):  # 20 seconds at 0.1s timestep
        vessel_nomoto.update(dt, rudder_command=rudder_cmd)
    
    print(f"Final state: {vessel_nomoto.state}")
    print(f"Distance traveled: {np.sqrt(vessel_nomoto.state.x**2 + vessel_nomoto.state.y**2):.2f}")
    print(f"Turn rate: {np.degrees(vessel_nomoto.state.turn_rate):.2f}°/s")
    
    print("\n" + "=" * 70)
    print("Key Differences:")
    print("  - Kinematic: Instant response to commands")
    print("  - Nomoto: Gradual response with realistic dynamics")
    print("=" * 70)


if __name__ == "__main__":
    test_vessel_models()