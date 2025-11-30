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
    
    Rudder Dynamics:
        Real rudders have limited slew rates. Per IMO standards, rudders must
        move from 35° to -35° (70° total) in approximately 11 seconds.
        This gives a maximum rudder rate of ~6.36°/s (0.111 rad/s).
        
        The model tracks both commanded and actual rudder angles, with the
        actual rudder moving toward the commanded position at the rate limit.
    """
    
    # IMO standard: 70° (35° to -35°) in 11 seconds = 6.36°/s
    IMO_RUDDER_RATE = np.radians(70.0 / 11.0)  # ~0.111 rad/s
    
    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0, 
                 heading: float = 0.0,
                 speed: float = 5.0,
                 max_speed: float = 10.0,
                 K: float = 0.3,
                 T: float = 5.0,
                 max_rudder: float = np.radians(35.0),  # 35° per IMO standard
                 rudder_rate: Optional[float] = None):
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
            rudder_rate: Maximum rudder rate in rad/s. Controls how fast the
                        rudder can physically move.
                        - None: Instant rudder response (legacy behavior)
                        - 0 or negative: Use IMO standard (~0.111 rad/s, 6.36°/s)
                        - Positive value: Use specified rate
                        IMO standard: 70° in 11s for 35° to -35° movement.
        """
        self.state = VesselState(x, y, heading, speed, turn_rate=0.0)
        self.max_speed = max_speed
        self.K = K
        self.T = T
        self.max_rudder = max_rudder
        
        # Rudder dynamics
        if rudder_rate is None:
            self.rudder_rate = None  # Instant rudder (backward compatible)
        elif rudder_rate <= 0:
            self.rudder_rate = self.IMO_RUDDER_RATE  # Use IMO standard
        else:
            self.rudder_rate = rudder_rate  # Use specified rate
        
        # Track commanded vs actual rudder position
        self.rudder_command = 0.0   # Commanded/desired rudder angle
        self.rudder_angle = 0.0     # Actual physical rudder angle
        
        print(f"Created NomotoVessel at ({x}, {y})")
        print(f"  Max speed: {max_speed} units/s")
        print(f"  K (gain): {K}")
        print(f"  T (time constant): {T}s")
        print(f"  Max rudder: {np.degrees(max_rudder):.1f}°")
        if self.rudder_rate is not None:
            print(f"  Rudder rate: {np.degrees(self.rudder_rate):.2f}°/s")
        else:
            print(f"  Rudder rate: Instant (no delay)")
    
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
        
        # Limit and store rudder command
        self.rudder_command = np.clip(rudder_command, -self.max_rudder, self.max_rudder)
        
        # Apply rudder rate limiting
        if self.rudder_rate is not None:
            # Calculate maximum rudder change this timestep
            max_delta = self.rudder_rate * dt
            # Move actual rudder toward commanded position
            rudder_error = self.rudder_command - self.rudder_angle
            delta = np.clip(rudder_error, -max_delta, max_delta)
            self.rudder_angle += delta
        else:
            # Instant rudder response (legacy behavior)
            self.rudder_angle = self.rudder_command
        
        # Nomoto first-order model: T * dψ/dt + ψ = K * δ
        # Rearranged: dψ/dt = (K * δ - ψ) / T
        # Uses actual rudder angle (not commanded) for realistic dynamics
        d_turn_rate = (self.K * self.rudder_angle - self.state.turn_rate) / self.T
        
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
    
    def get_rudder_angle(self) -> float:
        """Get current actual rudder angle in radians."""
        return self.rudder_angle
    
    def get_rudder_command(self) -> float:
        """Get current commanded rudder angle in radians."""
        return self.rudder_command
    
    def get_rudder_error(self) -> float:
        """Get difference between commanded and actual rudder (rad)."""
        return self.rudder_command - self.rudder_angle
    
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
    
    # Test 2: Nomoto vessel (instant rudder - legacy)
    print("\n" + "=" * 70)
    print("2. Testing Nomoto Vessel (Instant Rudder)")
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
    
    # Test 3: Nomoto vessel with realistic rudder rate
    print("\n" + "=" * 70)
    print("3. Testing Nomoto Vessel (IMO Rudder Rate)")
    print("-" * 70)
    
    # Use rudder_rate=0 to get IMO standard rate
    vessel_realistic = NomotoVessel(x=0, y=0, heading=0, speed=5.0, K=0.3, T=5.0, rudder_rate=0)
    print(f"Initial state: {vessel_realistic.state}")
    
    # Simulate with hard-over rudder command
    print("\nSimulating 20 seconds with 35° (hard over) rudder command...")
    rudder_cmd = np.radians(35)  # Hard starboard
    
    print("\nRudder movement over time:")
    for i in range(200):  # 20 seconds at 0.1s timestep
        vessel_realistic.update(dt, rudder_command=rudder_cmd)
        # Print rudder position at key intervals
        if i in [0, 10, 20, 30, 40, 50, 100]:
            print(f"  t={i*dt:5.1f}s: Commanded={np.degrees(vessel_realistic.get_rudder_command()):5.1f}°, "
                  f"Actual={np.degrees(vessel_realistic.get_rudder_angle()):5.1f}°, "
                  f"Error={np.degrees(vessel_realistic.get_rudder_error()):5.1f}°")
    
    print(f"\nFinal state: {vessel_realistic.state}")
    print(f"Distance traveled: {np.sqrt(vessel_realistic.state.x**2 + vessel_realistic.state.y**2):.2f}")
    print(f"Turn rate: {np.degrees(vessel_realistic.state.turn_rate):.2f}°/s")
    print(f"Rudder: Commanded={np.degrees(vessel_realistic.get_rudder_command()):.1f}°, "
          f"Actual={np.degrees(vessel_realistic.get_rudder_angle()):.1f}°")
    
    # Test 4: Demonstrate rudder slew time
    print("\n" + "=" * 70)
    print("4. Rudder Slew Test (35° to -35°)")
    print("-" * 70)
    
    vessel_slew = NomotoVessel(x=0, y=0, heading=0, speed=5.0, rudder_rate=0)
    max_rudder_deg = np.degrees(vessel_slew.max_rudder)
    
    # Start at 35° starboard
    for _ in range(150):  # Let rudder reach max starboard
        vessel_slew.update(dt, rudder_command=np.radians(35))
    
    start_angle = vessel_slew.get_rudder_angle()
    print(f"Starting position: {np.degrees(start_angle):.1f}°")
    print(f"Target: -{max_rudder_deg:.1f}° (hard port)")
    
    # Command hard port and measure time
    target_angle = -vessel_slew.max_rudder
    steps = 0
    while vessel_slew.get_rudder_angle() > target_angle + np.radians(0.1):  # Within 0.1° of target
        vessel_slew.update(dt, rudder_command=np.radians(-35))
        steps += 1
        if steps > 200:  # Safety limit
            break
    
    slew_time = steps * dt
    total_travel = np.degrees(start_angle - vessel_slew.get_rudder_angle())
    print(f"Final position: {np.degrees(vessel_slew.get_rudder_angle()):.1f}°")
    print(f"Total travel: {total_travel:.1f}°")
    print(f"Time to complete: {slew_time:.1f} seconds")
    print(f"Expected (IMO): ~{70.0 / np.degrees(vessel_slew.rudder_rate):.1f} seconds for 70°")
    
    print("\n" + "=" * 70)
    print("Key Differences:")
    print("  - Kinematic: Instant response to commands")
    print("  - Nomoto (instant rudder): Gradual turn response, instant rudder")
    print("  - Nomoto (IMO rate): Realistic rudder movement + turn dynamics")
    print("=" * 70)


if __name__ == "__main__":
    test_vessel_models()