"""
Collision Detection for Maritime Navigation

Implements:
- CPA (Closest Point of Approach) - minimum distance between vessels
- TCPA (Time to CPA) - when will minimum distance occur
- Collision risk assessment
- Relative motion calculations
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CollisionInfo:
    """
    Information about a potential collision.
    
    Attributes:
        vessel1_id: ID of first vessel
        vessel2_id: ID of second vessel
        current_distance: Current distance between vessels
        cpa_distance: Distance at closest point of approach
        tcpa: Time to closest point of approach (seconds)
        relative_bearing: Bearing of vessel2 from vessel1 (radians)
        is_collision_risk: Whether this is considered a collision risk
        risk_level: Risk level (0=none, 1=low, 2=medium, 3=high)
    """
    vessel1_id: int
    vessel2_id: int
    current_distance: float
    cpa_distance: float
    tcpa: float
    relative_bearing: float
    is_collision_risk: bool
    risk_level: int
    
    def __repr__(self):
        risk_names = ['NONE', 'LOW', 'MEDIUM', 'HIGH']
        return (f"CollisionInfo(V{self.vessel1_id}↔V{self.vessel2_id}: "
                f"CPA={self.cpa_distance:.1f} in {self.tcpa:.1f}s, "
                f"Risk={risk_names[self.risk_level]})")


class CollisionDetector:
    """
    Detects potential collisions between vessels using CPA/TCPA calculations.
    """
    
    def __init__(self, safe_distance: float = 8.0, 
                 warning_distance: float = 15.0,
                 time_horizon: float = 60.0):
        """
        Initialize collision detector.
        
        Args:
            safe_distance: Minimum safe distance between vessels
            warning_distance: Distance to start warning about collision
            time_horizon: Maximum time ahead to check (seconds)
        """
        self.safe_distance = safe_distance
        self.warning_distance = warning_distance
        self.time_horizon = time_horizon
    
    def calculate_cpa_tcpa(self,
                          pos1: Tuple[float, float],
                          vel1: Tuple[float, float],
                          pos2: Tuple[float, float],
                          vel2: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate Closest Point of Approach (CPA) and Time to CPA (TCPA).
        
        This is the fundamental collision detection calculation.
        
        Args:
            pos1: Position of vessel 1 (x, y)
            vel1: Velocity of vessel 1 (vx, vy)
            pos2: Position of vessel 2 (x, y)
            vel2: Velocity of vessel 2 (vx, vy)
            
        Returns:
            Tuple of (CPA distance, TCPA time)
        """
        # Relative position
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # Relative velocity
        dvx = vel2[0] - vel1[0]
        dvy = vel2[1] - vel1[1]
        
        # Current distance
        current_dist = np.sqrt(dx**2 + dy**2)
        
        # Relative speed
        relative_speed_sq = dvx**2 + dvy**2
        
        # If relative speed is very small, vessels are moving together
        if relative_speed_sq < 1e-6:
            # Not approaching - CPA is current distance
            return current_dist, float('inf')
        
        # TCPA calculation
        # Time when vessels are closest = when d(distance)/dt = 0
        # This occurs at t = -(relative_pos · relative_vel) / |relative_vel|²
        
        dot_product = dx * dvx + dy * dvy
        tcpa = -dot_product / relative_speed_sq
        
        # If TCPA is negative, vessels are moving apart
        if tcpa < 0:
            return current_dist, tcpa
        
        # Calculate CPA distance
        # Position at TCPA
        pos1_at_cpa = (pos1[0] + vel1[0] * tcpa, pos1[1] + vel1[1] * tcpa)
        pos2_at_cpa = (pos2[0] + vel2[0] * tcpa, pos2[1] + vel2[1] * tcpa)
        
        cpa_dx = pos2_at_cpa[0] - pos1_at_cpa[0]
        cpa_dy = pos2_at_cpa[1] - pos1_at_cpa[1]
        cpa_distance = np.sqrt(cpa_dx**2 + cpa_dy**2)
        
        return cpa_distance, tcpa
    
    def calculate_relative_bearing(self,
                                   pos1: Tuple[float, float],
                                   heading1: float,
                                   pos2: Tuple[float, float]) -> float:
        """
        Calculate relative bearing of vessel 2 from vessel 1.
        
        Args:
            pos1: Position of vessel 1
            heading1: Heading of vessel 1 (radians)
            pos2: Position of vessel 2
            
        Returns:
            Relative bearing in radians (-π to π)
            0 = dead ahead, π/2 = starboard, -π/2 = port
        """
        # Absolute bearing to vessel 2
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        absolute_bearing = np.arctan2(dy, dx)
        
        # Relative bearing
        relative_bearing = absolute_bearing - heading1
        
        # Normalize to [-π, π]
        relative_bearing = np.arctan2(np.sin(relative_bearing),
                                     np.cos(relative_bearing))
        
        return relative_bearing
    
    def assess_collision_risk(self,
                             pos1: Tuple[float, float],
                             heading1: float,
                             speed1: float,
                             pos2: Tuple[float, float],
                             heading2: float,
                             speed2: float,
                             vessel1_id: int = 0,
                             vessel2_id: int = 1) -> CollisionInfo:
        """
        Assess collision risk between two vessels.
        
        Args:
            pos1: Position of vessel 1
            heading1: Heading of vessel 1 (radians)
            speed1: Speed of vessel 1
            pos2: Position of vessel 2
            heading2: Heading of vessel 2 (radians)
            speed2: Speed of vessel 2
            vessel1_id: ID of vessel 1
            vessel2_id: ID of vessel 2
            
        Returns:
            CollisionInfo object with assessment
        """
        # Convert heading and speed to velocity vectors
        vel1 = (speed1 * np.cos(heading1), speed1 * np.sin(heading1))
        vel2 = (speed2 * np.cos(heading2), speed2 * np.sin(heading2))
        
        # Calculate current distance
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        current_distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate CPA and TCPA
        cpa_distance, tcpa = self.calculate_cpa_tcpa(pos1, vel1, pos2, vel2)
        
        # Calculate relative bearing
        relative_bearing = self.calculate_relative_bearing(pos1, heading1, pos2)
        
        # Assess risk
        is_collision_risk = False
        risk_level = 0  # 0=none, 1=low, 2=medium, 3=high
        
        # Only consider if TCPA is positive and within time horizon
        if 0 < tcpa < self.time_horizon:
            if cpa_distance < self.safe_distance:
                # High risk - will come very close
                is_collision_risk = True
                risk_level = 3
            elif cpa_distance < self.warning_distance:
                # Medium risk - within warning distance
                is_collision_risk = True
                if tcpa < 20.0:  # Less than 20 seconds
                    risk_level = 2
                else:
                    risk_level = 1
        
        # Also check current distance for immediate risk
        if current_distance < self.safe_distance:
            is_collision_risk = True
            risk_level = 3
        
        return CollisionInfo(
            vessel1_id=vessel1_id,
            vessel2_id=vessel2_id,
            current_distance=current_distance,
            cpa_distance=cpa_distance,
            tcpa=tcpa,
            relative_bearing=relative_bearing,
            is_collision_risk=is_collision_risk,
            risk_level=risk_level
        )
    
    def determine_encounter_type(self, relative_bearing: float,
                                 heading1: float, heading2: float) -> str:
        """
        Determine the type of encounter for COLREGs application.
        
        Args:
            relative_bearing: Relative bearing of vessel 2 from vessel 1
            heading1: Heading of vessel 1
            heading2: Heading of vessel 2
            
        Returns:
            Encounter type: 'head-on', 'crossing-starboard', 'crossing-port', 
                          'overtaking', or 'safe'
        """
        # Calculate relative heading
        relative_heading = heading2 - heading1
        relative_heading = np.arctan2(np.sin(relative_heading),
                                     np.cos(relative_heading))
        
        # Convert to degrees for easier logic
        rel_bearing_deg = np.degrees(relative_bearing)
        rel_heading_deg = np.degrees(relative_heading)
        
        # Head-on: vessels approaching nearly opposite directions
        # Relative bearing ±22.5° from dead ahead
        # Relative heading ±10° from opposite
        if abs(rel_bearing_deg) < 22.5 and abs(abs(rel_heading_deg) - 180) < 10:
            return 'head-on'
            
        # Crossing Ahead (in the dead-ahead sector but not head-on)
        if abs(rel_bearing_deg) < 22.5:
            # Determine direction of crossing based on relative heading
            # Normalize rel_heading_deg to [-180, 180]
            while rel_heading_deg > 180: rel_heading_deg -= 360
            while rel_heading_deg <= -180: rel_heading_deg += 360
            
            if rel_heading_deg > 0:
                return 'crossing-starboard' # Crossing from right to left
            else:
                return 'crossing-port' # Crossing from left to right
        
        # Overtaking: approaching from astern (within 22.5° of dead astern)
        # when vessel is moving faster
        if abs(abs(rel_bearing_deg) - 180) < 22.5:
            return 'overtaking'
        
        # Crossing from starboard (right side)
        # Relative bearing -112.5° to -22.5° (starboard side)
        if -112.5 < rel_bearing_deg < -22.5:
            return 'crossing-starboard'
        
        # Crossing from port (left side)
        # Relative bearing 22.5° to 112.5° (port side)
        if 22.5 < rel_bearing_deg < 112.5:
            return 'crossing-port'
        
        # Safe passing - well separated
        return 'safe'


def test_collision_detection():
    """Test collision detection algorithms."""
    print("=" * 70)
    print("COLLISION DETECTION TESTS")
    print("=" * 70)
    
    detector = CollisionDetector(safe_distance=8.0, warning_distance=15.0)
    
    # Test 1: Head-on collision course
    print("\n1. HEAD-ON COLLISION COURSE")
    print("-" * 70)
    print("Two vessels approaching directly")
    
    pos1 = (0, 0)
    heading1 = 0  # East
    speed1 = 2.0
    
    pos2 = (50, 0)
    heading2 = np.pi  # West
    speed2 = 2.0
    
    info = detector.assess_collision_risk(pos1, heading1, speed1,
                                         pos2, heading2, speed2,
                                         vessel1_id=1, vessel2_id=2)
    
    print(f"  {info}")
    print(f"  Encounter type: {detector.determine_encounter_type(info.relative_bearing, heading1, heading2)}")
    
    # Test 2: Crossing situation
    print("\n2. CROSSING SITUATION (STARBOARD)")
    print("-" * 70)
    print("Vessel 2 crossing from starboard")
    
    pos1 = (0, 0)
    heading1 = 0  # East
    speed1 = 2.0
    
    pos2 = (30, -20)
    heading2 = np.pi / 2  # North
    speed2 = 2.0
    
    info = detector.assess_collision_risk(pos1, heading1, speed1,
                                         pos2, heading2, speed2,
                                         vessel1_id=1, vessel2_id=2)
    
    print(f"  {info}")
    print(f"  Encounter type: {detector.determine_encounter_type(info.relative_bearing, heading1, heading2)}")
    print(f"  Relative bearing: {np.degrees(info.relative_bearing):.1f}°")
    
    # Test 3: Safe passing
    print("\n3. SAFE PASSING")
    print("-" * 70)
    print("Vessels passing with safe distance")
    
    pos1 = (0, 0)
    heading1 = 0  # East
    speed1 = 2.0
    
    pos2 = (20, 30)
    heading2 = 0  # Also East (parallel)
    speed2 = 2.0
    
    info = detector.assess_collision_risk(pos1, heading1, speed1,
                                         pos2, heading2, speed2,
                                         vessel1_id=1, vessel2_id=2)
    
    print(f"  {info}")
    print(f"  Encounter type: {detector.determine_encounter_type(info.relative_bearing, heading1, heading2)}")
    
    # Test 4: Overtaking
    print("\n4. OVERTAKING SITUATION")
    print("-" * 70)
    print("Fast vessel overtaking slower vessel")
    
    pos1 = (10, 0)
    heading1 = 0  # East
    speed1 = 1.5  # Slower
    
    pos2 = (0, 0)
    heading2 = 0  # East
    speed2 = 3.0  # Faster
    
    info = detector.assess_collision_risk(pos1, heading1, speed1,
                                         pos2, heading2, speed2,
                                         vessel1_id=1, vessel2_id=2)
    
    print(f"  {info}")
    print(f"  Encounter type: {detector.determine_encounter_type(info.relative_bearing, heading1, heading2)}")
    print(f"  Relative bearing: {np.degrees(info.relative_bearing):.1f}°")
    
    # Test 5: Time series prediction
    print("\n5. TIME SERIES PREDICTION")
    print("-" * 70)
    print("Tracking CPA over time as vessels approach")
    
    pos1 = (0, 0)
    heading1 = 0
    speed1 = 2.0
    
    pos2_start = (50, 5)
    heading2 = np.pi
    speed2 = 2.0
    
    print("\n  Time | Current Dist | CPA Dist | TCPA | Risk")
    print("  " + "-" * 55)
    
    for t in range(0, 15, 3):
        # Update position of vessel 2
        pos2 = (pos2_start[0] + speed2 * np.cos(heading2) * t,
                pos2_start[1] + speed2 * np.sin(heading2) * t)
        
        info = detector.assess_collision_risk(pos1, heading1, speed1,
                                             pos2, heading2, speed2)
        
        risk_names = ['NONE', 'LOW', 'MED', 'HIGH']
        print(f"  {t:4.0f}s | {info.current_distance:11.2f} | "
              f"{info.cpa_distance:8.2f} | {info.tcpa:4.1f} | "
              f"{risk_names[info.risk_level]}")
    
    print("\n" + "=" * 70)
    print("✓ Collision detection tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_collision_detection()