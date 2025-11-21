"""
Collision Avoidance for Autonomous Vessels

Implements avoidance maneuvers when collision risk is detected:
- Course alteration
- Speed reduction
- Combined maneuvers
- Basic COLREGs-inspired rules
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class AvoidanceAction:
    """
    Represents an avoidance action.
    
    Attributes:
        type: Type of action ('alter_course', 'reduce_speed', 'stop', 'resume')
        heading_change: Desired heading change in radians (0 if not course change)
        speed_factor: Speed multiplier (1.0 = normal, 0.5 = half speed, 0 = stop)
        reason: Reason for the action
        priority: Priority level (higher = more urgent)
    """
    type: str
    heading_change: float = 0.0
    speed_factor: float = 1.0
    reason: str = ""
    priority: int = 0
    
    def __repr__(self):
        return (f"AvoidanceAction({self.type}: "
                f"hdg_change={np.degrees(self.heading_change):.1f}°, "
                f"speed={self.speed_factor:.2f}, priority={self.priority})")


class CollisionAvoidance:
    """
    Implements collision avoidance strategies.
    """
    
    def __init__(self, 
                 safe_distance: float = 8.0,
                 warning_distance: float = 15.0,
                 avoidance_angle: float = np.radians(30)):
        """
        Initialize collision avoidance.
        
        Args:
            safe_distance: Minimum safe distance
            warning_distance: Distance to start avoiding
            avoidance_angle: Standard course alteration angle (radians)
        """
        self.safe_distance = safe_distance
        self.warning_distance = warning_distance
        self.avoidance_angle = avoidance_angle
        
        # State tracking
        self.active_avoidance = False
        self.avoided_vessels = set()  # Track which vessels we're avoiding
        self.original_speed = None
    
    def determine_avoidance_action(self,
                                   our_pos: Tuple[float, float],
                                   our_heading: float,
                                   our_speed: float,
                                   collision_info,
                                   encounter_type: str,
                                   obstacle_pos: Tuple[float, float],
                                   obstacle_heading: float) -> Optional[AvoidanceAction]:
        """
        Determine appropriate avoidance action based on collision risk.
        
        Args:
            our_pos: Our position
            our_heading: Our heading
            our_speed: Our speed
            collision_info: CollisionInfo object
            encounter_type: Type of encounter (head-on, crossing, etc.)
            obstacle_pos: Obstacle position
            obstacle_heading: Obstacle heading
            
        Returns:
            AvoidanceAction or None if no action needed
        """
        # No action if no collision risk
        if not collision_info.is_collision_risk:
            return None
        
        # Get relative bearing
        rel_bearing = collision_info.relative_bearing
        
        # Choose action based on encounter type and risk level
        if collision_info.risk_level == 3:  # HIGH RISK
            return self._high_risk_action(encounter_type, rel_bearing, 
                                         collision_info, our_heading, obstacle_heading)
        
        elif collision_info.risk_level == 2:  # MEDIUM RISK
            return self._medium_risk_action(encounter_type, rel_bearing,
                                          collision_info)
        
        elif collision_info.risk_level == 1:  # LOW RISK
            return self._low_risk_action(encounter_type, rel_bearing)
        
        return None
    
    def _high_risk_action(self, encounter_type: str, rel_bearing: float,
                         collision_info, our_heading: float, 
                         obstacle_heading: float) -> AvoidanceAction:
        """Determine action for high-risk collision."""
        
        if encounter_type == 'head-on':
            # COLREGs Rule 14: Both vessels alter course to starboard
            # We turn right (starboard)
            return AvoidanceAction(
                type='alter_course',
                heading_change=self.avoidance_angle,  # Turn right
                speed_factor=0.7,  # Reduce speed slightly
                reason=f'Head-on collision risk (CPA={collision_info.cpa_distance:.1f})',
                priority=3
            )
        
        elif encounter_type == 'crossing-starboard':
            # COLREGs Rule 15: We are give-way vessel, other has right of way
            # Take early and substantial action
            # Turn right to pass behind the other vessel
            return AvoidanceAction(
                type='alter_course',
                heading_change=self.avoidance_angle * 1.5,  # Large turn right
                speed_factor=0.6,  # Reduce speed
                reason=f'Crossing from starboard - give way (CPA={collision_info.cpa_distance:.1f})',
                priority=3
            )
        
        elif encounter_type == 'crossing-port':
            # COLREGs Rule 15: We are stand-on vessel
            # Maintain course and speed, but if collision imminent, act
            if collision_info.tcpa < 10.0:  # Very close
                # Emergency action: turn right
                return AvoidanceAction(
                    type='alter_course',
                    heading_change=self.avoidance_angle,
                    speed_factor=0.5,
                    reason=f'Crossing from port - emergency action (CPA={collision_info.cpa_distance:.1f})',
                    priority=3
                )
            else:
                # Maintain course (stand-on vessel)
                return AvoidanceAction(
                    type='maintain',
                    speed_factor=1.0,
                    reason='Crossing from port - stand-on vessel',
                    priority=2
                )
        
        elif encounter_type == 'overtaking':
            # We're being overtaken - maintain course and speed
            return AvoidanceAction(
                type='maintain',
                speed_factor=1.0,
                reason='Being overtaken - maintain course',
                priority=1
            )
        
        else:
            # Generic avoidance: turn away from obstacle
            # Determine which way to turn based on relative bearing
            if -np.pi/2 < rel_bearing < np.pi/2:
                # Obstacle ahead - turn right if on right, left if on left
                turn_direction = 1 if rel_bearing > 0 else -1
            else:
                # Obstacle behind - minimal turn
                turn_direction = 1 if rel_bearing > 0 else -1
            
            return AvoidanceAction(
                type='alter_course',
                heading_change=self.avoidance_angle * turn_direction,
                speed_factor=0.7,
                reason=f'Generic avoidance (CPA={collision_info.cpa_distance:.1f})',
                priority=3
            )
    
    def _medium_risk_action(self, encounter_type: str, rel_bearing: float,
                           collision_info) -> AvoidanceAction:
        """Determine action for medium-risk collision."""
        
        if encounter_type == 'crossing-starboard':
            # Give way with moderate action
            return AvoidanceAction(
                type='alter_course',
                heading_change=self.avoidance_angle,
                speed_factor=0.8,
                reason=f'Crossing from starboard - moderate action',
                priority=2
            )
        
        else:
            # Slight course alteration
            turn_direction = 1 if rel_bearing > 0 else -1
            return AvoidanceAction(
                type='alter_course',
                heading_change=self.avoidance_angle * 0.7 * turn_direction,
                speed_factor=0.9,
                reason=f'Moderate risk - slight course change',
                priority=2
            )
    
    def _low_risk_action(self, encounter_type: str, rel_bearing: float) -> AvoidanceAction:
        """Determine action for low-risk collision."""
        
        # Just reduce speed slightly as precaution
        return AvoidanceAction(
            type='reduce_speed',
            speed_factor=0.9,
            reason='Low risk - precautionary speed reduction',
            priority=1
        )
    
    def apply_avoidance(self,
                       desired_heading: float,
                       desired_speed: float,
                       avoidance_actions: List[AvoidanceAction]) -> Tuple[float, float]:
        """
        Apply avoidance actions to modify desired heading and speed.
        
        Args:
            desired_heading: Original desired heading from path follower
            desired_speed: Original desired speed
            avoidance_actions: List of avoidance actions from all obstacles
            
        Returns:
            Tuple of (modified_heading, modified_speed)
        """
        if not avoidance_actions:
            self.active_avoidance = False
            return desired_heading, desired_speed
        
        self.active_avoidance = True
        
        # Sort by priority (highest first)
        avoidance_actions.sort(key=lambda x: x.priority, reverse=True)
        
        # Take the highest priority action
        primary_action = avoidance_actions[0]
        
        # Apply heading change
        modified_heading = desired_heading
        if primary_action.type in ['alter_course', 'emergency']:
            modified_heading = desired_heading + primary_action.heading_change
            # Normalize to [-π, π]
            modified_heading = np.arctan2(np.sin(modified_heading),
                                         np.cos(modified_heading))
        
        # Apply speed change
        # Take minimum speed factor from all high-priority actions
        speed_factors = [action.speed_factor for action in avoidance_actions 
                        if action.priority >= primary_action.priority - 1]
        modified_speed = desired_speed * min(speed_factors)
        
        return modified_heading, modified_speed
    
    def should_resume_path(self,
                          collision_infos: List,
                          current_distance_to_path: float) -> bool:
        """
        Determine if it's safe to resume following the original path.
        
        Args:
            collision_infos: List of current collision info
            current_distance_to_path: How far we are from original path
            
        Returns:
            True if safe to resume path following
        """
        # Check if all collision risks are clear
        any_risk = any(info['info'].is_collision_risk for info in collision_infos)
        
        if not any_risk:
            # All clear, but don't resume if we're far from path
            if current_distance_to_path < self.warning_distance:
                return True
        
        return False


def test_collision_avoidance():
    """Test collision avoidance logic."""
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.environment.collision_detection import CollisionDetector, CollisionInfo
    
    print("=" * 70)
    print("COLLISION AVOIDANCE TESTS")
    print("=" * 70)
    
    avoider = CollisionAvoidance(
        safe_distance=8.0,
        warning_distance=15.0,
        avoidance_angle=np.radians(30)
    )
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Head-on collision',
            'encounter': 'head-on',
            'rel_bearing': 0,
            'risk_level': 3,
            'cpa': 3.0,
            'tcpa': 15.0
        },
        {
            'name': 'Crossing from starboard',
            'encounter': 'crossing-starboard',
            'rel_bearing': np.radians(45),
            'risk_level': 3,
            'cpa': 5.0,
            'tcpa': 20.0
        },
        {
            'name': 'Crossing from port',
            'encounter': 'crossing-port',
            'rel_bearing': np.radians(-45),
            'risk_level': 3,
            'cpa': 5.0,
            'tcpa': 20.0
        },
        {
            'name': 'Medium risk - starboard',
            'encounter': 'crossing-starboard',
            'rel_bearing': np.radians(60),
            'risk_level': 2,
            'cpa': 12.0,
            'tcpa': 30.0
        },
        {
            'name': 'Low risk',
            'encounter': 'safe',
            'rel_bearing': np.radians(90),
            'risk_level': 1,
            'cpa': 18.0,
            'tcpa': 40.0
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("-" * 70)
        
        # Create mock collision info
        collision_info = CollisionInfo(
            vessel1_id=0,
            vessel2_id=1,
            current_distance=20.0,
            cpa_distance=scenario['cpa'],
            tcpa=scenario['tcpa'],
            relative_bearing=scenario['rel_bearing'],
            is_collision_risk=(scenario['risk_level'] > 0),
            risk_level=scenario['risk_level']
        )
        
        # Determine avoidance action
        action = avoider.determine_avoidance_action(
            our_pos=(0, 0),
            our_heading=0,
            our_speed=2.0,
            collision_info=collision_info,
            encounter_type=scenario['encounter'],
            obstacle_pos=(20, 0),
            obstacle_heading=np.pi
        )
        
        if action:
            print(f"  Action: {action.type}")
            print(f"  Heading change: {np.degrees(action.heading_change):.1f}°")
            print(f"  Speed factor: {action.speed_factor:.2f}")
            print(f"  Reason: {action.reason}")
            print(f"  Priority: {action.priority}")
        else:
            print("  No action required")
    
    # Test applying multiple actions
    print("\n" + "=" * 70)
    print("MULTIPLE OBSTACLES TEST")
    print("-" * 70)
    
    actions = [
        AvoidanceAction('alter_course', heading_change=np.radians(30), 
                       speed_factor=0.7, priority=3),
        AvoidanceAction('reduce_speed', speed_factor=0.8, priority=2),
        AvoidanceAction('alter_course', heading_change=np.radians(-20),
                       speed_factor=0.9, priority=2)
    ]
    
    original_heading = 0
    original_speed = 2.0
    
    print(f"Original: heading={np.degrees(original_heading):.1f}°, speed={original_speed:.1f}")
    print(f"\nActions to apply:")
    for action in actions:
        print(f"  {action}")
    
    modified_heading, modified_speed = avoider.apply_avoidance(
        original_heading, original_speed, actions
    )
    
    print(f"\nModified: heading={np.degrees(modified_heading):.1f}°, speed={modified_speed:.1f}")
    print(f"  Heading change: {np.degrees(modified_heading - original_heading):.1f}°")
    print(f"  Speed change: {(modified_speed/original_speed - 1)*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("✓ Collision avoidance tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_collision_avoidance()