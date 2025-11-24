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


@dataclass
class CandidateManeuver:
    """Represents a candidate avoidance maneuver to be evaluated."""
    heading_change: float
    speed_factor: float
    score: float = -float('inf')
    is_safe: bool = False
    reason: str = ""
    trajectory: List[Dict] = None  # Store simulated trajectory for scoring
    
    def __repr__(self):
        return (f"Maneuver(hdg={np.degrees(self.heading_change):.0f}°, "
                f"spd={self.speed_factor:.1f}, safe={self.is_safe}, score={self.score:.1f})")


class CollisionAvoidance:
    """
    Implements collision avoidance strategies.
    """
    
    def __init__(self, 
                 safe_distance: float = 8.0,
                 warning_distance: float = 15.0,
                 avoidance_angle: float = np.radians(30),
                 grid_world=None):
        """
        Initialize collision avoidance.
        
        Args:
            safe_distance: Minimum safe distance
            warning_distance: Distance to start avoiding
            avoidance_angle: Standard course alteration angle (radians)
            grid_world: Optional GridWorld object for static obstacle checking
        """
        self.safe_distance = safe_distance
        self.warning_distance = warning_distance
        self.avoidance_angle = avoidance_angle
        self.grid_world = grid_world
        
        # State tracking
        self.active_avoidance = False
        self.avoided_vessels = set()  # Track which vessels we're avoiding
        self.original_speed = None
        
        # Define candidate maneuvers for predictive search
        self.candidates = self._generate_candidates()

    def _generate_candidates(self) -> List[CandidateManeuver]:
        """Generate a set of candidate maneuvers to evaluate."""
        candidates = []
        
        # 1. Maintain course (baseline)
        candidates.append(CandidateManeuver(0.0, 1.0, reason="Maintain course"))
        
        # 2. Speed adjustments
        candidates.append(CandidateManeuver(0.0, 0.5, reason="Slow down"))
        candidates.append(CandidateManeuver(0.0, 0.0, reason="Stop"))
        candidates.append(CandidateManeuver(0.0, -0.5, reason="Reverse"))
        
        # 3. Course alterations (Starboard/Right - preferred by COLREGs)
        for deg in [15, 30, 45, 60, 90]:
            candidates.append(CandidateManeuver(np.radians(-deg), 1.0, reason=f"Turn Stbd {deg}°"))
            candidates.append(CandidateManeuver(np.radians(-deg), 0.5, reason=f"Turn Stbd {deg}° + Slow"))
            
        # 4. Course alterations (Port/Left - less preferred)
        for deg in [15, 30, 45, 60, 90]:
            candidates.append(CandidateManeuver(np.radians(deg), 1.0, reason=f"Turn Port {deg}°"))
            candidates.append(CandidateManeuver(np.radians(deg), 0.5, reason=f"Turn Port {deg}° + Slow"))
            
        return candidates

    def _is_maneuver_safe(self, start_pos: Tuple[float, float], 
                         heading: float, speed: float, 
                         duration: float = 10.0) -> bool:
        """
        Check if a maneuver is safe from static obstacles.
        
        Args:
            start_pos: Starting position (x, y)
            heading: Heading in radians
            speed: Speed in units/s
            duration: Time horizon to check
            
        Returns:
            True if safe, False if collision with static obstacle
        """
        if self.grid_world is None:
            return True
            
        # Project position
        dx = speed * np.cos(heading) * duration
        dy = speed * np.sin(heading) * duration
        
        end_x = start_pos[0] + dx
        end_y = start_pos[1] + dy
        
        # Check discrete points along the path
        steps = int(max(abs(dx), abs(dy))) + 1
        for i in range(steps + 1):
            t = i / steps
            x = int(start_pos[0] + dx * t)
            y = int(start_pos[1] + dy * t)
            
            # Check bounds
            if (x < 0 or x >= self.grid_world.width or 
                y < 0 or y >= self.grid_world.height):
                return False
                
            # Check obstacle (using raw grid, not inflated, for now)
            # Ideally should use inflated grid if available
            if self.grid_world.grid[y, x] > 0.5:
                return False
                
        return True
    
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
        Simple rule-based approach following COLREGs principles.
        
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
        
        # Determine priority based on risk level
        priority = collision_info.risk_level
        
        # Determine turn direction and magnitude based on encounter type
        if encounter_type == 'head-on':
            # Rule 14: Both vessels turn starboard
            turn_angle = -self.avoidance_angle  # Negative = starboard/right
            speed_factor = 0.7 if collision_info.risk_level >= 3 else 0.85
            reason = f"Head-on: Turn starboard (CPA={collision_info.cpa_distance:.1f}u)"
            
        elif encounter_type == 'crossing-starboard':
            # Rule 15: Give way - turn starboard to pass behind
            turn_angle = -self.avoidance_angle * 1.5
            speed_factor = 0.6 if collision_info.risk_level >= 3 else 0.75
            reason = f"Crossing-starboard: Give way (CPA={collision_info.cpa_distance:.1f}u)"
            
        elif encounter_type == 'crossing-port':
            # Rule 15: Stand-on vessel, but take action if danger persists
            if collision_info.risk_level >= 3 or collision_info.tcpa < 10.0:
                # Emergency: must take action
                turn_angle = -np.radians(45)  # Larger starboard turn
                speed_factor = 0.5
                reason = f"Crossing-port: Emergency action (CPA={collision_info.cpa_distance:.1f}u)"
            else:
                # Maintain course but slow down
                turn_angle = 0.0
                speed_factor = 0.8
                reason = f"Crossing-port: Stand-on, reduce speed"
                
        elif encounter_type == 'overtaking':
            # Rule 13: Overtaking vessel keeps clear
            # Turn to pass on safe side
            if collision_info.relative_bearing < 0:
                turn_angle = self.avoidance_angle  # Pass on port side
            else:
                turn_angle = -self.avoidance_angle  # Pass on starboard side
            speed_factor = 0.9
            reason = f"Overtaking: Keep clear (CPA={collision_info.cpa_distance:.1f}u)"
            
        else:  # Unknown or safe passing
            turn_angle = -self.avoidance_angle if collision_info.relative_bearing > 0 else self.avoidance_angle
            speed_factor = 0.8
            reason = f"Unknown encounter: Precautionary action (CPA={collision_info.cpa_distance:.1f}u)"
        
        # Check if maneuver would hit land
        new_heading = our_heading + turn_angle
        new_speed = our_speed * speed_factor
        
        if not self._is_maneuver_safe(our_pos, new_heading, new_speed, duration=10.0):
            # Try opposite turn (safety override)
            alt_turn = -turn_angle
            alt_heading = our_heading + alt_turn
            
            if self._is_maneuver_safe(our_pos, alt_heading, new_speed * 0.7, duration=8.0):
                return AvoidanceAction(
                    'emergency',
                    alt_turn,
                    speed_factor * 0.7,
                    f"SAFETY OVERRIDE: {reason} (primary turn blocked by land)",
                    priority + 1
                )
            else:
                # Both turns blocked - stop
                return AvoidanceAction(
                    'emergency',
                    0.0,
                    0.0,
                    f"EMERGENCY STOP: {reason} (all turns blocked)",
                    priority + 2
                )
        
        return AvoidanceAction(
            'alter_course',
            turn_angle,
            speed_factor,
            reason,
            priority
        )
    
    def _high_risk_action(self, encounter_type: str, rel_bearing: float,
                         collision_info, our_heading: float, 
                         obstacle_heading: float,
                         our_pos: Tuple[float, float],
                         our_speed: float) -> AvoidanceAction:
        """Determine action for high-risk collision with static safety check."""
        
        # Determine base avoidance angle
        # If CPA is very small (< 3.0), use larger angle (Emergency Turn)
        base_angle = self.avoidance_angle
        if collision_info.cpa_distance < 3.0:
            base_angle = np.radians(60) # Aggressive turn
        
        # Helper to create action and check safety with alternatives
        def create_safe_action(type_str, hdg_change, spd_factor, reason_str, prio, allow_opposite=True):
            # 1. Try Primary Action
            new_heading = our_heading + hdg_change
            new_speed = our_speed * spd_factor
            
            if self._is_maneuver_safe(our_pos, new_heading, new_speed):
                return AvoidanceAction(type_str, hdg_change, spd_factor, reason_str, prio)
            
            # 2. Try Primary Action with Reduced Speed
            # Slower speed often allows for safer maneuvering in tight spaces
            safe_speed = max(our_speed * 0.5, 0.5) # Ensure we don't stop completely
            if self._is_maneuver_safe(our_pos, new_heading, safe_speed):
                return AvoidanceAction(type_str, hdg_change, 0.5, 
                                     f"{reason_str} (Reduced speed for static safety)", prio)

            # 3. Try Harder Turn (if not already max)
            # If standard turn is blocked, a sharper turn might clear the obstacle
            if abs(hdg_change) < np.radians(80):
                hard_turn = np.sign(hdg_change) * np.radians(90) # 90 degree turn
                if self._is_maneuver_safe(our_pos, our_heading + hard_turn, safe_speed):
                    return AvoidanceAction(type_str, hard_turn, 0.5,
                                         f"{reason_str} (Hard turn for static safety)", prio)

            # 4. Try Opposite Turn (Departure from COLREGs)
            # Only if allowed (e.g. not turning INTO a crossing vessel)
            if allow_opposite:
                reverse_turn = -hdg_change
                if self._is_maneuver_safe(our_pos, our_heading + reverse_turn, safe_speed):
                    return AvoidanceAction('emergency', reverse_turn, 0.5,
                                         f"Departure from rules: {reason_str} blocked by land", prio + 1)
                
                # 5. Try Opposite Hard Turn
                hard_reverse = np.sign(reverse_turn) * np.radians(90)
                if self._is_maneuver_safe(our_pos, our_heading + hard_reverse, safe_speed):
                    return AvoidanceAction('emergency', hard_reverse, 0.5,
                                         f"Departure from rules: Hard turn to avoid grounding", prio + 1)

            # 6. Last Resort: Reverse Engines
            # If all turns are blocked by land, we must attempt to stop/reverse
            return AvoidanceAction(
                'emergency', 
                0.0, 
                -0.5,  # Reverse engines
                f"MANEUVER RESTRICTED! Reversing. (Orig: {reason_str})", 
                prio + 2
            )

        if encounter_type == 'head-on':
            # COLREGs Rule 14: Turn starboard (negative angle)
            # Opposite turn (Port) is risky but better than grounding, usually allowed if Starboard is blocked
            return create_safe_action(
                'alter_course',
                -base_angle,
                0.7,
                f'Head-on collision risk (CPA={collision_info.cpa_distance:.1f})',
                3,
                allow_opposite=True 
            )
        
        elif encounter_type == 'crossing-starboard':
            # COLREGs Rule 15: Give way (turn right/starboard)
            # Opposite turn (Port) means turning BEHIND the vessel (if we are fast) or PARALLEL.
            # Turning Port for a vessel on Starboard is generally "turning away" from the collision point if we are crossing.
            # It is safer than turning Port for a vessel on Port.
            return create_safe_action(
                'alter_course',
                -base_angle * 1.5,
                0.6,
                f'Crossing from starboard - give way (CPA={collision_info.cpa_distance:.1f})',
                3,
                allow_opposite=True
            )
        
        elif encounter_type == 'crossing-port':
            # Stand-on vessel
            if collision_info.tcpa < 10.0 or collision_info.cpa_distance < 3.0:
                # Emergency action: turn right (starboard)
                # Opposite turn (Port) means turning INTO the vessel. DANGEROUS.
                # We should NOT allow opposite turn here. Better to Reverse.
                # Use larger angle (60 deg) to facilitate passing behind
                return create_safe_action(
                    'alter_course',
                    -np.radians(60),
                    0.5,
                    f'Crossing from port - emergency action (CPA={collision_info.cpa_distance:.1f})',
                    3,
                    allow_opposite=False # Disable turning Port into the vessel
                )
            else:
                return AvoidanceAction(
                    type='maintain',
                    speed_factor=1.0,
                    reason='Crossing from port - stand-on vessel',
                    priority=2
                )
        
        elif encounter_type == 'overtaking':
            return AvoidanceAction(
                type='maintain',
                speed_factor=1.0,
                reason='Being overtaken - maintain course',
                priority=1
            )
        
        else:
            # Generic avoidance
            # Default to Starboard turn (Right)
            return create_safe_action(
                'alter_course',
                -base_angle,
                0.7,
                f'Generic avoidance (CPA={collision_info.cpa_distance:.1f})',
                3
            )
    
    def _medium_risk_action(self, encounter_type: str, rel_bearing: float,
                           collision_info) -> AvoidanceAction:
        """Determine action for medium-risk collision."""
        
        if encounter_type == 'crossing-starboard':
            # Give way with moderate action (turn right)
            return AvoidanceAction(
                type='alter_course',
                heading_change=-self.avoidance_angle,
                speed_factor=0.8,
                reason=f'Crossing from starboard - moderate action',
                priority=2
            )
        
        else:
            # Slight course alteration
            # If obstacle is on Left (rel_bearing > 0), turn Right (-1)
            turn_direction = -1 if rel_bearing > 0 else 1
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
        
        # Check for conflicting high-priority actions (e.g. Land vs Ship)
        # If we have multiple high priority actions (>=3) with opposing heading changes
        high_prio_actions = [a for a in avoidance_actions if a.priority >= 3]
        if len(high_prio_actions) > 1:
            turns = [a.heading_change for a in high_prio_actions if abs(a.heading_change) > 0.1]
            if any(t > 0 for t in turns) and any(t < 0 for t in turns):
                # Conflict detected! (e.g. Turn Left for Land, Turn Right for Ship)
                # Safest action is to STOP/REVERSE and maintain heading (or turn to avoid Land if critical)
                
                # If one is Priority 4 (Critical Land), we must respect it, but maybe stop too
                critical_land = next((a for a in high_prio_actions if a.priority >= 4), None)
                
                if critical_land:
                    # We must turn to avoid land, but also stop to avoid ship
                    primary_action = critical_land
                    primary_action.speed_factor = -0.5 # Force reverse
                    primary_action.reason += " + CONFLICT! Reversing"
                else:
                    # Just conflicting dynamic/static risks
                    primary_action = AvoidanceAction(
                        'emergency', 0.0, -0.5, "CONFLICTING AVOIDANCE! Reversing", 5
                    )
                    # Insert at beginning so it's logged as primary
                    avoidance_actions.insert(0, primary_action)

        # Apply heading change
        modified_heading = desired_heading
        if primary_action.type in ['alter_course', 'emergency']:
            modified_heading = desired_heading + primary_action.heading_change
            # Normalize to [-π, π]
            modified_heading = np.arctan2(np.sin(modified_heading),
                                         np.cos(modified_heading))
        
        # Apply speed change
        # Take minimum speed factor from all high-priority actions
        # If we are reversing (negative speed factor), ensure we use that
        speed_factors = [action.speed_factor for action in avoidance_actions 
                        if action.priority >= primary_action.priority - 1]
        
        min_speed_factor = min(speed_factors)
        
        # If any action requires reverse, we reverse
        if min_speed_factor < 0:
             modified_speed = desired_speed * min_speed_factor # This might be positive if desired_speed is negative? No, desired_speed is usually positive.
             # If desired_speed is positive, modified_speed becomes negative (Reverse)
        else:
             modified_speed = desired_speed * min_speed_factor
        
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

    def _cast_ray(self, start_pos: Tuple[float, float], 
                  heading: float, max_dist: float) -> Optional[float]:
        """
        Cast a ray and return distance to first static obstacle.
        
        Args:
            start_pos: Starting position (x, y)
            heading: Ray heading in radians
            max_dist: Maximum distance to check
            
        Returns:
            Distance to obstacle or None if clear
        """
        if self.grid_world is None:
            return None
            
        dx = max_dist * np.cos(heading)
        dy = max_dist * np.sin(heading)
        
        # Number of steps (check every 0.5 units for better precision)
        steps = int(max_dist * 2)
        if steps == 0:
            return None
            
        for i in range(1, steps + 1):
            t = i / steps
            # Current point on ray
            px = start_pos[0] + dx * t
            py = start_pos[1] + dy * t
            
            # Grid coordinates
            gx, gy = int(px), int(py)
            
            # Check bounds
            if (gx < 0 or gx >= self.grid_world.width or 
                gy < 0 or gy >= self.grid_world.height):
                return np.sqrt((px - start_pos[0])**2 + (py - start_pos[1])**2)
                
            # Check obstacle
            if self.grid_world.grid[gy, gx] > 0.5:
                return np.sqrt((px - start_pos[0])**2 + (py - start_pos[1])**2)
                
        return None

    def check_static_obstacles(self, 
                             pos: Tuple[float, float], 
                             heading: float, 
                             speed: float) -> Optional[AvoidanceAction]:
        """
        Check for imminent collision with static obstacles (land).
        
        Args:
            pos: Current position
            heading: Current heading
            speed: Current speed
            
        Returns:
            AvoidanceAction if land is detected, else None
        """
        if self.grid_world is None:
            return None
            
        # Lookahead based on speed, but at least 20m
        lookahead_dist = max(speed * 15.0, 20.0)
        
        # Cast rays: Center, Port (-20°), Starboard (+20°)
        # Using wider whiskers to detect land masses earlier
        angles = [0, np.radians(20), -np.radians(20)]
        distances = []
        
        for angle in angles:
            dist = self._cast_ray(pos, heading + angle, lookahead_dist)
            distances.append(dist if dist is not None else float('inf'))
            
        center_dist, left_dist, right_dist = distances # Note: +20 is Left (Port) in standard math if heading is 0=East? 
        # Wait, standard math: 0=East, +90=North.
        # If heading is East (0), +20 is North-East (Left/Port).
        # So +20 is Port, -20 is Starboard.
        
        min_dist = min(distances)
        
        # If everything is clear, return None
        if min_dist == float('inf'):
            return None
            
        # Determine urgency
        priority = 3 # High
        if min_dist < self.safe_distance:
            priority = 4 # Critical
            
        # Decision logic
        if center_dist < lookahead_dist:
            # Center blocked - we need to turn
            # Turn towards the side with more space
            if left_dist > right_dist:
                # Port is clearer -> Turn Port (Left)
                return AvoidanceAction('alter_course', np.radians(45), 0.5, 
                                     f"LAND AHEAD ({center_dist:.1f}m)! Turning Port", priority)
            else:
                # Starboard is clearer -> Turn Starboard (Right)
                return AvoidanceAction('alter_course', -np.radians(45), 0.5, 
                                     f"LAND AHEAD ({center_dist:.1f}m)! Turning Stbd", priority)
                                     
        elif left_dist < lookahead_dist:
             # Port side blocked -> Turn Starboard (Right)
             return AvoidanceAction('alter_course', -np.radians(30), 0.8, 
                                  f"Land on Port ({left_dist:.1f}m) - Turn Stbd", priority)
                                  
        elif right_dist < lookahead_dist:
             # Starboard side blocked -> Turn Port (Left)
             return AvoidanceAction('alter_course', np.radians(30), 0.8, 
                                  f"Land on Stbd ({right_dist:.1f}m) - Turn Port", priority)
                                  
        return None

    def find_best_maneuver(self, 
                          our_pos: Tuple[float, float],
                          our_heading: float,
                          our_speed: float,
                          obstacles: List[Dict],
                          original_heading: float,
                          time_horizon: float = 25.0) -> Optional[AvoidanceAction]:
        """
        Find the best avoidance maneuver by simulating trajectories.
        
        Args:
            our_pos: Current position (x, y)
            our_heading: Current heading
            our_speed: Current speed
            obstacles: List of obstacle dicts (must contain 'x', 'y', 'heading', 'speed')
            original_heading: The heading we want to be on (to score path adherence)
            time_horizon: How far ahead to simulate (seconds)
            
        Returns:
            Best AvoidanceAction or None if no safe maneuver found
        """
        best_maneuver = None
        best_score = -float('inf')
        
        # Track the "least bad" maneuver if all are unsafe
        best_unsafe_maneuver = None
        max_unsafe_dist = -1.0
        
        # Evaluate all candidates
        for maneuver in self.candidates:
            # 1. Simulate trajectory
            is_safe, min_dist, trajectory = self._simulate_trajectory(
                our_pos, our_heading, our_speed,
                maneuver, obstacles, time_horizon
            )
            
            maneuver.is_safe = is_safe
            maneuver.trajectory = trajectory  # Store for scoring
            
            if is_safe:
                # 2. Score maneuver
                score = self._score_maneuver(
                    maneuver, min_dist, original_heading
                )
                maneuver.score = score
                
                if score > best_score:
                    best_score = score
                    best_maneuver = maneuver
            else:
                # CRITICAL: If maneuver hits land (min_dist = 0), give it -infinity score
                # This ensures land-crossing maneuvers are NEVER chosen
                if min_dist == 0.0:  # Hit land
                    maneuver.score = -float('inf')
                else:
                    # Track best unsafe maneuver (furthest collision with vessels)
                    if min_dist > max_unsafe_dist:
                        max_unsafe_dist = min_dist
                        best_unsafe_maneuver = maneuver
        
        if best_maneuver:
            return AvoidanceAction(
                type='alter_course' if abs(best_maneuver.heading_change) > 0.01 else 'maintain',
                heading_change=best_maneuver.heading_change,
                speed_factor=best_maneuver.speed_factor,
                reason=f"Predictive: {best_maneuver.reason} (Score={best_maneuver.score:.1f})",
                priority=4 # High priority for predictive result
            )
            
        # If no safe maneuver found, check if we're trapped by land
        # If best unsafe option has dist=0 (land collision), DON'T use it - stop instead
        if best_unsafe_maneuver:
            if max_unsafe_dist == 0.0:
                # ALL maneuvers hit land - STOP completely
                return AvoidanceAction(
                    type='emergency',
                    heading_change=0.0,
                    speed_factor=0.0,  # STOP
                    reason="CRITICAL: Surrounded by land! STOPPING.",
                    priority=5
                )
            elif max_unsafe_dist > 5.0:
                # At least avoiding vessels, not land
                return AvoidanceAction(
                    type='emergency',
                    heading_change=best_unsafe_maneuver.heading_change,
                    speed_factor=best_unsafe_maneuver.speed_factor,
                    reason=f"EMERGENCY: Best unsafe option ({best_unsafe_maneuver.reason}, dist={max_unsafe_dist:.1f})",
                    priority=5
                )

        # Absolute last resort: Stop and wait
        return AvoidanceAction(
            type='emergency',
            heading_change=0.0,
            speed_factor=0.0,  # Stop instead of reverse
            reason="NO SAFE MANEUVER FOUND! STOPPING.",
            priority=5
        )

    def _simulate_trajectory(self, 
                           start_pos: Tuple[float, float],
                           start_heading: float,
                           start_speed: float,
                           maneuver: CandidateManeuver,
                           obstacles: List[Dict],
                           duration: float,
                           dt: float = 0.5) -> Tuple[bool, float, List[Dict]]:
        """
        Simulate a maneuver and check for collisions.
        
        Returns:
            (is_safe, min_distance_to_any_obstacle, trajectory)
        """
        # Apply maneuver parameters
        sim_heading = start_heading + maneuver.heading_change
        sim_speed = start_speed * maneuver.speed_factor
        
        # If reversing, cap the speed
        if sim_speed < 0:
            sim_speed = max(sim_speed, -0.3)
        
        # Current state
        cx, cy = start_pos
        min_dist = float('inf')
        trajectory = []  # Store trajectory points for scoring
        
        steps = int(duration / dt)
        
        # Vessel safety radius (accounts for vessel size + safety margin)
        # Balanced at 3.0 - larger values cause over-conservative avoidance
        vessel_radius = 3.0
        
        for i in range(steps):
            t = (i + 1) * dt
            
            # Update our position (simple kinematic model)
            cx += sim_speed * np.cos(sim_heading) * dt
            cy += sim_speed * np.sin(sim_heading) * dt
            
            # Store trajectory point
            trajectory.append({'x': cx, 'y': cy, 'heading': sim_heading})
            
            # 1. Check Static Obstacles (Land) - CRITICAL CHECK
            if self.grid_world:
                # Check all integer grid cells within vessel radius
                # This ensures we don't miss any land cells
                cx_int, cy_int = int(round(cx)), int(round(cy))
                radius_cells = int(np.ceil(vessel_radius))
                
                # Check a square grid around the vessel
                for dx in range(-radius_cells, radius_cells + 1):
                    for dy in range(-radius_cells, radius_cells + 1):
                        gx = cx_int + dx
                        gy = cy_int + dy
                        
                        # Check if this grid cell is within vessel radius
                        cell_dist = np.sqrt(dx**2 + dy**2)
                        if cell_dist > vessel_radius:
                            continue
                        
                        # Check bounds
                        if (gx < 0 or gx >= self.grid_world.width or 
                            gy < 0 or gy >= self.grid_world.height):
                            return False, 0.0, trajectory  # Out of bounds = grounding
                            
                        # Check for land
                        if self.grid_world.grid[gy, gx] > 0.5:
                            return False, 0.0, trajectory  # Hit land = grounding
            
            # 2. Check Dynamic Obstacles
            for obs in obstacles:
                # Predict obstacle position (constant velocity assumption)
                ox = obs['x'] + obs['speed'] * np.cos(obs['heading']) * t
                oy = obs['y'] + obs['speed'] * np.sin(obs['heading']) * t
                
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                min_dist = min(min_dist, dist)
                
                if dist < self.safe_distance:
                    return False, dist, trajectory # Collision with vessel
                    
        return True, min_dist, trajectory

    def _score_maneuver(self, maneuver: CandidateManeuver, 
                       min_dist: float, original_heading: float) -> float:
        """
        Score a safe maneuver. Higher is better.
        
        Priorities:
        1. Safety (Distance to obstacles)
        2. Path adherence (Stay close to planned route)
        3. COLREGs (Starboard turns preferred)
        4. Efficiency (Speed maintenance)
        """
        score = 0.0
        
        # 1. Safety Bonus (up to 100 points)
        # Reward keeping extra distance beyond safe_distance
        score += min(min_dist, 20.0) * 5.0
        
        # 1b. LAND PROXIMITY PENALTY - Check trajectory for land hazards
        # MAXIMUM SEVERITY - absolutely prevent approaching coastlines
        if maneuver.trajectory and self.grid_world:
            land_penalty_total = 0.0
            for point in maneuver.trajectory[::3]:  # Check every 3rd point (more frequent)
                gx, gy = int(point['x']), int(point['y'])
                if 0 <= gx < self.grid_world.width and 0 <= gy < self.grid_world.height:
                    # Check surrounding cells for land with LARGER buffer
                    for dx in range(-8, 9):  # 8-cell buffer (increased from 4)
                        for dy in range(-8, 9):
                            cx, cy = gx + dx, gy + dy
                            if 0 <= cx < self.grid_world.width and 0 <= cy < self.grid_world.height:
                                if self.grid_world.grid[cy, cx] > 0.5:
                                    dist_to_land = np.sqrt(dx**2 + dy**2)
                                    if dist_to_land < 4.0:
                                        land_penalty_total += 1000.0  # ABSOLUTELY CRITICAL
                                    elif dist_to_land < 6.0:
                                        land_penalty_total += 400.0  # CRITICAL zone
                                    elif dist_to_land < 8.0:
                                        land_penalty_total += 100.0   # HIGH RISK zone
            score -= land_penalty_total  # NO CAP - full penalty applied
        
        # 2. Path Adherence - Return to desired course when safe
        # When we have good clearance, strongly favor returning to original heading
        heading_deviation = abs(maneuver.heading_change)
        
        if min_dist > self.warning_distance * 1.5:  # Plenty of clearance (> 27 units)
            # STRONGLY favor returning to path - minimal deviation preferred
            score -= heading_deviation * 20.0  # High penalty for deviation when safe
        elif min_dist > self.warning_distance:  # Good clearance (> 18 units)
            # Moderately favor returning to path
            score -= heading_deviation * 12.0
        else:  # Close to obstacles
            # Allow more deviation when avoiding
            score -= heading_deviation * 5.0
        
        # 3. COLREGs Compliance - REMOVED
        # Safety overrides COLREGs - if starboard leads to grounding, turn port
        # The land proximity penalties will ensure safe maneuvers are chosen
        # No preference for starboard turns when near coastlines
        
        # 4. Efficiency
        # Penalize speed reduction
        if maneuver.speed_factor < 1.0:
            score -= (1.0 - maneuver.speed_factor) * 30.0
        if maneuver.speed_factor < 0: # Reversing
            score -= 150.0  # Heavy penalty for reversing
            
        return score
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