"""
Collision Avoidance for Autonomous Vessels

Implements avoidance maneuvers when collision risk is detected:
- Course alteration
- Speed reduction
- Combined maneuvers
- Basic COLREGs-inspired rules
"""

import numpy as np
import logging
import os
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

# Set up file-based logging for debug
_log_file = os.path.join(os.path.dirname(__file__), '..', '..', 'collision_avoidance_debug.log')
_ca_logger = logging.getLogger('collision_avoidance')
_ca_logger.setLevel(logging.DEBUG)
_ca_logger.handlers = []
_file_handler = logging.FileHandler(_log_file, mode='w')
_file_handler.setFormatter(logging.Formatter('%(message)s'))
_ca_logger.addHandler(_file_handler)
_ca_logger.propagate = False

def _log(msg: str):
    """Log to file only."""
    _ca_logger.debug(msg)


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
    Implements collision avoidance strategies with dynamics-aware prediction.
    
    Features:
    - Predictive trajectory simulation using Nomoto vessel dynamics
    - Maneuver commitment to prevent oscillation (S-shaped tracks)
    - Graduated response (try small course changes before hard-over)
    - Continuous risk re-assessment during committed maneuvers
    - Automatic return to track when danger clears
    """
    
    # IMO standard rudder rate: 70° in 11 seconds
    IMO_RUDDER_RATE = np.radians(70.0 / 11.0)  # ~0.111 rad/s (6.36°/s)
    
    def __init__(self, 
                 safe_distance: float = 8.0,
                 warning_distance: float = 15.0,
                 avoidance_angle: float = np.radians(30),
                 grid_world=None,
                 # Vessel dynamics parameters for prediction
                 vessel_K: float = 0.3,
                 vessel_T: float = 5.0,
                 rudder_rate: Optional[float] = None,
                 max_rudder: float = np.radians(35.0)):
        """
        Initialize collision avoidance with dynamics-aware prediction.
        
        Args:
            safe_distance: Minimum safe distance
            warning_distance: Distance to start avoiding
            avoidance_angle: Standard course alteration angle (radians)
            grid_world: Optional GridWorld object for static obstacle checking
            vessel_K: Nomoto gain constant (responsiveness)
            vessel_T: Nomoto time constant (how quickly vessel responds)
            rudder_rate: Maximum rudder rate (rad/s). None=instant, 0=IMO standard
            max_rudder: Maximum rudder angle (radians)
        """
        self.safe_distance = safe_distance
        self.warning_distance = warning_distance
        self.avoidance_angle = avoidance_angle
        self.grid_world = grid_world
        
        # Vessel dynamics for trajectory prediction
        self.vessel_K = vessel_K
        self.vessel_T = vessel_T
        self.max_rudder = max_rudder
        if rudder_rate is None:
            self.rudder_rate = None  # Instant rudder (legacy)
        elif rudder_rate <= 0:
            self.rudder_rate = self.IMO_RUDDER_RATE
        else:
            self.rudder_rate = rudder_rate
        
        # State tracking
        self.active_avoidance = False
        self.avoided_vessels = set()  # Track which vessels we're avoiding
        self.original_speed = None
        
        # Maneuver commitment state (prevents oscillation)
        self.committed_maneuver = None      # Current committed AvoidanceAction
        self.committed_absolute_heading = None  # ABSOLUTE target heading (not relative!)
        self.commit_start_time = None       # When commitment started
        self.commit_duration = 12.0         # Base duration - vessel needs time to respond!
        self.predicted_cpa_at_commit = None # CPA predicted when maneuver was committed
        self.predicted_tcpa_at_commit = None
        self.return_to_track_heading = None # Original desired heading to return to
        self.last_min_cpa = float('inf')    # Track if CPA is improving
        
        # Current vessel state (updated externally)
        self.current_turn_rate = 0.0
        self.current_rudder = 0.0
        
        # Define candidate maneuvers for predictive search
        self.candidates = self._generate_candidates()

    def _generate_candidates(self) -> List[CandidateManeuver]:
        """
        Generate candidate course alterations for predictive evaluation.
        
        COURSE CHANGES ONLY - no speed adjustments.
        Speed adjustments don't change CPA geometry, they only delay TCPA.
        A proper avoidance maneuver changes the geometry of the encounter.
        
        Uses graduated steps to find MINIMUM effective course change:
        - Fine steps (5°, 10°, 15°) for early/mild avoidance
        - Medium steps (20°, 25°, 30°) for moderate avoidance  
        - Large steps (40°, 50°, 60°) for emergency avoidance
        
        COLREGs preference: Starboard turns tried first (Rule 14, 15, 17).
        """
        candidates = []
        
        # 1. Maintain course (baseline - if safe, don't turn!)
        candidates.append(CandidateManeuver(0.0, 1.0, reason="Maintain course"))
        
        # 2. Graduated course alterations - Starboard FIRST (COLREGs preference)
        # These are tried in order, so fine adjustments come before hard turns
        for deg in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
            candidates.append(CandidateManeuver(np.radians(-deg), 1.0, reason=f"Turn Stbd {deg}°"))
            
        # 3. Port turns - only when starboard is blocked by land/obstacles
        for deg in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
            candidates.append(CandidateManeuver(np.radians(deg), 1.0, reason=f"Turn Port {deg}°"))
            
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
        Apply avoidance actions to modify desired heading.
        
        COURSE CHANGES ONLY - speed remains unchanged.
        Speed adjustments don't change CPA geometry, they only delay the problem.
        
        IMPORTANT: Uses committed_absolute_heading if available, otherwise
        computes from desired_heading + heading_change. This ensures we
        maintain a consistent avoidance heading throughout the maneuver.
        
        Args:
            desired_heading: Original desired heading from path follower
            desired_speed: Original desired speed
            avoidance_actions: List of avoidance actions
            
        Returns:
            Tuple of (modified_heading, original_speed)
        """
        if not avoidance_actions:
            self.active_avoidance = False
            return desired_heading, desired_speed
        
        self.active_avoidance = True
        
        # If we have a committed absolute heading, USE IT!
        # This is the key to preventing S-tracks - maintain ONE consistent heading
        if self.committed_absolute_heading is not None:
            return self.committed_absolute_heading, desired_speed
        
        # Fallback: compute from action (should rarely happen now)
        # Sort by priority (highest first)
        avoidance_actions.sort(key=lambda x: x.priority, reverse=True)
        
        # Take the highest priority action
        primary_action = avoidance_actions[0]

        # Apply heading change only
        modified_heading = desired_heading
        if primary_action.type in ['alter_course', 'emergency']:
            modified_heading = desired_heading + primary_action.heading_change
            # Normalize to [-π, π]
            modified_heading = np.arctan2(np.sin(modified_heading),
                                         np.cos(modified_heading))
        
        # Keep speed unchanged - course changes are what matter!
        return modified_heading, desired_speed
    
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

    def get_avoidance_action(self,
                            current_time: float,
                            our_pos: Tuple[float, float],
                            our_heading: float,
                            our_speed: float,
                            current_turn_rate: float,
                            current_rudder: float,
                            obstacles: List[Dict],
                            collision_infos: List[Dict],
                            desired_heading: float) -> Tuple[Optional[AvoidanceAction], bool]:
        """
        HOLISTIC collision avoidance with STRICT commitment.
        """
        # Update internal dynamics state
        self.current_turn_rate = current_turn_rate
        self.current_rudder = current_rudder
        
        # Calculate current situation from ALL obstacles
        current_min_cpa = float('inf')
        current_min_tcpa = float('inf')
        max_risk = 0
        
        for info_dict in collision_infos:
            info = info_dict['info']
            if info.cpa_distance < current_min_cpa:
                current_min_cpa = info.cpa_distance
            if info.tcpa > 0 and info.tcpa < current_min_tcpa:
                current_min_tcpa = info.tcpa
            max_risk = max(max_risk, info.risk_level)
        
        _log(f"\n=== t={current_time:.1f}s === pos=({our_pos[0]:.1f},{our_pos[1]:.1f}) hdg={np.degrees(our_heading):.1f}° spd={our_speed:.1f}")
        _log(f"  min_cpa={current_min_cpa:.1f} min_tcpa={current_min_tcpa:.1f} max_risk={max_risk}")
        _log(f"  committed_maneuver={self.committed_maneuver}")
        _log(f"  committed_absolute_heading={np.degrees(self.committed_absolute_heading) if self.committed_absolute_heading else None}")
        _log(f"  commit_start_time={self.commit_start_time} commit_duration={self.commit_duration}")
        
        # --- CASE 1: No risk - clear commitment and return to track ---
        # KEY INSIGHT: We can only return to track when the obstacle has PASSED, not just
        # when CPA has improved. Otherwise we'll turn back into the danger!
        #
        # Conditions for "no risk" (obstacle has truly passed):
        # 1. All obstacles are behind us (TCPA <= 0 or TCPA = inf), OR
        # 2. CPA is VERY safe (> warning_distance + buffer) AND TCPA is small (obstacle passing now)
        #
        # We do NOT return to track just because CPA improved - the improvement might be
        # because of our avoidance maneuver, and returning will undo it!
        
        obstacles_behind = current_min_tcpa <= 0 or current_min_tcpa == float('inf')
        obstacle_passing_safely = current_min_cpa > (self.warning_distance + 5.0) and current_min_tcpa < 30.0
        
        no_risk = obstacles_behind or obstacle_passing_safely
        
        if no_risk:
            _log(f"  CASE 1: No risk (behind={obstacles_behind}, passing_safely={obstacle_passing_safely})")
            if self.committed_maneuver is not None:
                _log(f"    Clearing commitment, returning to track")
                self.committed_maneuver = None
                self.committed_absolute_heading = None
                self.commit_start_time = None
                self.active_avoidance = False
                self.return_to_track_heading = None
            _log(f"  RETURNING: None, return_to_track=True")
            return None, True
        
        # --- CASE 2: Currently committed - HONOR THE COMMITMENT ---
        # Check if the obstacle has ACTUALLY passed (not just momentary CPA improvement)
        if self.committed_maneuver is not None and self.commit_start_time is not None:
            time_in_commit = current_time - self.commit_start_time
            _log(f"  CASE 2: Committed - time_in_commit={time_in_commit:.1f}s (duration={self.commit_duration}s)")
            _log(f"    Absolute heading target: {np.degrees(self.committed_absolute_heading):.1f}°")
            
            # Check if we can exit commitment early because obstacle has ACTUALLY passed
            # CRITICAL: Don't clear just because CPA improved - that might be temporary!
            # Obstacle has passed if:
            # 1. TCPA is negative/zero (danger is now BEHIND us), OR
            # 2. CPA is safe AND TCPA is very large (obstacle was always far away)
            obstacle_passed = current_min_tcpa <= 0
            plenty_time = current_min_tcpa > 120.0  # More than 2 minutes away
            cpa_is_safe = current_min_cpa > self.warning_distance
            
            can_clear_commitment = obstacle_passed or (cpa_is_safe and plenty_time)
            
            if can_clear_commitment:
                _log(f"    Obstacle cleared! (tcpa={current_min_tcpa:.1f}, cpa={current_min_cpa:.1f}, passed={obstacle_passed})")
                _log(f"    Clearing commitment, returning to track")
                self.committed_maneuver = None
                self.committed_absolute_heading = None
                self.commit_start_time = None
                self.active_avoidance = False
                self.return_to_track_heading = None
                return None, True
            
            if time_in_commit < self.commit_duration:
                # Check if vessel has actually reached the committed heading
                # If still turning, the CPA based on current heading is MISLEADING!
                raw_heading_error = our_heading - self.committed_absolute_heading
                # Normalize to [-π, π]
                heading_error = abs((raw_heading_error + np.pi) % (2 * np.pi) - np.pi)
                vessel_on_committed_heading = heading_error < np.radians(5.0)  # Within 5 degrees
                
                _log(f"    Heading error to committed: {np.degrees(heading_error):.1f}° (on_heading={vessel_on_committed_heading})")
                
                # Only consider override if:
                # 1. Vessel HAS reached committed heading (so CPA calc is valid), AND
                # 2. CPA is still dangerously low (< 1.0), AND
                # 3. We've been committed for at least 5 seconds (give maneuver time)
                actual_collision = (vessel_on_committed_heading and 
                                   current_min_cpa < 1.0 and 
                                   time_in_commit > 5.0)
                _log(f"    Still in window. actual_collision={actual_collision} (on_hdg: {vessel_on_committed_heading}, cpa<1.0: {current_min_cpa<1.0}, time>5s: {time_in_commit>5.0})")
                
                # ALWAYS honor commitment unless we're ON the committed heading and still in danger
                # Re-planning while turning doesn't help - the CPA calc doesn't reflect where we're going!
                if not actual_collision:
                    _log(f"  RETURNING: Keep committed maneuver {self.committed_maneuver}")
                    return self.committed_maneuver, False
                else:
                    _log(f"    ACTUAL COLLISION override - vessel on committed heading but still in danger!")
            else:
                _log(f"    Commitment EXPIRED (time_in_commit={time_in_commit:.1f} >= {self.commit_duration}s)")
            
            # Commitment expired or actual collision - clear and re-plan
            _log(f"    Clearing commitment, will replan")
            self.committed_maneuver = None
            self.committed_absolute_heading = None
            self.commit_start_time = None
        
        # --- CASE 3: Need to find a maneuver ---
        _log(f"  CASE 3: Finding new maneuver")
        
        # Store original heading for return-to-track AND use it for avoidance calculation
        if self.return_to_track_heading is None:
            self.return_to_track_heading = desired_heading
            _log(f"    Stored return_to_track_heading={np.degrees(desired_heading):.1f}°")
        
        # Project 5 minutes ahead
        projection_time = 300.0
        
        # Find the best course change
        best_action = self._find_minimum_safe_course_change(
            our_pos=our_pos,
            our_heading=our_heading,
            our_speed=our_speed,
            obstacles=obstacles,
            desired_heading=self.return_to_track_heading,
            projection_time=projection_time,
            current_turn_rate=current_turn_rate,
            current_rudder=current_rudder
        )
        
        if best_action:
            # COMMIT to this course change
            self.committed_maneuver = best_action
            self.commit_start_time = current_time
            
            # Calculate and store ABSOLUTE target heading
            # This is the key fix: avoidance heading = return_to_track + heading_change
            self.committed_absolute_heading = self.return_to_track_heading + best_action.heading_change
            self.committed_absolute_heading = np.arctan2(
                np.sin(self.committed_absolute_heading),
                np.cos(self.committed_absolute_heading)
            )
            _log(f"    Calculated absolute heading: {np.degrees(self.return_to_track_heading):.1f}° + {np.degrees(best_action.heading_change):.1f}° = {np.degrees(self.committed_absolute_heading):.1f}°")
            
            # Commit duration: longer for bigger turns (vessel needs time)
            turn_magnitude = abs(best_action.heading_change)
            if turn_magnitude < np.radians(15):
                self.commit_duration = 15.0  # Small turn
            elif turn_magnitude < np.radians(30):
                self.commit_duration = 25.0  # Medium turn
            else:
                self.commit_duration = 35.0  # Large turn - needs lots of time
            
            self.predicted_cpa_at_commit = current_min_cpa
            self.active_avoidance = True
            
            _log(f"  NEW COMMITMENT: {best_action}")
            _log(f"    commit_start_time={self.commit_start_time}, duration={self.commit_duration}s")
            _log(f"  RETURNING: {best_action}")
            return best_action, False
        
        # Fallback: static obstacle check
        _log(f"  No dynamic solution, checking static obstacles")
        static_action = self.check_static_obstacles(our_pos, our_heading, our_speed)
        if static_action:
            self.committed_maneuver = static_action
            self.commit_start_time = current_time
            self.commit_duration = 20.0
            self.active_avoidance = True
            _log(f"  STATIC ACTION: {static_action}")
            return static_action, False
        
        _log(f"  RETURNING: None (no action found)")
        return None, False

    def _find_minimum_safe_course_change(self,
                                         our_pos: Tuple[float, float],
                                         our_heading: float,
                                         our_speed: float,
                                         obstacles: List[Dict],
                                         desired_heading: float,
                                         projection_time: float,
                                         current_turn_rate: float,
                                         current_rudder: float) -> Optional[AvoidanceAction]:
        """
        Find the MINIMUM course change that achieves safe CPA with ALL obstacles.
        
        NEW APPROACH: Two-phase trajectory simulation
        Phase 1: Turn to avoidance heading, maintain until dynamic obstacle clears
        Phase 2: Return to track (desired_heading) - must not hit static obstacles
        
        A course is SAFE if:
        - Phase 1: Maintains safe distance from dynamic obstacles AND doesn't hit land
        - Phase 2: Returns to track without hitting land
        """
        _log(f"    _find_minimum_safe_course_change: evaluating {len(self.candidates)} candidates")
        _log(f"      obstacles: {len(obstacles)}, desired_heading: {np.degrees(desired_heading):.1f}°")
        
        candidate_results = []
        
        for candidate in self.candidates:
            # Simulate TWO-PHASE trajectory
            is_safe, min_cpa, time_to_clear = self._simulate_two_phase_trajectory(
                start_pos=our_pos,
                start_heading=our_heading,
                start_speed=our_speed,
                avoidance_heading_change=candidate.heading_change,
                return_heading=desired_heading,
                obstacles=obstacles,
                max_duration=projection_time,
                current_turn_rate=current_turn_rate,
                current_rudder=current_rudder
            )
            
            candidate_results.append((candidate, is_safe, min_cpa, time_to_clear))
            
            if is_safe and min_cpa >= self.safe_distance:
                # This is the minimum course change that works!
                _log(f"      FOUND SAFE: {candidate.reason} -> CPA={min_cpa:.1f}, clear_time={time_to_clear:.0f}s")
                return AvoidanceAction(
                    type='alter_course' if abs(candidate.heading_change) > 0.01 else 'maintain',
                    heading_change=candidate.heading_change,
                    speed_factor=1.0,
                    reason=f"{candidate.reason} (CPA={min_cpa:.1f})",
                    priority=3 if abs(candidate.heading_change) < np.radians(20) else 4
                )
        
        # Log all results for debugging
        _log(f"    No safe option found. All results:")
        for candidate, is_safe, min_cpa, time_to_clear in candidate_results:
            _log(f"      {candidate.reason}: safe={is_safe}, CPA={min_cpa:.1f}, clear={time_to_clear:.0f}s")
        
        # No fully safe option - find best unsafe option (highest CPA that doesn't hit land)
        best_unsafe_cpa = -1
        best_unsafe_candidate = None
        
        for candidate, is_safe, min_cpa, time_to_clear in candidate_results:
            # Prefer options that don't hit land (min_cpa > 0)
            if min_cpa > best_unsafe_cpa:
                best_unsafe_cpa = min_cpa
                best_unsafe_candidate = candidate
        
        if best_unsafe_candidate and best_unsafe_cpa > 0:
            _log(f"    BEST UNSAFE: {best_unsafe_candidate.reason} -> CPA={best_unsafe_cpa:.1f}")
            return AvoidanceAction(
                type='emergency',
                heading_change=best_unsafe_candidate.heading_change,
                speed_factor=1.0,
                reason=f"BEST: {best_unsafe_candidate.reason} (CPA={best_unsafe_cpa:.1f})",
                priority=5
            )
        
        _log(f"    NO SOLUTION FOUND!")
        return None

    def _simulate_two_phase_trajectory(self,
                                       start_pos: Tuple[float, float],
                                       start_heading: float,
                                       start_speed: float,
                                       avoidance_heading_change: float,
                                       return_heading: float,
                                       obstacles: List[Dict],
                                       max_duration: float,
                                       current_turn_rate: float,
                                       current_rudder: float,
                                       dt: float = 1.0) -> Tuple[bool, float, float]:
        """
        Simulate TWO-PHASE avoidance trajectory:
        
        Phase 1: Turn to avoidance heading, maintain until ALL dynamic obstacles clear
                 (distance > warning_distance and moving away)
        Phase 2: Return to track heading - must be safe from static obstacles
        
        Returns:
            (is_safe, minimum_CPA, time_when_obstacles_clear)
        """
        avoidance_heading = start_heading + avoidance_heading_change
        
        # Initialize vessel state
        cx, cy = start_pos
        heading = start_heading
        turn_rate = current_turn_rate
        rudder = current_rudder
        speed = start_speed
        
        min_cpa = float('inf')
        time_to_clear = max_duration
        phase = 1  # 1 = avoiding, 2 = returning to track
        target_heading = avoidance_heading
        
        steps = int(max_duration / dt)
        
        # PD controller gains - ship-like gentle control
        Kp = 1.0   # Gentle proportional response
        Kd = 1.5   # Damping to prevent overshoot
        
        for step in range(steps):
            t = step * dt
            
            # --- Vessel Dynamics ---
            heading_error = target_heading - heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            rudder_cmd = Kp * heading_error - Kd * turn_rate
            rudder_cmd = np.clip(rudder_cmd, -self.max_rudder, self.max_rudder)
            
            if self.rudder_rate is not None:
                max_delta = self.rudder_rate * dt
                rudder_delta = np.clip(rudder_cmd - rudder, -max_delta, max_delta)
                rudder += rudder_delta
            else:
                rudder = rudder_cmd
            
            d_turn_rate = (self.vessel_K * rudder - turn_rate) / self.vessel_T
            turn_rate += d_turn_rate * dt
            heading += turn_rate * dt
            heading = np.arctan2(np.sin(heading), np.cos(heading))
            
            cx += speed * np.cos(heading) * dt
            cy += speed * np.sin(heading) * dt
            
            # --- Check Land Collision (Static Obstacles) ---
            if self.grid_world is not None:
                gx, gy = int(round(cx)), int(round(cy))
                
                # Out of bounds is OK (open sea beyond map) - just skip land check
                if 0 <= gx < self.grid_world.width and 0 <= gy < self.grid_world.height:
                    # Check for land with safety buffer
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            check_x, check_y = gx + dx, gy + dy
                            if 0 <= check_x < self.grid_world.width and 0 <= check_y < self.grid_world.height:
                                if self.grid_world.grid[check_y, check_x] > 0.5:
                                    dist_to_land = np.sqrt(dx**2 + dy**2)
                                    if dist_to_land < 2.5:
                                        return False, 0.0, t  # Hit land
            
            # --- Check Dynamic Obstacles ---
            all_clear = True
            for obs in obstacles:
                # Predict obstacle position at this time
                ox = obs['x'] + obs['speed'] * np.cos(obs['heading']) * t
                oy = obs['y'] + obs['speed'] * np.sin(obs['heading']) * t
                
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                min_cpa = min(min_cpa, dist)
                
                # Check for collision
                if dist < self.safe_distance * 0.5:
                    return False, dist, t
                
                # Check if this obstacle is still a threat
                if dist < self.warning_distance:
                    all_clear = False
                else:
                    # Check if moving away (relative velocity)
                    # Our velocity
                    our_vx = speed * np.cos(heading)
                    our_vy = speed * np.sin(heading)
                    # Obstacle velocity
                    obs_vx = obs['speed'] * np.cos(obs['heading'])
                    obs_vy = obs['speed'] * np.sin(obs['heading'])
                    # Relative position
                    rel_x = ox - cx
                    rel_y = oy - cy
                    # Relative velocity
                    rel_vx = obs_vx - our_vx
                    rel_vy = obs_vy - our_vy
                    # Closing speed (negative = closing, positive = separating)
                    closing = (rel_x * rel_vx + rel_y * rel_vy) / (dist + 0.001)
                    
                    if closing < 0:  # Still closing
                        all_clear = False
            
            # --- Phase Transition ---
            if phase == 1 and all_clear and t > 10.0:
                # All dynamic obstacles are clear - switch to Phase 2
                phase = 2
                target_heading = return_heading
                time_to_clear = t
        
        # Trajectory is safe if we never hit anything and min_cpa >= safe_distance
        is_safe = min_cpa >= self.safe_distance
        return is_safe, min_cpa, time_to_clear

    def _simulate_trajectory_holistic(self,
                                      start_pos: Tuple[float, float],
                                      start_heading: float,
                                      start_speed: float,
                                      course_change: float,
                                      obstacles: List[Dict],
                                      duration: float,
                                      current_turn_rate: float,
                                      current_rudder: float,
                                      dt: float = 1.0) -> Tuple[bool, float, List[Dict]]:
        """
        Simulate vessel trajectory with Nomoto dynamics, checking ALL obstacles.
        
        Projects the vessel forward for `duration` seconds, modeling:
        - Rudder rate limits (IMO standard)
        - Nomoto turn dynamics (K, T parameters)
        - Current vessel state
        
        Checks distance to ALL obstacles and land at each timestep.
        
        Returns:
            (is_safe, minimum_CPA_to_any_obstacle, trajectory_points)
        """
        # Target heading after course change
        target_heading = start_heading + course_change
        
        # Initialize vessel state
        cx, cy = start_pos
        heading = start_heading
        turn_rate = current_turn_rate
        rudder = current_rudder
        speed = start_speed
        
        min_cpa = float('inf')
        trajectory = []
        
        steps = int(duration / dt)
        
        # PD controller gains - ship-like gentle control
        Kp = 1.0   # Gentle proportional response
        Kd = 1.5   # Damping to prevent overshoot
        
        for step in range(steps):
            t = step * dt
            
            # --- Vessel Dynamics ---
            # Calculate heading error to target
            heading_error = target_heading - heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # PD controller for rudder command
            rudder_cmd = Kp * heading_error - Kd * turn_rate
            rudder_cmd = np.clip(rudder_cmd, -self.max_rudder, self.max_rudder)
            
            # Apply rudder rate limit
            if self.rudder_rate is not None:
                max_delta = self.rudder_rate * dt
                rudder_delta = np.clip(rudder_cmd - rudder, -max_delta, max_delta)
                rudder += rudder_delta
            else:
                rudder = rudder_cmd
            
            # Nomoto dynamics: dψ/dt = (K * δ - ψ) / T
            d_turn_rate = (self.vessel_K * rudder - turn_rate) / self.vessel_T
            turn_rate += d_turn_rate * dt
            
            # Update heading
            heading += turn_rate * dt
            heading = np.arctan2(np.sin(heading), np.cos(heading))
            
            # Update position
            cx += speed * np.cos(heading) * dt
            cy += speed * np.sin(heading) * dt
            
            trajectory.append({'x': cx, 'y': cy, 'heading': heading, 'time': t})
            
            # --- Check Land Collision ---
            if self.grid_world is not None:
                gx, gy = int(round(cx)), int(round(cy))
                
                # Out of bounds is OK (open sea beyond map) - just skip land check
                if 0 <= gx < self.grid_world.width and 0 <= gy < self.grid_world.height:
                    # Check for land (with small buffer)
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            check_x, check_y = gx + dx, gy + dy
                            if 0 <= check_x < self.grid_world.width and 0 <= check_y < self.grid_world.height:
                                if self.grid_world.grid[check_y, check_x] > 0.5:
                                    dist_to_land = np.sqrt(dx**2 + dy**2)
                                    if dist_to_land < 2.0:
                                        return False, 0.0, trajectory
            
            # --- Check ALL Dynamic Obstacles ---
            for obs in obstacles:
                # Predict obstacle position at this time
                ox = obs['x'] + obs['speed'] * np.cos(obs['heading']) * t
                oy = obs['y'] + obs['speed'] * np.sin(obs['heading']) * t
                
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                min_cpa = min(min_cpa, dist)
                
                # Check if collision (within safe distance)
                if dist < self.safe_distance * 0.5:
                    return False, dist, trajectory
        
        # Trajectory is safe if min_cpa >= safe_distance
        is_safe = min_cpa >= self.safe_distance
        return is_safe, min_cpa, trajectory

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
        
        # Cast rays: Center, Port (-30°), Starboard (+30°)
        # Using wider whiskers to detect land masses earlier
        angles = [0, np.radians(30), -np.radians(30)]
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
                return AvoidanceAction('alter_course', np.radians(45), 1.0, 
                                     f"LAND AHEAD ({center_dist:.1f}m)! Turning Port", priority)
            else:
                # Starboard is clearer -> Turn Starboard (Right)
                return AvoidanceAction('alter_course', -np.radians(45), 1.0, 
                                     f"LAND AHEAD ({center_dist:.1f}m)! Turning Stbd", priority)
                                     
        elif left_dist < lookahead_dist:
             # Port side blocked -> Turn Starboard (Right)
             return AvoidanceAction('alter_course', -np.radians(30), 1.0, 
                                  f"Land on Port ({left_dist:.1f}m) - Turn Stbd", priority)
                                  
        elif right_dist < lookahead_dist:
             # Starboard side blocked -> Turn Port (Left)
             return AvoidanceAction('alter_course', np.radians(30), 1.0, 
                                  f"Land on Stbd ({right_dist:.1f}m) - Turn Port", priority)
                                  
        return None

    def find_best_maneuver(self, 
                          our_pos: Tuple[float, float],
                          our_heading: float,
                          our_speed: float,
                          obstacles: List[Dict],
                          original_heading: float,
                          time_horizon: float = 25.0,
                          current_turn_rate: float = 0.0,
                          current_rudder: float = 0.0) -> Optional[AvoidanceAction]:
        """
        Find the best avoidance maneuver by simulating trajectories with Nomoto dynamics.
        
        Args:
            our_pos: Current position (x, y)
            our_heading: Current heading
            our_speed: Current speed
            obstacles: List of obstacle dicts (must contain 'x', 'y', 'heading', 'speed')
            original_heading: The heading we want to be on (to score path adherence)
            time_horizon: How far ahead to simulate (seconds)
            current_turn_rate: Current vessel turn rate (rad/s)
            current_rudder: Current actual rudder angle (radians)
            
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
            # 1. Simulate trajectory with dynamics
            is_safe, min_dist, trajectory = self._simulate_trajectory(
                our_pos, our_heading, our_speed,
                maneuver, obstacles, time_horizon,
                dt=0.5,
                current_turn_rate=current_turn_rate,
                current_rudder=current_rudder
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
                           dt: float = 0.5,
                           current_turn_rate: float = 0.0,
                           current_rudder: float = 0.0) -> Tuple[bool, float, List[Dict]]:
        """
        Simulate a maneuver using Nomoto vessel dynamics and check for collisions.
        
        This accurately predicts what heading changes are achievable given:
        - Rudder rate limits (IMO standard: 6.36°/s)
        - Nomoto turn response (T, K parameters)
        - Current vessel state (turn rate, rudder angle)
        
        Args:
            start_pos: Starting position (x, y)
            start_heading: Starting heading (radians)
            start_speed: Starting speed
            maneuver: Candidate maneuver to evaluate
            obstacles: List of obstacle dicts with x, y, heading, speed
            duration: Simulation duration (seconds)
            dt: Time step (seconds)
            current_turn_rate: Current vessel turn rate (rad/s)
            current_rudder: Current actual rudder angle (radians)
            
        Returns:
            (is_safe, min_distance_to_any_obstacle, trajectory)
        """
        # Target heading we want to achieve
        target_heading = start_heading + maneuver.heading_change
        sim_speed = start_speed * maneuver.speed_factor
        
        # If reversing, cap the speed
        if sim_speed < 0:
            sim_speed = max(sim_speed, -0.3)
        
        # Initialize vessel dynamics state
        cx, cy = start_pos
        heading = start_heading
        turn_rate = current_turn_rate
        rudder = current_rudder
        
        min_dist = float('inf')
        trajectory = []
        
        steps = int(duration / dt)
        
        # Vessel safety radius
        vessel_radius = 3.0
        
        # PD controller gains - ship-like gentle control
        Kp = 1.0   # Gentle proportional response
        Kd = 1.5   # Damping to prevent overshoot
        
        for i in range(steps):
            t = (i + 1) * dt
            
            # --- NOMOTO DYNAMICS SIMULATION ---
            
            # 1. Compute rudder command using PD controller to achieve target heading
            heading_error = target_heading - heading
            # Normalize to [-π, π]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # PD control: rudder_cmd = Kp * error - Kd * turn_rate
            rudder_cmd = Kp * heading_error - Kd * turn_rate
            rudder_cmd = np.clip(rudder_cmd, -self.max_rudder, self.max_rudder)
            
            # 2. Apply rudder rate limiting
            if self.rudder_rate is not None:
                max_delta = self.rudder_rate * dt
                rudder_error = rudder_cmd - rudder
                delta = np.clip(rudder_error, -max_delta, max_delta)
                rudder += delta
            else:
                rudder = rudder_cmd
            
            # 3. Nomoto first-order model: T * dr/dt + r = K * δ
            # Rearranged: dr/dt = (K * δ - r) / T
            d_turn_rate = (self.vessel_K * rudder - turn_rate) / self.vessel_T
            turn_rate += d_turn_rate * dt
            
            # 4. Update heading
            heading += turn_rate * dt
            heading = np.arctan2(np.sin(heading), np.cos(heading))
            
            # 5. Update position
            cx += sim_speed * np.cos(heading) * dt
            cy += sim_speed * np.sin(heading) * dt
            
            # Store trajectory point
            trajectory.append({
                'x': cx, 'y': cy, 'heading': heading, 
                'turn_rate': turn_rate, 'rudder': rudder
            })
            
            # --- COLLISION CHECKS ---
            
            # 1. Check Static Obstacles (Land)
            if self.grid_world:
                cx_int, cy_int = int(round(cx)), int(round(cy))
                radius_cells = int(np.ceil(vessel_radius))
                
                for dx in range(-radius_cells, radius_cells + 1):
                    for dy in range(-radius_cells, radius_cells + 1):
                        gx = cx_int + dx
                        gy = cy_int + dy
                        
                        cell_dist = np.sqrt(dx**2 + dy**2)
                        if cell_dist > vessel_radius:
                            continue
                        
                        if (gx < 0 or gx >= self.grid_world.width or 
                            gy < 0 or gy >= self.grid_world.height):
                            return False, 0.0, trajectory
                            
                        if self.grid_world.grid[gy, gx] > 0.5:
                            return False, 0.0, trajectory
            
            # 2. Check Dynamic Obstacles
            for obs in obstacles:
                ox = obs['x'] + obs['speed'] * np.cos(obs['heading']) * t
                oy = obs['y'] + obs['speed'] * np.sin(obs['heading']) * t
                
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                min_dist = min(min_dist, dist)
                
                if dist < self.safe_distance:
                    return False, dist, trajectory
                    
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
        land_penalty_total = 0.0  # Initialize outside the if block
        if maneuver.trajectory and self.grid_world:
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
        
        # 3. COLREGs Compliance - Prefer starboard turns when safe from land
        # Only apply if land penalty is low (not near coastline)
        if land_penalty_total < 100:  # Not near land
            if maneuver.heading_change < 0:  # Starboard turn (negative = right)
                score += 15.0  # Bonus for COLREGs-compliant starboard turn
            elif maneuver.heading_change > 0:  # Port turn
                score -= 10.0  # Small penalty for port turn
        
        # 4. STRONGLY prefer course changes over speed reduction
        # Course changes modify CPA geometry; speed changes only delay TCPA
        if abs(maneuver.heading_change) > np.radians(5):
            score += 25.0  # Significant bonus for actual course change
        
        # 5. Penalize speed reduction - it rarely solves the problem
        if maneuver.speed_factor < 1.0:
            score -= (1.0 - maneuver.speed_factor) * 50.0  # Higher penalty!
        if maneuver.speed_factor <= 0.3:  # Near stop
            score -= 80.0  # Heavy penalty - stopping rarely helps
        if maneuver.speed_factor < 0:  # Reversing
            score -= 200.0  # Massive penalty
            
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