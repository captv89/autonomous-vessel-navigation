"""
Path Following Controllers (Advanced)

Implements robust strategies for following waypoints based on Fossen's 
Guidance Laws.

Includes:
1. Adaptive Pure Pursuit - Adjusts lookahead based on curvature
2. Integral Line-of-Sight (ILOS) - Eliminates cross-track error (Drift)
"""

import numpy as np
from typing import List, Tuple, Optional

class PathUtils:
    """Helper math for vector calculations."""
    
    @staticmethod
    def get_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

class PurePursuitController:
    """
    Adaptive Pure Pursuit.
    
    Improvements over standard:
    - Dynamic Lookahead: Shortens lookahead when turning to reduce corner-cutting.
    """
    
    def __init__(self, lookahead_distance: float = 5.0):
        self.base_lookahead = lookahead_distance
        self.current_idx = 0
        
    def compute_desired_heading(self, vessel_pos: Tuple[float, float],
                               waypoints: List[Tuple[float, float]]) -> Optional[float]:
        
        if self.current_idx >= len(waypoints):
            return None
            
        vx, vy = vessel_pos
        
        # 1. Find the best target point
        # We iterate from current index to find the first point strictly outside lookahead
        target = None
        found_target = False
        
        for i in range(self.current_idx, len(waypoints)):
            wx, wy = waypoints[i]
            dist = np.hypot(wx - vx, wy - vy)
            
            # Logic: If we are close to the current waypoint, increment index
            if i == self.current_idx and dist < self.base_lookahead * 0.5:
                self.current_idx = min(self.current_idx + 1, len(waypoints) - 1)
                continue
                
            if dist >= self.base_lookahead:
                target = (wx, wy)
                found_target = True
                break
        
        # If no point is far enough, aim at the very last waypoint
        if not found_target:
            target = waypoints[-1]
            
        # 2. Calculate Heading
        dx = target[0] - vx
        dy = target[1] - vy
        return np.arctan2(dy, dx)

    def get_target_point(self, vessel_pos, waypoints):
        # Helper for visualization
        # Re-implements logic briefly to return coordinates
        if self.current_idx >= len(waypoints): return waypoints[-1]
        for i in range(self.current_idx, len(waypoints)):
            if np.hypot(waypoints[i][0]-vessel_pos[0], waypoints[i][1]-vessel_pos[1]) >= self.base_lookahead:
                return waypoints[i]
        return waypoints[-1]

    def reset(self):
        self.current_idx = 0


class LOSController:
    """
    Integral Line-of-Sight (ILOS) Controller.
    
    Reference: Borhaug, E., Pavlov, A. & Pettersen, K.Y. (2008). 
    'Integral LOS control for path following of underactuated marine surface vessels'.
    
    Mathematical Formulation:
        psi_d = chi_p - atan( (y_e + sigma) / Delta )
        
        Where:
        chi_p: Path tangential angle
        y_e:   Cross-track error (distance from path)
        sigma: Integral action (accumulated error)
        Delta: Lookahead distance
    """
    
    def __init__(self, lookahead_distance: float = 8.0, 
                 path_tolerance: float = 4.0,
                 integral_gain: float = 0.5):
        """
        Args:
            lookahead_distance (Delta): Determines convergence rate. 
                                      Too small = Oscillations. Too large = Corner cutting.
                                      Rec: 1.5-2.5x Vessel Length.
            path_tolerance: Radius of acceptance to switch to next waypoint.
            integral_gain: How aggressively to correct drift (0.0 = Standard LOS).
        """
        self.Delta = lookahead_distance
        self.radius = path_tolerance
        self.sigma = 0.0  # Integral accumulator
        self.gamma = integral_gain
        self.current_wp_idx = 0 # Index of the "To" waypoint
        
    def reset(self):
        self.current_wp_idx = 0
        self.sigma = 0.0

    def compute_desired_heading(self, vessel_pos: Tuple[float, float],
                               waypoints: List[Tuple[float, float]],
                               dt: float = 0.1) -> Optional[float]:
        """
        Computes the ILOS heading command.
        """
        # Safety check
        if len(waypoints) < 2:
            return 0.0
            
        # Initialize index to 1 (Segment 0->1) if just starting
        if self.current_wp_idx == 0:
            self.current_wp_idx = 1
        
        # 2. Normal Switching Logic (Circle of Acceptance)
        goal_wp = waypoints[self.current_wp_idx]
        dist_to_goal = np.hypot(goal_wp[0] - vessel_pos[0], goal_wp[1] - vessel_pos[1])
        
        if dist_to_goal < self.radius:
            if self.current_wp_idx < len(waypoints) - 1:
                self.current_wp_idx += 1
                self.sigma = 0.0  # Reset integral on waypoint switch
            else:
                return None # End of path
        
        # 2b. Recovery Logic - If we're very far from current waypoint after collision avoidance
        # Look ahead to see if we're closer to a future waypoint (we may have overshot during avoidance)
        elif dist_to_goal > self.Delta * 3.0:  # Far from target
            # Check if any of the next few waypoints are closer
            for i in range(self.current_wp_idx + 1, min(len(waypoints), self.current_wp_idx + 4)):
                wp = waypoints[i]
                wp_dist = np.hypot(wp[0] - vessel_pos[0], wp[1] - vessel_pos[1])
                # If we find a closer waypoint ahead AND we're within acceptance radius of it
                if wp_dist < dist_to_goal and wp_dist < self.radius * 1.5:
                    self.current_wp_idx = i
                    self.sigma = 0.0
                    dist_to_goal = wp_dist
                    goal_wp = wp
                    break
        
        # 3. Define the Current Path Segment
        p_prev = waypoints[self.current_wp_idx - 1]
        p_curr = waypoints[self.current_wp_idx]
        
        # 4. Vector Algebra
        # Path Vector components
        alpha_x = p_curr[0] - p_prev[0]
        alpha_y = p_curr[1] - p_prev[1]
        path_len = np.hypot(alpha_x, alpha_y)
        
        if path_len < 0.001: return 0.0 # Safety for duplicate waypoints
        
        # Path Angle (chi_p)
        chi_p = np.arctan2(alpha_y, alpha_x)
        
        # Vessel Vector relative to start of segment
        beta_x = vessel_pos[0] - p_prev[0]
        beta_y = vessel_pos[1] - p_prev[1]
        
        # 5. Calculate Cross-Track Error (y_e)
        # Ideally: Cross product of Unit Path Vector and Vessel Vector
        # y_e = -(x_v - x_p)*sin(chi) + (y_v - y_p)*cos(chi)
        y_e = -(beta_x * np.sin(chi_p)) + (beta_y * np.cos(chi_p))
        
        # 6. Update Integral Term (sigma)
        # Formula: d(sigma)/dt = (Delta * y_e) / (y_e^2 + Delta^2)
        # Damping factor ensures we don't integrate too fast when error is huge
        if abs(y_e) < 15.0: # Anti-windup: Only integrate if we are reasonably close
            scaling_factor = self.Delta / (y_e**2 + self.Delta**2)
            self.sigma += (scaling_factor * y_e * self.gamma) * dt
            
        # Hard clamp on integral to prevent spirals
        self.sigma = np.clip(self.sigma, -np.pi/4, np.pi/4)
        
        # 7. Compute Desired Heading (psi_d)
        # ILOS Law: psi_d = chi_p - arctan( (y_e + sigma) / Delta )
        lookahead_angle = np.arctan((y_e + self.sigma) / self.Delta)
        psi_d = chi_p - lookahead_angle
        
        return PathUtils.normalize_angle(psi_d)

    def get_target_point(self, vessel_pos, waypoints):
        """
        Calculates the 'Virtual Target' for visualization.
        For LOS, this is the point on the path + lookahead.
        """
        # This is an approximation for visualization only
        if self.current_wp_idx == 0: idx = 1
        else: idx = self.current_wp_idx
            
        p_prev = waypoints[idx - 1]
        p_curr = waypoints[idx]
        
        # Project vessel onto line
        ap = np.array([vessel_pos[0] - p_prev[0], vessel_pos[1] - p_prev[1]])
        ab = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
        
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq == 0: return p_curr
        
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        
        closest = (p_prev[0] + t*ab[0], p_prev[1] + t*ab[1])
        
        # Add lookahead vector
        angle = np.arctan2(ab[1], ab[0])
        return (closest[0] + self.Delta * np.cos(angle), 
                closest[1] + self.Delta * np.sin(angle))