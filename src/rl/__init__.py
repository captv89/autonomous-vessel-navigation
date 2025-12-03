"""
Reinforcement Learning Module for Autonomous Vessel Navigation

This module provides Gymnasium-compatible environments for training
RL agents to navigate through static and dynamic obstacles.

Learning Progression:
    Phase 1: Static obstacles only
    Phase 2: Add dynamic obstacles (current)
    Phase 3: Multi-vessel scenarios
"""

from .rl_environment import VesselNavigationEnv

__all__ = ['VesselNavigationEnv']
