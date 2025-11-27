# Autonomous Vessel Navigation

A comparative study of classical and reinforcement learning approaches to autonomous maritime navigation.

## Phase 1: Classical Navigation (COMPLETE & TESTED)

Implemented:
- A* pathfinding
- Grid world environment
- Static obstacles
- Path Optimization - String pulling (Line of Sight)
- Vessel physics (Kinematic & Nomoto models)
- Path following (Pure Pursuit & ILOS)
- Dynamic obstacles
- Collision detection (CPA/TCPA)
- Collision avoidance (COLREGs-inspired)

## Phase 2: Reinforcement Learning (IN PROGRESS)

Coming next:
- RL environment wrapper
- DQN/PPO implementation
- Training infrastructure
- Performance comparison


## Setup
```bash
# Install dependencies
uv sync

# Run demos
uv run python examples/navigation_with_avoidance.py
```

## Project Structure

- `src/environment/` - Grid world and simulation environment
- `src/pathfinding/` - A* and other pathfinding algorithms
- `src/vessel/` - Vessel physics models
- `src/visualization/` - Plotting and animation tools
