# Autonomous Vessel Navigation

A comparative study of pathfinding algorithms for autonomous vessel navigation.

## Project Phases

- **Phase 1**: Static obstacle avoidance with classical algorithms (A*)
- **Phase 2**: Dynamic obstacle avoidance
- **Phase 3**: Multiple vessels with COLREGs compliance
- **Phase 4**: Environmental factors (weather, currents)

## Setup
```bash
# Install dependencies
uv sync

# Run example
uv run python examples/basic_navigation.py
```

## Project Structure

- `src/environment/` - Grid world and simulation environment
- `src/pathfinding/` - A* and other pathfinding algorithms
- `src/vessel/` - Vessel physics models
- `src/visualization/` - Plotting and animation tools
