# Hybrid-Robot-Navigation-System
This project aims to develop a hybrid robot navigation system by integrating deep learning for perception and global planning with traditional local planning methods, focusing on robust and efficient autonomous navigation.

## Simulation Setup

This project uses [PyBullet](https://pybullet.org/) for a lightweight simulation
environment. To run the demo simulation:

```bash
# Install dependencies
bash environment/setup.sh

# Launch the example simulator
python scripts/simulation/run_simulation.py
```

The sample script spawns a plane and an R2D2 robot model to validate that the
simulation environment is properly configured.

## Mapping Utilities

The `mapping` package contains simple grid data structures useful for testing navigation algorithms.

```python
from mapping.occupancy_grid import OccupancyGrid

# Create a 10x10 grid with 0.1m resolution
grid = OccupancyGrid(width=10, height=10, resolution=0.1)

# Mark a cell as occupied
grid.set_cell(5, 5, 1)

# Convert to a probability map
prob_map = grid.to_probability_map()
print(prob_map.get_cell(5, 5))  # 1.0
```


## Dataset Generation

A script is provided to automatically generate synthetic occupancy grids for training. It builds random scenes in PyBullet and performs batched ray casting to save `.npy` grids with metadata.

```bash
python scripts/data_generation/generate_pybullet_envs.py --output-dir data/raw --num-scenes 50 --archetype clutter --resolution 128
```

