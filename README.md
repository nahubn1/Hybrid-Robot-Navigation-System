# Hybrid-Robot-Navigation-System
This project aims to develop a hybrid robot navigation system by integrating deep learning for perception and global planning with traditional local planning methods, focusing on robust and efficient autonomous navigation.

## Simulation Setup

This project uses [PyBullet](https://pybullet.org/) for a lightweight simulation
environment. To run the demo simulation:

```bash
# Install Python dependencies
pip install -r environment/requirements.txt

# Launch the example simulator
python scripts/simulation/run_simulation.py
```

The sample script spawns a plane and an R2D2 robot model to validate that the
simulation environment is properly configured.
