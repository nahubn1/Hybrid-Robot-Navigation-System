# Hybrid Robot Navigation System

This project provides a modular framework for hybrid robot navigation, supporting both simulation and real robot operation. It includes components for sensor integration, path planning, and control algorithms.

## Project Structure
- `sensors/` - Code for integrating various sensors (e.g., LIDAR, camera, IMU)
- `planning/` - Path planning algorithms
- `control/` - Control logic for robot movement
- `simulation/` - Simulation environment setup and scripts
- `robot/` - Real robot interface and drivers
- `tests/` - Unit and integration tests

## Getting Started
1. Ensure you have Python 3.8+ installed.
2. (Optional) Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the main application or simulation as needed.

## Contributing
- Follow modular design principles.
- Write tests for new features.
- Document your code.

## License
MIT License
