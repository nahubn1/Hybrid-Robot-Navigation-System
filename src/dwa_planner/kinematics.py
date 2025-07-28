from typing import Tuple
import numpy as np

from .data_structures import RobotState, Trajectory


def predict_trajectory(
    start: RobotState, command: Tuple[float, float], dt: float, steps: int
) -> Trajectory:
    """Predict future states from a starting state using a simple
    differential drive model."""

    v, omega = command
    x, y, theta = start.x, start.y, start.theta
    traj_states = []
    for _ in range(steps):
        theta += omega * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        traj_states.append(RobotState(x, y, theta, v, omega))
    return Trajectory(traj_states)
