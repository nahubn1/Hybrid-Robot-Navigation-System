from dataclasses import dataclass, field
from typing import List


@dataclass
class RobotState:
    """State of a differential drive robot."""

    x: float
    y: float
    theta: float
    v: float
    omega: float


@dataclass
class RobotConfiguration:
    """Physical and kinematic constraints of the robot."""

    max_speed: float
    min_speed: float
    max_omega: float
    max_accel: float
    max_omega_dot: float
    footprint_radius: float


@dataclass
class Trajectory:
    """Predicted trajectory as a list of states."""

    states: List[RobotState] = field(default_factory=list)
