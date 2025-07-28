from pathlib import Path
import sys

import numpy as np
import pytest

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from dwa_planner.data_structures import RobotConfiguration, RobotState
from dwa_planner.kinematics import predict_trajectory
from dwa_planner.planner import DynamicWindowPlanner


def test_predict_trajectory_straight_line():
    state = RobotState(0.0, 0.0, 0.0, 0.0, 0.0)
    traj = predict_trajectory(state, (1.0, 0.0), dt=1.0, steps=3)
    assert len(traj.states) == 3
    assert traj.states[-1].x == pytest.approx(3.0)
    assert traj.states[-1].y == pytest.approx(0.0)


def test_dynamic_window_planner_plan_basic():
    cfg = RobotConfiguration(
        max_speed=1.0,
        min_speed=0.0,
        max_omega=1.0,
        max_accel=0.5,
        max_omega_dot=1.0,
        footprint_radius=0.5,
    )
    planner = DynamicWindowPlanner(cfg, {'goal_dist_cost_gain': 1.0})
    state = RobotState(0.0, 0.0, 0.0, 0.0, 0.0)
    goal = (2.0, 0.0)
    obstacle_map = np.zeros((10, 10), dtype=int)
    v, w = planner.plan(state, goal, obstacle_map, dt=1.0, predict_steps=3)
    assert v >= 0.0
    assert abs(w) <= cfg.max_omega
