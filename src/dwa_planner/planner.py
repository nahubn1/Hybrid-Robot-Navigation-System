from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from utils.image_processing import distance_transform
from .data_structures import RobotConfiguration, RobotState, Trajectory
from .kinematics import predict_trajectory


class DynamicWindowPlanner:
    def __init__(self, config: RobotConfiguration, cost_gains: Dict[str, float]):
        self.config = config
        self.gains = {
            'goal_dist_cost_gain': 1.0,
            'obstacle_cost_gain': 1.0,
            'velocity_cost_gain': 1.0,
        }
        self.gains.update(cost_gains)

    def _calculate_dynamic_window(self, state: RobotState, dt: float) -> Dict[str, float]:
        v_min = max(self.config.min_speed, state.v - self.config.max_accel * dt)
        v_max = min(self.config.max_speed, state.v + self.config.max_accel * dt)
        omega_min = max(-self.config.max_omega, state.omega - self.config.max_omega_dot * dt)
        omega_max = min(self.config.max_omega, state.omega + self.config.max_omega_dot * dt)
        return {
            'v_min': v_min,
            'v_max': v_max,
            'omega_min': omega_min,
            'omega_max': omega_max,
        }

    def _sample_velocities(self, window: Dict[str, float], num_v: int = 5, num_w: int = 5) -> List[Tuple[float, float]]:
        vs = np.linspace(window['v_min'], window['v_max'], num_v)
        ws = np.linspace(window['omega_min'], window['omega_max'], num_w)
        return [(v, w) for v in vs for w in ws]

    def _min_distance_to_obstacle(self, traj: Trajectory, dist_map: np.ndarray) -> float:
        min_dist = math.inf
        for s in traj.states:
            x = int(round(s.x))
            y = int(round(s.y))
            if 0 <= y < dist_map.shape[0] and 0 <= x < dist_map.shape[1]:
                dist = dist_map[y, x]
            else:
                dist = 0.0
            min_dist = min(min_dist, dist)
        return min_dist

    def _score_trajectory(
        self, traj: Trajectory, goal: Tuple[float, float], dist_map: np.ndarray
    ) -> float:
        final = traj.states[-1]
        goal_dist = math.hypot(goal[0] - final.x, goal[1] - final.y)
        if dist_map is not None:
            obstacle_dist = self._min_distance_to_obstacle(traj, dist_map) - self.config.footprint_radius
            if obstacle_dist < 0:
                obstacle_cost = math.inf
            else:
                obstacle_cost = 1.0 / (obstacle_dist + 1e-5)
        else:
            obstacle_cost = 0.0
        velocity_cost = self.config.max_speed - final.v
        return (
            self.gains['goal_dist_cost_gain'] * goal_dist
            + self.gains['obstacle_cost_gain'] * obstacle_cost
            + self.gains['velocity_cost_gain'] * velocity_cost
        )

    def _is_collision(self, traj: Trajectory, dist_map: np.ndarray) -> bool:
        if dist_map is None:
            return False
        for s in traj.states:
            x = int(round(s.x))
            y = int(round(s.y))
            if 0 <= y < dist_map.shape[0] and 0 <= x < dist_map.shape[1]:
                if dist_map[y, x] <= self.config.footprint_radius:
                    return True
            else:
                return True
        return False

    def plan(
        self,
        current_state: RobotState,
        local_goal: Tuple[float, float],
        obstacle_map: np.ndarray,
        *,
        dt: float = 0.1,
        predict_steps: int = 5,
    ) -> Tuple[float, float]:
        window = self._calculate_dynamic_window(current_state, dt)
        samples = self._sample_velocities(window)
        dist_map = distance_transform(obstacle_map)

        best_traj: Trajectory | None = None
        min_cost = math.inf
        best_cmd = (0.0, 0.0)
        for v, w in samples:
            traj = predict_trajectory(current_state, (v, w), dt, predict_steps)
            if self._is_collision(traj, dist_map):
                continue
            cost = self._score_trajectory(traj, local_goal, dist_map)
            if cost < min_cost:
                min_cost = cost
                best_traj = traj
                best_cmd = (v, w)
        return best_cmd
