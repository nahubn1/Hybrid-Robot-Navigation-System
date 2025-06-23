from dataclasses import dataclass
import math
from typing import List

import pybullet as p


@dataclass
class DifferentialDriveRobot:
    """Simple differential drive kinematics wrapper for a PyBullet robot."""

    robot_id: int
    wheel_radius: float
    wheel_base: float
    left_speed: float = 0.0
    right_speed: float = 0.0

    def set_wheel_speeds(self, left: float, right: float) -> None:
        self.left_speed = float(left)
        self.right_speed = float(right)

    def step(self, dt: float) -> None:
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        v = 0.5 * self.wheel_radius * (self.left_speed + self.right_speed)
        omega = self.wheel_radius * (self.right_speed - self.left_speed) / self.wheel_base
        new_pos = [
            pos[0] + v * math.cos(yaw) * dt,
            pos[1] + v * math.sin(yaw) * dt,
            pos[2],
        ]
        new_yaw = yaw + omega * dt
        p.resetBasePositionAndOrientation(
            self.robot_id, new_pos, p.getQuaternionFromEuler([0.0, 0.0, new_yaw])
        )
        p.stepSimulation()


@dataclass
class LidarSensor:
    """Simple planar lidar using ray casting."""

    robot_id: int
    range: float = 5.0
    num_rays: int = 36
    z_offset: float = 0.2

    def scan(self) -> List[float]:
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        ray_from, ray_to = [], []
        for i in range(self.num_rays):
            angle = yaw + 2 * math.pi * i / self.num_rays
            start = [pos[0], pos[1], pos[2] + self.z_offset]
            end = [
                pos[0] + self.range * math.cos(angle),
                pos[1] + self.range * math.sin(angle),
                pos[2] + self.z_offset,
            ]
            ray_from.append(start)
            ray_to.append(end)
        results = p.rayTestBatch(ray_from, ray_to)
        distances = [self.range * r[2] if r[0] != -1 else self.range for r in results]
        return distances
