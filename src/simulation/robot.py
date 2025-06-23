from dataclasses import dataclass
import math
from typing import List

import pybullet as p


@dataclass
class HolonomicRobot:
    """Simple holonomic kinematics wrapper for a PyBullet robot."""

    robot_id: int
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    def set_velocity(self, vx: float, vy: float, yaw_rate: float) -> None:
        self.vx = float(vx)
        self.vy = float(vy)
        self.yaw_rate = float(yaw_rate)

    def step(self, dt: float) -> None:
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        dx = (self.vx * math.cos(yaw) - self.vy * math.sin(yaw)) * dt
        dy = (self.vx * math.sin(yaw) + self.vy * math.cos(yaw)) * dt
        new_pos = [
            pos[0] + dx,
            pos[1] + dy,
            pos[2],
        ]
        new_yaw = yaw + self.yaw_rate * dt
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
