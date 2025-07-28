from __future__ import annotations

import math
from typing import List, Tuple


class PathFollower:
    """Compute a short-term goal along a global path."""

    def __init__(self, lookahead_distance: float) -> None:
        if lookahead_distance <= 0:
            raise ValueError("lookahead_distance must be positive")
        self.lookahead_distance = float(lookahead_distance)

    def get_local_goal(
        self, position: Tuple[float, float], path: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Return a point ``lookahead_distance`` ahead along ``path``."""
        if not path:
            raise ValueError("path must contain at least one point")
        if len(path) == 1:
            return tuple(float(v) for v in path[0])

        # Precompute segment and cumulative lengths
        seg_lengths: List[float] = []
        cumulative: List[float] = [0.0]
        for p1, p2 in zip(path[:-1], path[1:]):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            seg_lengths.append(length)
            cumulative.append(cumulative[-1] + length)

        # Find closest projection of the robot onto the path
        best_dist2 = math.inf
        proj_len = 0.0
        for i, (p1, p2) in enumerate(zip(path[:-1], path[1:])):
            vx = p2[0] - p1[0]
            vy = p2[1] - p1[1]
            seg_len2 = vx * vx + vy * vy
            if seg_len2 == 0:
                continue
            wx = position[0] - p1[0]
            wy = position[1] - p1[1]
            t = (wx * vx + wy * vy) / seg_len2
            t = max(0.0, min(1.0, t))
            proj_x = p1[0] + t * vx
            proj_y = p1[1] + t * vy
            dist2 = (position[0] - proj_x) ** 2 + (position[1] - proj_y) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                proj_len = cumulative[i] + seg_lengths[i] * t

        target_dist = proj_len + self.lookahead_distance
        total_length = cumulative[-1]
        if target_dist >= total_length:
            return tuple(float(v) for v in path[-1])

        for i, (p1, p2) in enumerate(zip(path[:-1], path[1:])):
            if target_dist <= cumulative[i + 1]:
                seg_len = seg_lengths[i]
                if seg_len == 0:
                    return tuple(float(v) for v in p2)
                ratio = (target_dist - cumulative[i]) / seg_len
                gx = p1[0] + ratio * (p2[0] - p1[0])
                gy = p1[1] + ratio * (p2[1] - p1[1])
                return (gx, gy)

        return tuple(float(v) for v in path[-1])
