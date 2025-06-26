#!/usr/bin/env python3
"""Visualize a generated training sample in PyBullet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


def load_sample(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def render_sample(data: dict) -> None:
    grid = data['map']
    res = grid.shape[0]
    cell = 0.05
    half = res * cell / 2.0

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf')

    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cell / 2] * 3)
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 2] * 3, rgbaColor=[0.6, 0.6, 0.6, 1])
    start_vis = p.createVisualShape(p.GEOM_SPHERE, radius=cell / 2, rgbaColor=[0, 1, 0, 1])
    goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=cell / 2, rgbaColor=[1, 0, 0, 1])

    for y in range(res):
        for x in range(res):
            val = int(grid[y, x])
            pos = [x * cell - half + cell / 2, y * cell - half + cell / 2, cell / 2]
            if val == 1:
                p.createMultiBody(0, box_col, box_vis, pos)
            elif val == 2:
                p.createMultiBody(0, -1, start_vis, pos)
            elif val == 3:
                p.createMultiBody(0, -1, goal_vis, pos)

    while p.isConnected():
        p.stepSimulation()


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize a training sample')
    parser.add_argument('file', type=str)
    args = parser.parse_args()

    data = load_sample(Path(args.file))
    render_sample(data)


if __name__ == '__main__':
    main()
