#!/usr/bin/env python3
"""Visualize a generated training sample in PyBullet."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import time

import numpy as np
import pybullet as p
import pybullet_data
import tkinter as tk
from tkinter import filedialog


def load_sample(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def render_sample(data: dict) -> None:
    grid = data['map']
    res = grid.shape[0]
    cell = 0.05
    half = res * cell / 2.0

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf')
    # Remove the default axis frame (if present)
    for i in range(p.getNumBodies()):
        info = p.getBodyInfo(i)
        if info and b'axis' in info[1]:
            p.removeBody(i)

    # Wall (very tall, gray)
    wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell * 3])
    wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell * 3], rgbaColor=[0.6, 0.6, 0.6, 1])
    # Rectangle (short, blue)
    rect_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cell / 2, cell, cell / 4])
    rect_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 2, cell, cell / 4], rgbaColor=[0.2, 0.2, 1, 1])
    # Triangle (short, green)
    tri_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell / 4])
    tri_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell / 4], rgbaColor=[0.2, 1, 0.2, 1])
    # U-shape (short, yellow)
    u_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell / 4])
    u_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell / 4], rgbaColor=[1, 1, 0.2, 1])
    # L-shape (short, magenta)
    l_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell / 4])
    l_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell / 4], rgbaColor=[1, 0, 1, 1])
    # T-shape (short, cyan)
    t_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell / 4])
    t_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 2, cell / 2, cell / 4], rgbaColor=[0, 1, 1, 1])
    # Start/Goal flags: reasonable size (pole + small sphere on top)
    flag_pole_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cell / 3, cell / 3, cell * 8])
    flag_pole_green_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 3, cell / 3, cell * 8], rgbaColor=[0, 1, 0, 1])
    flag_pole_red_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[cell / 3, cell / 3, cell * 8], rgbaColor=[1, 0, 0, 1])
    flag_sphere_green = p.createVisualShape(p.GEOM_SPHERE, radius=cell*2, rgbaColor=[0, 1, 0, 1])
    flag_sphere_red = p.createVisualShape(p.GEOM_SPHERE, radius=cell*2, rgbaColor=[1, 0, 0, 1])
    for y in range(res):
        for x in range(res):
            val = int(grid[y, x])
            pos = [x * cell - half + cell / 2, y * cell - half + cell / 2, cell / 2]
            if val == 1:
                p.createMultiBody(0, wall_col, wall_vis, [pos[0], pos[1], cell * 3])
            elif val == 2:
                p.createMultiBody(0, rect_col, rect_vis, [pos[0], pos[1], cell / 2])
            elif val == 3:
                p.createMultiBody(0, tri_col, tri_vis, [pos[0], pos[1], cell / 2])
            elif val == 5:
                p.createMultiBody(0, u_col, u_vis, [pos[0], pos[1], cell / 2])
            elif val == 6:
                p.createMultiBody(0, l_col, l_vis, [pos[0], pos[1], cell / 2])
            elif val == 7:
                p.createMultiBody(0, t_col, t_vis, [pos[0], pos[1], cell / 2])
            elif val == 8:
                # Start flag: green pole + green sphere
                pole_pos = [pos[0], pos[1], cell * 1.2 + cell / 2]
                sphere_pos = [pos[0], pos[1], cell * 8.75 + cell / 2]
                p.createMultiBody(0, flag_pole_col, flag_pole_green_vis, pole_pos)
                p.createMultiBody(0, -1, flag_sphere_green, sphere_pos)
            elif val == 9:
                # Goal flag: red pole + red sphere
                pole_pos = [pos[0], pos[1], cell * 1.2 + cell / 2]
                sphere_pos = [pos[0], pos[1], cell * 8.75 + cell / 2]
                p.createMultiBody(0, flag_pole_col, flag_pole_red_vis, pole_pos)
                p.createMultiBody(0, -1, flag_sphere_red, sphere_pos)
    # ...existing code...
def main() -> None:
    import os
    parser = argparse.ArgumentParser(description='Visualize a training sample')
    parser.add_argument('--file', type=str, help='Path to .npz sample file')
    parser.add_argument('--dir', type=str, default='data/training_samples', help='Directory with .npz samples')
    args = parser.parse_args()

    def load_and_render(sample_file):
        data = load_sample(Path(sample_file))
        p.resetSimulation()
        render_sample(data)

    if args.file:
        sample_file = args.file
    else:
        sample_file = select_sample_with_dialog(args.dir)

    p.connect(p.GUI)
    # Hide the default axis/GUI overlay for a cleaner view
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    except Exception:
        pass
    load_and_render(sample_file)
    print("Press 'S' to select a new environment, or close the window to exit.")
    pulse_angle = 0
    while p.isConnected():
        keys = p.getKeyboardEvents()
        if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
            sample_file = select_sample_with_dialog(args.dir)
            load_and_render(sample_file)
        # Pulsing effect for spheres (start/goal): scale up/down
        pulse_angle += 0.1
        scale = 1.0 + 0.3 * np.sin(pulse_angle)
        # Find all bodies and scale spheres at z ~ cell*6
        for i in range(p.getNumBodies()):
            pos, _ = p.getBasePositionAndOrientation(i)
            if abs(pos[2] - 0.3) < 0.05 or abs(pos[2] - 0.3*6) < 0.2:  # cell*6 is ~0.3 for cell=0.05
                try:
                    p.resetBasePositionAndOrientation(i, pos, [0,0,0,1])
                    p.changeVisualShape(i, -1, rgbaColor=None, specularColor=None, textureUniqueId=-1, meshScale=[scale,scale,scale])
                except Exception:
                    pass
        p.stepSimulation()
        import time
        time.sleep(0.05)


def select_sample_with_dialog(directory):
    """Open a file picker dialog to select a .npz sample file from the given directory."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=directory,
        title="Select a training sample",
        filetypes=[("NumPy files", "*.npz")]
    )
    root.destroy()
    if not file_path:
        raise RuntimeError("No file selected.")
    return file_path


if __name__ == '__main__':
    main()
