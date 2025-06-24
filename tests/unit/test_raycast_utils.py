from pathlib import Path
import sys

import numpy as np
import pybullet as p
import pybullet_data

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))
ASSET_PATH = Path(__file__).resolve().parents[2] / 'assets'

from data_generation.raycast_utils import generate_occupancy_grid


def test_generate_occupancy_grid_basic():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf')
    cube = ASSET_PATH / 'cube.urdf'
    p.loadURDF(str(cube), [0, 0, 0.25], globalScaling=0.5)
    grid = generate_occupancy_grid(area_size=2.0, resolution=10, height=2.0)
    assert grid.shape == (10, 10)
    assert np.any(grid == 1)
    p.disconnect()
