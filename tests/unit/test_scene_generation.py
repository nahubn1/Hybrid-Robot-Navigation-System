from pathlib import Path
import sys

import pytest
p = pytest.importorskip("pybullet")
import pybullet_data

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))
ASSET_PATH = Path(__file__).resolve().parents[2] / 'assets'

from data_generation.pybullet_scene_generator import (
    create_cluttered_scene,
    create_room_scene,
    create_maze_scene,
)


def test_create_cluttered_scene():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf')
    cube = ASSET_PATH / 'cube.urdf'
    ids = create_cluttered_scene(obstacle_count=3, area_size=2.0, urdf_path=str(cube), seed=1)
    assert len(ids) == 3
    p.disconnect()


def test_create_room_scene():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf')
    wall = ASSET_PATH / 'wall.urdf'
    ids = create_room_scene(num_rooms=1, area_size=2.0, urdf_path=str(wall), seed=2)
    assert len(ids) > 0
    p.disconnect()


def test_create_maze_scene():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf')
    wall = ASSET_PATH / 'wall.urdf'
    ids = create_maze_scene(grid_size=2, passage_width=1.0, urdf_path=str(wall), seed=3)
    assert len(ids) > 0
    p.disconnect()
