import pytest
p = pytest.importorskip("pybullet")
from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from simulation.environment_generator import generate_random_2d_environment


def test_generate_random_2d_environment_count():
    p.connect(p.DIRECT)
    ids = generate_random_2d_environment(num_obstacles=5, area_size=2.0)
    assert len(ids) == 5
    for body_id in ids:
        pos, _ = p.getBasePositionAndOrientation(body_id)
        assert pos[2] > 0
    p.disconnect()
