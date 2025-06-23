from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from mapping.occupancy_grid import OccupancyGrid
from mapping.probability_map import ProbabilityMap


def test_occupancy_grid_init_and_set():
    grid = OccupancyGrid(width=3, height=2, resolution=1.0, default_value=0)
    assert grid.grid.shape == (2, 3)
    assert grid.get_cell(1, 1) == 0
    grid.set_cell(1, 1, 1)
    assert grid.get_cell(1, 1) == 1


def test_probability_map_conversion():
    occ = OccupancyGrid(width=2, height=2, resolution=0.5)
    occ.set_cell(0, 0, 1)
    pmap = occ.to_probability_map()
    assert pmap.get_cell(0, 0) == 1.0
    assert pmap.get_cell(1, 1) == 0.0
    occ2 = pmap.to_occupancy_grid()
    assert occ2.get_cell(0, 0) == 1
    assert occ2.get_cell(1, 1) == 0
