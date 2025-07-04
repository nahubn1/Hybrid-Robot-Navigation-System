from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(SRC_PATH))

from scripts.data_generation.generate_ground_truth import densify_path
from utils.graph_helpers import bresenham_line


def test_densify_path_matches_bresenham():
    nodes = [(0, 0), (3, 1)]
    densified = densify_path(nodes, step=1.0)
    expected = [(int(x), int(y)) for x, y in bresenham_line(0, 0, 3, 1)]
    assert densified == expected
    assert len(densified) == len(expected)
