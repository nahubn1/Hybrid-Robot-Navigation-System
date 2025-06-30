from pathlib import Path
import sys
import random

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from data_generation.office_map_generator import Leaf, flood_fill_connected, generate_office_map
from data_generation.connectivity_graph import build_room_adjacency_graph


def test_build_room_adjacency_graph():
    r1 = Leaf(0, 0, 5, 5, id=0)
    r2 = Leaf(5, 0, 5, 5, id=1)
    graph = build_room_adjacency_graph([r1, r2], min_length=1)
    assert graph.number_of_edges() == 1
    wall = list(graph.edges(data=True))[0][2]['wall']
    assert wall.orientation == 'v'
    assert wall.pos == 5
    assert wall.start == 0 and wall.end == 5


def test_generate_office_map_connectivity():
    cfg = {
        'map_resolution': 50,
        'shell_thickness': 1,
        'bsp': {'min_leaf_size': 15, 'wall_thickness': 2},
        'doors': {'width_range': [2, 2], 'additional_door_probability': 0.0},
        'obstacles': {'count_range_per_room': [0, 0]},
    }
    rng = random.Random(0)
    grid = generate_office_map(cfg, rng)
    assert flood_fill_connected(grid)
