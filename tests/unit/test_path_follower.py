from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC_PATH))

from hybrid_system.path_follower import PathFollower


def test_straight_line_lookahead():
    pf = PathFollower(lookahead_distance=2.0)
    path = [(0.0, 0.0), (10.0, 0.0)]
    goal = pf.get_local_goal((0.0, 0.0), path)
    assert goal == (2.0, 0.0)


def test_off_path_projection():
    pf = PathFollower(lookahead_distance=2.0)
    path = [(0.0, 0.0), (10.0, 0.0)]
    goal = pf.get_local_goal((1.0, 1.0), path)
    assert goal == (3.0, 0.0)


def test_multi_segment_lookahead():
    pf = PathFollower(lookahead_distance=4.0)
    path = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    goal = pf.get_local_goal((2.0, 0.0), path)
    assert goal == (5.0, 1.0)


def test_near_end_returns_goal():
    pf = PathFollower(lookahead_distance=2.0)
    path = [(0.0, 0.0), (10.0, 0.0)]
    goal = pf.get_local_goal((9.5, 0.0), path)
    assert goal == (10.0, 0.0)
