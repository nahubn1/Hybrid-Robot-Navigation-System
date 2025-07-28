from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC_PATH))

from hybrid_system import Navigator, NavigatorState


class DummyPlanner:
    pass


def test_navigator_set_goal_changes_state():
    nav = Navigator(DummyPlanner(), DummyPlanner())
    assert nav.state == NavigatorState.IDLE
    nav.set_goal((1.0, 2.0))
    assert nav.state == NavigatorState.PLANNING
