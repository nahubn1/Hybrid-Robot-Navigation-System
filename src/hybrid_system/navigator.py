from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import List, Tuple, Optional, Any

import numpy as np

from .path_follower import PathFollower
from dwa_planner import RobotState


class NavigatorState(Enum):
    """Enumeration of the Navigator's possible states."""

    IDLE = auto()
    PLANNING = auto()
    NAVIGATING = auto()
    RE_PLANNING = auto()
    RECOVERY = auto()
    GOAL_REACHED = auto()
    FAILED = auto()


@dataclass
class Navigator:
    """Finite state machine managing global and local planners."""

    global_planner: Any
    local_planner: Any
    path_follower: PathFollower
    static_map: Optional[np.ndarray] = None
    goal_tolerance: float = 0.5
    recovery_steps: int = 20
    state: NavigatorState = NavigatorState.IDLE
    goal: Optional[Tuple[float, float]] = None
    global_path: List[Tuple[float, float]] = field(default_factory=list)
    _recovery_count: int = 0

    def set_goal(self, goal: Tuple[float, float]) -> None:
        """Assign a new goal and transition to the planning state."""
        self.goal = tuple(float(v) for v in goal)
        self.global_path.clear()
        self._recovery_count = 0
        self.state = NavigatorState.PLANNING

    def tick(
        self,
        current_state: Optional[RobotState] = None,
        local_map: Optional[np.ndarray] = None,
        static_map: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Advance the state machine and return a velocity command."""
        if self.state == NavigatorState.IDLE:
            return 0.0, 0.0

        if self.state in (NavigatorState.PLANNING, NavigatorState.RE_PLANNING):
            if self.goal is None or current_state is None or static_map is None:
                return 0.0, 0.0
            start = (int(round(current_state.x)), int(round(current_state.y)))
            goal = (int(round(self.goal[0])), int(round(self.goal[1])))
            clearance = getattr(self.local_planner, "config", None)
            if clearance is not None:
                clearance = getattr(clearance, "footprint_radius", 0.0)
            else:
                clearance = 0.0
            path = self.global_planner.plan(start, goal, static_map, clearance)
            if path:
                self.global_path = [(float(x), float(y)) for x, y in path]
                self.state = NavigatorState.NAVIGATING
            else:
                self.state = NavigatorState.FAILED
            return 0.0, 0.0

        if self.state == NavigatorState.NAVIGATING:
            if (self.goal is None or current_state is None or local_map is None):
                return 0.0, 0.0
            dx = self.goal[0] - current_state.x
            dy = self.goal[1] - current_state.y
            if math.hypot(dx, dy) <= self.goal_tolerance:
                self.state = NavigatorState.GOAL_REACHED
                return 0.0, 0.0

            local_goal = self.path_follower.get_local_goal(
                (current_state.x, current_state.y), self.global_path
            )
            lgx = int(round(local_goal[0]))
            lgy = int(round(local_goal[1]))
            if (
                0 <= lgy < local_map.shape[0]
                and 0 <= lgx < local_map.shape[1]
                and local_map[lgy, lgx] != 0
            ):
                self.state = NavigatorState.RE_PLANNING
                return 0.0, 0.0

            cmd = self.local_planner.plan(current_state, local_goal, local_map)
            if cmd is None:
                self.state = NavigatorState.RECOVERY
                self._recovery_count = self.recovery_steps
                return 0.0, 0.0
            return cmd

        if self.state == NavigatorState.RECOVERY:
            if self._recovery_count > 0:
                self._recovery_count -= 1
                return 0.0, 0.5
            self.state = NavigatorState.RE_PLANNING
            return 0.0, 0.0

        if self.state == NavigatorState.GOAL_REACHED:
            return 0.0, 0.0

        if self.state == NavigatorState.FAILED:
            return 0.0, 0.0

        return 0.0, 0.0
