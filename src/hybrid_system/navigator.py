from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Any


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
    state: NavigatorState = NavigatorState.IDLE
    goal: Optional[Tuple[float, float]] = None
    global_path: List[Tuple[float, float]] = field(default_factory=list)

    def set_goal(self, goal: Tuple[float, float]) -> None:
        """Assign a new goal and transition to the planning state."""
        self.goal = tuple(float(v) for v in goal)
        self.global_path.clear()
        self.state = NavigatorState.PLANNING

    def tick(self) -> Tuple[float, float]:
        """Advance the state machine and return a velocity command."""
        if self.state == NavigatorState.IDLE:
            return 0.0, 0.0
        if self.state == NavigatorState.PLANNING:
            # Placeholder for future global planning logic
            return 0.0, 0.0
        if self.state == NavigatorState.NAVIGATING:
            # Placeholder for future navigation logic
            return 0.0, 0.0
        if self.state == NavigatorState.RE_PLANNING:
            # Placeholder for future re-planning logic
            return 0.0, 0.0
        if self.state == NavigatorState.RECOVERY:
            # Placeholder for future recovery behavior
            return 0.0, 0.0
        if self.state == NavigatorState.GOAL_REACHED:
            return 0.0, 0.0
        if self.state == NavigatorState.FAILED:
            return 0.0, 0.0
        return 0.0, 0.0
