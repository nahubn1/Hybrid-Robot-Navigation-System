import numpy as np
from dataclasses import dataclass, field

@dataclass
class OccupancyGrid:
    width: int
    height: int
    resolution: float
    default_value: int = 0
    grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grid = np.full((self.height, self.width), self.default_value, dtype=int)

    def get_cell(self, x: int, y: int) -> int:
        return int(self.grid[y, x])

    def set_cell(self, x: int, y: int, value: int) -> None:
        self.grid[y, x] = int(value)

    def reset(self, value: int = 0) -> None:
        self.grid.fill(int(value))

    def to_probability_map(self) -> 'ProbabilityMap':
        from .probability_map import ProbabilityMap
        prob = self.grid.astype(float)
        prob[prob != 0] = 1.0
        return ProbabilityMap(self.width, self.height, self.resolution, grid=prob)
