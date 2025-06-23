import numpy as np
from dataclasses import dataclass, field

@dataclass
class ProbabilityMap:
    width: int
    height: int
    resolution: float
    default_value: float = 0.0
    grid: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        if self.grid is None:
            self.grid = np.full((self.height, self.width), self.default_value, dtype=float)
        else:
            self.grid = self.grid.astype(float)

    def get_cell(self, x: int, y: int) -> float:
        return float(self.grid[y, x])

    def set_cell(self, x: int, y: int, value: float) -> None:
        self.grid[y, x] = float(value)

    def reset(self, value: float = 0.0) -> None:
        self.grid.fill(float(value))

    def to_occupancy_grid(self, threshold: float = 0.5) -> 'OccupancyGrid':
        from .occupancy_grid import OccupancyGrid
        occ = (self.grid >= threshold).astype(int)
        grid = OccupancyGrid(self.width, self.height, self.resolution)
        grid.grid = occ
        return grid
