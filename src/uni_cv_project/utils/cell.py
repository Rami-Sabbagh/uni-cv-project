import numpy as np

from enum import Enum

from typing import *
from numpy.typing import NDArray

class Direction(Enum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3,


directions: list[Direction] = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

directions_edges: dict[Direction, tuple[slice | int, slice | int]] = {
    Direction.UP: (0, slice(None)),
    Direction.DOWN: (-1, slice(None)),
    Direction.LEFT: (slice(None), 0),
    Direction.RIGHT: (slice(None), -1),
}

directions_opposites: dict[Direction, Direction] = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

class Cell:
    
    @staticmethod
    def edge(cell: NDArray, direction: Direction) -> NDArray:
        return cell[directions_edges[direction]]


    @staticmethod
    def edge_coherence(base: NDArray, neighbor: NDArray, dir: Direction) -> float:
        base_edge, neighbor_edge = Cell.edge(base, dir), Cell.edge(neighbor, directions_opposites[dir])
        return np.average(Cell.__euclidean_distance(base_edge, neighbor_edge), -1)


    @staticmethod
    def __euclidean_distance(a: NDArray, b: NDArray) -> NDArray:
        return np.sqrt(np.sum(np.square(a - b), -1))