import numpy as np

from utils.cell import Direction, Cell, directions, directions_opposites


from typing import *
from numpy.typing import NDArray


class Puzzle:
    shape: tuple[int, int]
    rows: int
    columns: int

    cells: list[NDArray]
    indices: list[tuple[int, int]]

    __max_cell_id: int

    # precomputed coherence scores between cells at different edges. 
    __coherence: NDArray

    def __init__(self, cells: NDArray):
        self.shape = cells.shape[:2]
        self.rows, self.columns = self.shape

        self.cells = list(cells.flat)
        self.__max_cell_id = len(self.cells) - 1
        self.indices = list(map(tuple, np.stack(np.indices(cells.shape), -1).reshape(-1, 2)))
        self.__coherence = Puzzle.__compute_coherence(self.cells)
    

    def get_coherence(self, cell_a: NDArray, cell_b: NDArray, dir: Direction) -> NDArray:
        return self.__coherence[cell_a][cell_b][dir.value[0]]


    @staticmethod
    def __compute_coherence(cells: list[NDArray]) -> NDArray:
        return np.array([
            [
                [
                    Cell.edge_coherence(cells[i], cells[j], dir)
                    for dir in directions
                ]
                for j in range(len(cells))
            ]
            for i in range(len(cells))
        ])
