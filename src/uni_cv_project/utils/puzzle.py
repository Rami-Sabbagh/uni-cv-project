import numpy as np

from utils.cell import Direction, Cell, directions, directions_opposites


from typing import *
from numpy.typing import NDArray


class Puzzle:


    def __init__(self, cells: NDArray):
        self.shape: tuple[int, int] = cells.shape[:2]
        self.rows, self.columns = self.shape

        self.cells: list[NDArray] = list(cells.flat)
        self.indices: list[tuple[int, int]] = list(map(tuple, np.stack(np.indices(cells.shape), -1).reshape(-1, 2)))

        # precomputed coherence scores between cells at different edges. 
        self.coherence_matrices: NDArray = Puzzle.__compute_coherence(self.cells)
    

    def get_coherence(self, cell_a: NDArray, cell_b: NDArray, dir: Direction) -> NDArray:
        return self.coherence_matrices[cell_a][cell_b][dir.value[0]]


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
