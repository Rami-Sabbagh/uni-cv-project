import cv2 as cv

import numpy as np

from copy import deepcopy
from functools import total_ordering

from utils.plots import Plots
from utils.cell import Direction
from utils.puzzle import Puzzle

from typing import *
from numpy.typing import NDArray


@total_ordering
class State:

    
    def __init__(self) -> None:
        self.puzzle: Puzzle

        self.shape: tuple[int, int]
        self.rows: int
        self.columns: int

        self.min_row: int
        self.max_row: int

        self.min_column: int
        self.max_column: int

        self.cells: dict[int, dict[int, int]]
        self.available_cells: set[int]
        self.free_neighbors: set[tuple[int, int]]

        self.coherence: float
        self.__hash: int


    @property
    def actions(self) -> Sequence[tuple[tuple[int, int], int]]: # (coords, cell_id)
        for neighbor in self.free_neighbors:
            for cell_id in self.available_cells:
                yield neighbor, cell_id


    @classmethod
    def create_initial_state(self, puzzle: Puzzle) -> Self:
        result = self()
        result.puzzle = puzzle

        result.shape = (0, 0)
        result.rows, result.columns = 0, 0

        result.min_row, result.max_row = 0, 0
        result.min_column, result.max_column = 0, 0

        result.cells = dict()
        result.available_cells = set(range(len(puzzle.cells)))
        result.free_neighbors = set([ (0, 0) ])

        result.coherence = .0
        result.__update_hash()
        
        return result


    def is_empty(self) -> bool:
        return self.rows == 0 and self.columns == 0


    def is_complete(self) -> bool:
        return self.shape == self.puzzle.shape and len(self.available_cells) == 0


    def get_cell_id(self, coords: tuple[int, int]) -> int:
        if not coords[0] in self.cells: return -1
        if not coords[1] in self.cells[coords[0]]: return -1
        return self.cells[coords[0]][coords[1]]
    

    def get_cell_coherence(self, coords: tuple[int, int], cell_id: int) -> tuple[float, int]:
        coherence = .0
        neighbors = 0

        for neighbor, direction in [
            ( (coords[0] - 1, coords[1]),  Direction.UP    ),
            ( (coords[0] + 1, coords[1]),  Direction.DOWN  ),
            ( (coords[0], coords[1] - 1),  Direction.LEFT  ),
            ( (coords[0], coords[1] + 1),  Direction.RIGHT ),
        ]:
            other_cell_id = self.get_cell_id(neighbor)
            if other_cell_id == -1: continue

            coherence += self.puzzle.get_coherence(cell_id, other_cell_id, direction)
            neighbors += 1
        
        return coherence, neighbors
    

    def copy(self) -> Self:
        result = State()
        result.puzzle = self.puzzle

        result.shape = self.shape
        result.rows, result.columns = self.rows, self.columns

        result.min_row, result.max_row = self.min_row, self.max_row
        result.min_column, result.max_column = self.min_column, self.max_column

        result.cells = deepcopy(self.cells)
        result.available_cells = self.available_cells.copy()
        result.free_neighbors = self.free_neighbors.copy()

        result.coherence = self.coherence
        result.__hash = self.__hash

        return result


    def toarray(self) -> NDArray:
        result = np.full(self.shape, -1, dtype=np.int32)

        for i, row in enumerate(self.cells.values()):
            for j, cell_id in enumerate(row.values()):
                result[i, j] = cell_id

        return result
    

    def to_cells(self, cells: list[NDArray] | None = None) -> NDArray:
        if cells is None: cells = self.puzzle.cells
        result = np.empty(self.shape, dtype=object)
        null_cell = np.zeros_like(cells[0])

        for i, row in enumerate(self.cells.values()):
            for j, cell_id in enumerate(row.values()):
                result[i, j] = cells[cell_id] if cell_id >= 0 and cell_id < len(cells) else null_cell
        
        return result


    def plot(
            self, *,
            cell_size: tuple[float, float] = (4., 4.5),
            format: int = cv.COLOR_BGR2RGB,
            title: str = 'State',
            dpi: int = 80,
        ) -> None:

        def generate_sequence():
            for i in range(self.min_row, self.min_row + self.rows):
                for j in range(self.min_column, self.min_column + self.columns):
                    if not i in self.cells:
                        yield None
                    elif not j in self.cells[i]:
                        yield None
                    else:
                        yield (f'({i:2d}, {j:2d})', self.puzzle.cells[self.cells[i][j]])
                        
        Plots.images(
            generate_sequence(), self.columns,
            title=title, cell_size=cell_size, format=format, dpi=dpi,
        )


    def apply(self, coords: tuple[int, int], cell_id: int) -> Self:
        if not coords in self.free_neighbors:
            raise Exception('The coordinates are not of a free cell with neighbors.')
        
        if not cell_id in self.available_cells:
            raise Exception('The requested image is not available.')

        if self.__check_overflow(coords):
            raise Exception('Cell out of bounds.')

        result = self.copy()

        result.available_cells.remove(cell_id)
        result.free_neighbors.remove(coords)

        result.__set_cell_id(coords, cell_id)
        result.__update_bounds(coords)

        if result.rows != self.rows or result.columns != self.columns:
            result.__check_neighbors()
        
        result.__add_free_neighbors(coords)

        result.coherence = self.coherence + self.get_cell_coherence(coords, cell_id)[0]
        result.__update_hash()

        return result


    def __set_cell_id(self, coords: tuple[int, int], cell_id: int) -> None:
        if not coords[0] in self.cells:
            self.cells[coords[0]] = dict()
        self.cells[coords[0]][coords[1]] = cell_id


    def __update_bounds(self, coords: tuple[int, int]) -> None:
        row, column = coords

        if row < self.min_row:
            self.min_row = row
            self.rows += 1
        
        if row >= self.max_row:
            self.max_row = row + 1
            self.rows += 1
        
        if column < self.min_column:
            self.min_column = column
            self.columns += 1
        
        if column >= self.max_column:
            self.max_column = column + 1
            self.columns += 1
        
        self.shape = (self.rows, self.columns)


    def __add_free_neighbors(self, coords: tuple[int, int]) -> None:
        for neighbor in [
            (coords[0] - 1, coords[1]),
            (coords[0] + 1, coords[1]),
            (coords[0], coords[1] - 1),
            (coords[0], coords[1] + 1),
        ]:
            if self.__check_overflow(neighbor):
                continue

            if self.get_cell_id(neighbor) == -1:
                self.free_neighbors.add(neighbor)
    

    def __check_neighbors(self) -> None:
        to_remove = list(filter(self.__check_overflow, self.free_neighbors))

        for item in to_remove:
            self.free_neighbors.remove(item)
    

    def __check_overflow(self, coords: tuple[int, int]) -> bool:
        row, column = coords

        if self.rows == self.puzzle.rows:
            if row < self.min_row or row >= self.max_row:
                return True
        
        if self.columns == self.puzzle.columns:
            if column < self.min_column or column >= self.max_column:
                return True
        
        return False
    

    def __update_hash(self) -> None:
        # NOTE: More performant solutions might be possible.
        self.__hash = hash(self.toarray().tobytes())


    def __hash__(self) -> int:
        return self.__hash
    

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)


    def __le__(self, other) -> bool:
        return hash(self) < hash(other)
    