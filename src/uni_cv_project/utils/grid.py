import cv2 as cv
import numpy as np
import streamlit as st

from numpy.random import default_rng

from typing import *
from numpy.typing import NDArray


class Grid:


    @staticmethod
    @st.cache_data(show_spinner=False)
    def split(
        img: NDArray, rows = 2, columns = 2, *,
        name: str | None = None,
    ) -> tuple[NDArray, NDArray]:
        
        cell_height, cell_width = img.shape[0] // rows, img.shape[1] // columns
        img = cv.resize(img, (cell_width * columns, cell_height * rows), interpolation=cv.INTER_LINEAR)

        rows_steps = np.arange(0, rows + 1, dtype=np.uint) * cell_height
        columns_steps = np.arange(0, columns + 1, dtype=np.uint) * cell_width

        cells = np.empty((rows, columns), dtype=object)
        for i in range(rows):
            for j in range(columns):
                cells[i, j] = img[
                    rows_steps[i]:rows_steps[i+1],
                    columns_steps[j]:columns_steps[j+1],
                ]
        
        prefix = '' if name is None else f'{name} • '
        cells_coords = np.stack(np.meshgrid(np.arange(columns), np.arange(rows)))
        cells_names = np.apply_along_axis(
            lambda x: f'{prefix}({x[1]:3d}, {x[0]:3d})',
            0, cells_coords)
        
        return cells, cells_names


    @staticmethod
    @st.cache_data(show_spinner=False)
    def shuffle(cells: NDArray, seed: int | None = 948135) -> NDArray:
        cells_shuffled = cells.flatten()
        default_rng(seed).shuffle(cells_shuffled)
        return cells_shuffled.reshape(cells.shape)


    @staticmethod
    @st.cache_data(show_spinner=False)
    def merge(cells: NDArray) -> NDArray:
        return np.concatenate(list(map(lambda x: np.concatenate(x, 1), cells)))
