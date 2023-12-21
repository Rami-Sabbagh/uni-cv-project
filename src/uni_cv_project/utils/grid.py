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
    

    @staticmethod
    def pyramid_down(cells: NDArray, depth=1) -> NDArray:
        result = np.empty_like(cells)
    
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                cell = cells[i, j]
                for _ in range(depth): cell = cv.pyrDown(cell)
                result[i, j] = cell

        return result

    @staticmethod
    def cvtColor(cells: NDArray, code: int) -> NDArray:
        result = np.empty_like(cells)
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                result[i, j] = cv.cvtColor(cells[i, j], code)
        
        return result



class GridIdentifier:


    @staticmethod
    def identify_grid_shape(
        image: NDArray, *,
        std_threshold: int = 4,
        widest_gap: int = 50,
    ) -> tuple[int, int]:
        
        edges_signals = GridIdentifier.derive_edges_signal(image)
        cleaned_signals = [
            GridIdentifier.cleanup_edge_signal(
                signal, std_threshold=std_threshold,
                widest_gap=widest_gap,
            )
            for signal in edges_signals
        ]

        length_encoding = [
            GridIdentifier.__length_encode(signal > .5)
            for signal in cleaned_signals
        ]

        cell_shape = [
            GridIdentifier.__detect_edge_length(encoding, widest_gap)
            for encoding in length_encoding
        ]

        grid_shape = [
            np.max([int(img_length / cell_length + .5), 1])
            for img_length, cell_length in zip(image.shape, cell_shape)
        ]

        return tuple(grid_shape)

    
    @staticmethod
    def derive_edges_signal(image: NDArray) -> tuple[NDArray, NDArray]:
        """
        Derive the average sobel of the image,
        first vertically then horizontally.
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        sobel = [
            cv.Sobel(gray, cv.CV_32F, dx, dy)
            for dx, dy in [(0, 1), (1, 0)]
        ]

        average = [
            np.average(np.abs(item), axis=axis)
            for item, axis in zip(sobel, [1, 0])
        ]

        diff = [ np.diff(item) for item in average ]

        return tuple(diff)


    @staticmethod
    def cleanup_edge_signal(
        signal: NDArray, *,
        std_threshold: int,
        widest_gap: int,
    ) -> NDArray:
        
        signal = cv.threshold(
            signal,
            GridIdentifier.__standard_deviation(signal) * std_threshold,
            1, cv.THRESH_BINARY,
        )[1]

        signal = cv.morphologyEx(
            signal, cv.MORPH_CLOSE,
            np.ones(widest_gap),
        )

        return signal


    @staticmethod
    def __length_encode(seq: Sequence[bool]) -> Sequence[tuple[bool, int]]:
        last, counter = next(seq.flat), 1

        for element in seq.flat:
            if last == element:
                counter += 1
            else:
                yield last, counter
                last, counter = element, 1
        
        yield last, counter


    @staticmethod    
    def __detect_edge_length(
        seq: Sequence[tuple[bool, int]],
        widest_gap: int,
    ) -> int:
        
        last_padding = 0
        last_length = -1
        min_length = -1

        for value, length in seq:
            if value and last_length != -1:
                padding = length // 2
                edge_length = last_length + last_padding + padding
                if edge_length < min_length and edge_length > widest_gap:
                    min_length = edge_length

                last_padding = length - padding
                last_length = -1

            elif value:
                last_padding = length

            else:
                last_length = length

        if last_length != -1:
            edge_length = last_padding + last_length
            
            if edge_length < min_length and edge_length > widest_gap:
                min_length = edge_length
        
        return edge_length


    @staticmethod
    def __standard_deviation(y: NDArray) -> float:
        mean = np.average(y)
        variance = np.average(np.square(y - mean))
        return np.sqrt(variance)
    


