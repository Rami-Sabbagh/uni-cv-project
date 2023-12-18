import os

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.patches as mp

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import total_ordering
from numpy.random import default_rng
from queue import Queue, PriorityQueue

from typing import *
from numpy.typing import NDArray

st.set_page_config('Puzzle Solver', '🧩')

'''
# Puzzle Solver

A university project built using [OpenCV], [Python] and [Streamlit]!

[OpenCV]: https://opencv.org/
[Python]: https://python.org/
[Streamlit]: https://streamlit.io/
'''

#region Utilities

def plot_image(
        img: NDArray, cmap: str | None = None, *,
        title: str | None = None, format: int = cv.COLOR_BGR2RGB
    ):

    ax: plt.Axes
    fig, ax = plt.subplots(dpi=140)
    
    ax.set(xticks=[], yticks=[])

    if title is not None:
        ax.set_title(title)

    if cmap is None:
        ax.imshow(cv.cvtColor(img, format))
    else:
        ax.imshow(img, cmap, vmin=0, vmax=255)
    
    st.write(fig)


def plot_images(images: Sequence[tuple[str, NDArray] | None], columns: int = 3, *,
                cmap: str | None = None, title: str | None = None, format = cv.COLOR_BGR2RGB,
                cell_size: tuple[float, float]=(4., 3.), dpi=80) -> None:
    
    images = list(images)
    rows = (len(images) + columns) // columns
    fig, axs = plt.subplots(rows, columns, figsize=(columns * cell_size[0], rows * cell_size[1]),
                            layout='constrained', dpi=dpi)

    if title is not None:
        fig.suptitle(title)
    
    ax: plt.Axes
    for ax in axs.flat:
        ax.set_visible(False)
    
    for i, entry in enumerate(images):
        if entry is None: continue
        subtitle, img = entry

        ax: plt.Axes = axs.flat[i]
        ax.set(xticks=[], yticks=[], title=subtitle, visible=True)
        if img.shape.count == 2:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        else:
            ax.imshow(cv.cvtColor(img, format))
    
    st.write(fig)

#endregion


#region Load Image

@st.cache_resource(show_spinner=False)
def load_image(file) -> tuple[str, NDArray]:
    if file is None:
        return 'uni-1.jpg', cv.imread('images/uni-1.jpg')
    else:
        return file.name, cv.imdecode(
            np.frombuffer(file.getvalue(), np.uint8),
            cv.IMREAD_COLOR,
        )

image_name, image = load_image(st.file_uploader('Target Image', ['png', 'jpg', 'gif', 'bmp']))
st.image(image, f'Target Image: "{image_name}"', channels='BGR')

#endregion


#region Split Image

'### Grid Split & Shuffle'

@st.cache_data(show_spinner=False)
def split_grid(img: NDArray, rows = 2, columns = 2, *, name: str | None = None) -> tuple[NDArray, NDArray]:
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


col1, col2, _ = st.columns([1, 1, 2])
grid_size = col1.number_input('Rows', 1, None, 2), col2.number_input('Columns', 1, None, 3)
# cell_size = col3.number_input('Cell Width', 1., 10., 4.), col4.number_input('Cell Height', 1., 10., 4.5)

cells, cells_names = split_grid(image, *grid_size, name=image_name)

#endregion
    
#region Shuffle Image
    
'##### Shuffle Seed'

seed = 948135

col1, col2, col3 = st.columns([3, 1, 1])
if col2.button('♻️ Randomize'): seed = int(datetime.now().timestamp())
seed = col1.number_input('Seed', value=seed, label_visibility='collapsed')

rng = default_rng(seed)

@st.cache_data(show_spinner=False)
def shuffle_grid(cells: NDArray) -> NDArray:
    cells_shuffled = cells.flatten()
    rng.shuffle(cells_shuffled)
    return cells_shuffled.reshape(cells.shape)


cells_shuffled = shuffle_grid(cells)

#endregion

#region Merge Image

@st.cache_data(show_spinner=False)
def merge_grid(cells: NDArray) -> NDArray:
    return np.concatenate(list(map(lambda x: np.concatenate(x, 1), cells)))

with st.spinner():
    image_shuffled = merge_grid(cells_shuffled)
    st.image(image_shuffled, f'"{image_name}" after split, shuffle and merge.', channels='BGR')


@st.cache_resource
def encode_image(img: NDArray) -> bytes:
    return cv.imencode('.jpg', img)[1].tobytes()

col3.download_button(
    label='💾 Save Image',
    data=encode_image(image_shuffled),
    file_name=f'{os.path.splitext(image_name)[0]}_shuffled_{seed}.jpg',
    mime='image/jpg',
)

#endregion
