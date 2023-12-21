import os

import cv2 as cv
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from time import perf_counter
from numpy.random import default_rng

from utils.grid import Grid, GridIdentifier
from utils.image import Image
from utils.matcher import PuzzleSolverRectHint

from typing import *
from numpy.typing import NDArray

st.set_page_config('Grid Matcher', '🔣')
rng = default_rng()


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
"""
# 🔣 Grid Matcher

This tool reorders a grid of rectangular pieces to match a reference image
as closest as possible.

To get started please select the target image you wish.
"""
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

@st.cache_resource(show_spinner=False)
def load_image(file, default = 'uni-1.jpg') -> tuple[str, NDArray]:
    if file is None:
        return default, cv.imread('images/' + default)
    else:
        return file.name, Image.decode(file.getvalue())

image_name, image = load_image(st.file_uploader('Target Image', ['png', 'jpg', 'gif', 'bmp', 'webp']))
# st.image(image, f'Target Image: "{image_name}"', channels='BGR')

hint_file = st.file_uploader('Hint Image', ['png', 'jpg', 'gif', 'bmp', 'webp'])
if hint_file is None: hint_name, hint = image_name, image
else: hint_name, hint = load_image(hint_file)

#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'## Grid Shape'
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

col1, col2 = st.columns([1, 3])
if col2.toggle('Automatic Detection', False):
    std_threshold = col1.number_input('$\sigma$ Threshold', 1, value=4)
    widest_gap = col1.number_input('Widest Gap', 10, value=50)

    with st.spinner('Detecting shape...'):
        rows, columns = GridIdentifier.identify_grid_shape(
            image, std_threshold=std_threshold, widest_gap=widest_gap,
        )

else:
    rows = col1.number_input('Rows', 1, None, 5)
    columns = col1.number_input('Columns', 1, None, 5)


col2.write(pd.DataFrame({
    'Rows': {
        'Grid': rows,
        'Cell': image.shape[0] // rows,
    },
    'Columns': {
        'Grid': columns,
        'Cell': image.shape[1] // columns,
    },
}))

shuffle_enabled = st.toggle('Shuffle Cells', True)

if shuffle_enabled:
    col1, col2 = st.columns([3, 1])
    seed = col1.number_input(
        'Seed', label_visibility='collapsed',
        value=rng.integers(1e9, 1e10) if col2.button('♻️ Randomize') else 948135, 
    )

#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'### Input Preview'
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#


with st.spinner('Preparing Grid...'):
    cells = Grid.split(image, rows, columns)[0]
    if shuffle_enabled: cells = Grid.shuffle(cells, seed)
    cells_gray = Grid.cvtColor(cells, cv.COLOR_BGR2GRAY)

    hint_cells = Grid.split(hint, rows, columns)[0]
    hint_cells_gray = Grid.cvtColor(hint_cells, cv.COLOR_BGR2GRAY)
    

col1, col2 = st.columns(2)

col1.image(Grid.merge(cells), f'Target Grid: "{image_name}"', channels='BGR')
col2.image(hint, f'Hint Image: "{hint_name}"', channels='BGR')


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'## Matching Results'
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

matcher = PuzzleSolverRectHint()

with st.spinner('Matching...'):
    matcher.fit(hint_cells_gray)
    forward_solution, reverse_solution = matcher.solve(cells_gray)

    cells_list = list(cells.flat)
    solution_cells = np.empty(len(forward_solution), dtype=object)

    for i, cell_id in enumerate(forward_solution):
        solution_cells[i] = cells_list[cell_id]

    solution_cells = solution_cells.reshape(cells.shape)
    solution_image = Grid.merge(solution_cells)

st.image(solution_image, 'Matched Solution', channels='BGR')


def render_solution_map():
    cell_rows, cell_columns = cells[0, 0].shape[:2]

    result = np.concatenate([
        Grid.merge(cells),
        cv.resize(hint, (cell_columns * columns, cell_rows * rows), interpolation=cv.INTER_LINEAR),
    ], axis=1)


    hint_coords: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns):
            hint_coords.append((
                int((column + .5) * cell_columns) + image.shape[1],
                int((row + .5) * cell_rows),
            ))
    
    id_grid = np.array(reverse_solution).reshape(cells.shape)
    
    palette = [
        [255, 0  , 0  ],
        [0  , 255, 0  ],
        [0  , 0  , 255],
        [255, 255, 0  ],
        [255, 0  , 255],
        [0  , 255, 255],
    ]

    arrows = np.zeros_like(result)
    
    for row in range(rows):
        for column in range(columns):
            point_from = (
                int((column + .5) * cell_columns),
                int((row + .5) * cell_rows),
            )

            to_cell_id = id_grid[row, column]
            point_to = hint_coords[to_cell_id]

            arrows = cv.arrowedLine(
                arrows, point_from, point_to,
                palette[(row + column) % len(palette)],
                10, tipLength=.01,
            )

    opacity = .8
    return cv.addWeighted(result, 1. - opacity, arrows, opacity, .0)


st.image(render_solution_map(), 'Solution Map', channels='bgr')
