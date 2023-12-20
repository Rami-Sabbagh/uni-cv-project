import os

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

from time import perf_counter
from numpy.random import default_rng

from utils.grid import Grid, GridIdentifier
from utils.image import Image
from utils.puzzle import Puzzle
from utils.isa_solver import ISAPuzzleSolver

from typing import *
from numpy.typing import NDArray

st.set_page_config('Grid ISA Solver', '🔍')
rng = default_rng()


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
"""
# 🔍 Grid ISA Solver

This tool uses an intelligent search algorithm to solve
a shuffled grid puzzle.

It takes a rectangular image as input, detects the grid shape
automatically, splits it into cells and searches for a solution.

To get started please select the target image you wish.
"""
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

@st.cache_resource(show_spinner=False)
def load_image(file) -> tuple[str, NDArray]:
    if file is None:
        return 'uni-1_02x03_shuffled_948135.jpg', cv.imread('images/uni-1_02x03_shuffled_948135.jpg')
    else:
        return file.name, Image.decode(file.getvalue())

image_name, image = load_image(st.file_uploader('Target Image', ['png', 'jpg', 'gif', 'bmp']))
st.image(image, f'Target Image: "{image_name}"', channels='BGR')


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'## Grid Shape'
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

col1, col2 = st.columns([1, 3])
if col2.toggle('Automatic Detection', True):
    std_threshold = col1.number_input('$\sigma$ Threshold', 1, value=4)
    widest_gap = col1.number_input('Widest Gap', 10, value=50)

    with st.spinner('Detecting shape...'):
        rows, columns = GridIdentifier.identify_grid_shape(
            image, std_threshold=std_threshold, widest_gap=widest_gap,
        )

else:
    rows = col1.number_input('Rows', 1, None, 2)
    columns = col1.number_input('Columns', 1, None, 3)


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

shuffle_enabled = st.toggle('Shuffle Cells')

if shuffle_enabled:
    col1, col2 = st.columns([3, 1])
    seed = col1.number_input(
        'Seed', label_visibility='collapsed',
        value=rng.integers(1e9, 1e10) if col2.button('♻️ Randomize') else 948135, 
    )


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'## Puzzle Solution'
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

col1, col2, col3, _ = st.columns([1, 1, 1, 2])
states_limit = col1.number_input('States Limit', 500, value=10_000)
best_of = col2.number_input('Best of', 1, value=1)
pyramid_depth = col3.number_input('Pyramid Depth', 0, value=3)


with st.status('Solving Puzzle...'):
    'Splitting Image...'
    cells_full = Grid.split(image, rows, columns)[0]

    if shuffle_enabled:
        'Shuffling Cells...'
        cells_full = Grid.shuffle(cells_full, seed)

    'Scaling Down...'
    cells = Grid.pyramid_down(cells_full, pyramid_depth)

    st.image(
        Grid.merge(cells),
        'Cells after shuffling (if enabled) and scale down.',
        channels='bgr',
    )

    'Initializing Puzzle...'
    puzzle = Puzzle(cells)

    'Searching for Solution...'
    start = perf_counter()
    solver = ISAPuzzleSolver(puzzle, states_limit=states_limit)
    solutions = list(zip(solver, range(best_of)))
    solutions.sort(key = lambda s: s[0].state.coherence)
    solution = solutions[0][0]
    end = perf_counter()

    'Rendering Solution...'
    cells_solved = solution.state.to_cells(list(cells_full.flat))
    image_solved = Grid.merge(cells_solved)


st.image(image_solved, f'Generated Solution.', channels='BGR')

with st.expander('Statistics'):
    f"""
    |              Metric | Value                            |
    |--------------------:|----------------------------------|
    | Visited states      | `{len(solver.visited)}`          |
    | States in queue     | `{solver.queue.qsize()}`         |
    | Solution Coherence  | `{solution.state.coherence:.2f}` |
    | Search time         | `{end - start :.2f}ms`           |
    | Evaluated Solutions | `{len(solutions)}` |
    """

st.download_button(
    label='💾 Save Solution',
    data=Image.encode(image_solved),
    file_name=f'{os.path.splitext(image_name)[0]}_solved.jpg',
    mime='image/jpg',
)
