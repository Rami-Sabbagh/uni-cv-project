import os

import cv2 as cv
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from time import perf_counter
from numpy.random import default_rng

from utils.cell import Direction, directions
from utils.grid import Grid, GridIdentifier
from utils.image import Image
from utils.puzzle import Puzzle
from utils.state import State
from utils.isa_solver import ISAPuzzleSolver, Solution

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
        return 'uni-1.jpg', cv.imread('images/uni-1.jpg')
    else:
        return file.name, Image.decode(file.getvalue())

image_name, image = load_image(st.file_uploader('Target Image', ['png', 'jpg', 'gif', 'bmp', 'webp']))
st.image(image, f'Target Image: "{image_name}"', channels='BGR')


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

shuffle_enabled = st.toggle('Shuffle Cells', True)

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
states_limit = col1.number_input('States Limit', 1, value=10_000)
best_of = col2.number_input('Best of', 1, value=1)
pyramid_depth = col3.number_input('Pyramid Depth', 0, value=3)
depth_only = st.checkbox('Depth Only Mode', True)

status = st.status('Solving Puzzle...')

with status:
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

    solver = ISAPuzzleSolver(
        puzzle, states_limit=states_limit,
        depth_only=depth_only
    )

    solutions = list(zip(solver, range(best_of)))
    solutions.sort(key = lambda s: s[0].state.coherence)
    solution: Solution = solutions[0][0] if len(solutions) > 0 else None

    end = perf_counter()

    if solution is not None:
        'Rendering Solution...'
        cells_solved = solution.state.to_cells(list(cells_full.flat))
        image_solved = Grid.merge(cells_solved)


if solution is None:
    status.update(state='error')
    st.error('Failed to find solution.')
else:

    st.image(image_solved, f'Generated Solution.', channels='BGR')

    st.download_button(
        label='💾 Save Solution',
        data=Image.encode(image_solved),
        file_name=f'{os.path.splitext(image_name)[0]}_solved.jpg',
        mime='image/jpg',
    )

    with st.expander('Statistics'):
        f"""
        |              Metric | Value                            |
        |--------------------:|----------------------------------|
        | Visited states      | `{len(solver.visited)}`          |
        | States in queue     | `{solver.queue.qsize()}`         |
        | Solution Coherence  | `{solution.state.coherence:.2f}` |
        | Search time         | `{end - start :.2f}s`            |
        | Evaluated Solutions | `{len(solutions)}`               |
        """

    with st.expander('Solution Sequence'):
        st.markdown(' → '.join(map(lambda x: f'`{x[0]} = {x[1]}`', solution.state.history)))

        total_steps = len(solution.state.history)
        col1, _ = st.columns([1, 3])
        step = col1.number_input(f'Step (0-{total_steps})', 0, total_steps, 0) - 1
        if step >= 0:
            sequence: list[State] = list(solution.state.sequence)
            step_image = Grid.merge(sequence[step].to_cells(list(cells_full.flat), solution.state))
            st.image(step_image, f'Solution at step {step}.', channels='bgr')
    

if not solver.queue.empty():
    with st.expander('Solver Queue'):
        limit = st.number_input('Preview Limit', 0, value=500)
        queue: list[tuple[tuple, State]] = []

        while not solver.queue.empty() and len(queue) < limit:
            queue.append(solver.queue.get())

        queue_df = pd.DataFrame(
            map(lambda i: (*i[0], i[1]), queue),
        )

        states = queue_df[len(queue_df.columns)-1]
        del queue_df[len(queue_df.columns)-1]

        queue_df['last_cell'] = states.apply(lambda s: (s.history[-1][1],))
        queue_df['last_coord'] = states.apply(lambda s: s.history[-1][0])
        queue_df['actions_len'] = states.apply(lambda s: len(list(s.actions)))
        queue_df['neighbors'] = states.apply(lambda s: len(s.free_neighbors))
        queue_df['hash'] = states.apply(lambda s: (hash(s),))
        
        st.write(queue_df)


with st.expander('Coherence Matrices'):

    normalize = st.toggle('Normalize along base cell axis.', True)
    
    def coherence_dataframe(direction: Direction) -> pd.DataFrame:
        matrix = puzzle.coherence_matrices[:, :, direction.value[0]]
        x, y = np.meshgrid(range(matrix.shape[0]), range(matrix.shape[1]))

        if normalize:
            min_value = np.min(matrix, axis=0)
            max_value = np.max(matrix, axis=0)

            matrix = (matrix - min_value) / (max_value - min_value)

        return pd.DataFrame({
            'base': x.ravel(),
            'neighbor': y.ravel(),
            'coherence': matrix.ravel(),
        })


    def coherence_chart(direction: Direction) -> alt.Chart:
        df = coherence_dataframe(direction)

        heatmap = alt.Chart(df).mark_rect().encode(
            alt.X('base:O').title('Base Cell'),
            alt.Y('neighbor:O').title('Neighbor Cell'),
            alt.Color('coherence:Q').sort('descending'),
        )

        labels = alt.Chart(df).mark_text().encode(
            alt.X('base:O').title('Base Cell'),
            alt.Y('neighbor:O').title('Neighbor Cell'), 
            alt.Text('coherence:Q', format='.2f').title('Coherence'),
        )

        return (heatmap + labels).properties(
            title=f'Coherence Matrix with Direction {direction.name.title()}',
            width=600, height=550,
        )

    for direction in directions:
        st.write(coherence_chart(direction))
