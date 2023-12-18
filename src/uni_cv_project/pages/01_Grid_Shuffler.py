import os

import cv2 as cv
import numpy as np
import streamlit as st

from numpy.random import default_rng

from utils.grid import Grid
from utils.image import Image

from typing import *
from numpy.typing import NDArray

st.set_page_config('Grid Shuffler', '🔀')
rng = default_rng()

#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
"""
# Grid Shuffler

This tool takes a rectangular image,
divides it into a grid of cells
and shuffles them.

To get started please select the target image you wish
and then configure how you wish the operation done.
"""
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

@st.cache_resource(show_spinner=False)
def load_image(file) -> tuple[str, NDArray]:
    if file is None:
        return 'uni-1.jpg', cv.imread('images/uni-1.jpg')
    else:
        return file.name, Image.decode(file.getvalue())

image_name, image = load_image(st.file_uploader('Target Image', ['png', 'jpg', 'gif', 'bmp']))
st.image(image, f'Target Image: "{image_name}"', channels='BGR')


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'### Grid Split & Shuffle'
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

col1, col2, _ = st.columns([1, 1, 2])
grid_size = col1.number_input('Rows', 1, None, 2), col2.number_input('Columns', 1, None, 3)

cells, cells_names = Grid.split(image, *grid_size, name=image_name)


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'##### Shuffle Seed' 
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

col1, col2, col3 = st.columns([3, 1, 1])
seed = col1.number_input(
    'Seed', label_visibility='collapsed',
    value=rng.integers(1e9, 1e10) if col2.button('♻️ Randomize') else 948135, 
)

cells_shuffled = Grid.shuffle(cells, seed)


with st.spinner():
    image_shuffled = Grid.merge(cells_shuffled)
    st.image(image_shuffled, f'"{image_name}" after split, shuffle and merge.', channels='BGR')


col3.download_button(
    label='💾 Save Image',
    data=Image.encode(image_shuffled),
    file_name=f'{os.path.splitext(image_name)[0]}_{grid_size[0]:02d}x{grid_size[1]:02d}_shuffled_{seed}.jpg',
    mime='image/jpg',
)

