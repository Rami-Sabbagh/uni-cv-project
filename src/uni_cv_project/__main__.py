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

from utils.grid import Grid
from utils.image import Image

from typing import *
from numpy.typing import NDArray

st.set_page_config('Puzzle Solver', '🧩')
rng = default_rng()

#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'''
# Puzzle Solver

A university project built using [OpenCV], [Python] and [Streamlit]!

[OpenCV]: https://opencv.org/
[Python]: https://python.org/
[Streamlit]: https://streamlit.io/
'''
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
