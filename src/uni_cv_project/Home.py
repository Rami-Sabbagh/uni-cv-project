import os

import cv2 as cv
import numpy as np
import streamlit as st

from numpy.random import default_rng

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

'To get started select the tool you wish from the sidebar ⬅️'