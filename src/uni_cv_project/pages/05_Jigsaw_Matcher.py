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

from typing import *
from numpy.typing import NDArray

st.set_page_config('Jigsaw Matcher', '🧩')
rng = default_rng()


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
"""
# 🧩 Jigsaw Matcher

This tool takes a image of a jigsaw puzzle, a reference image
and find the correct places of the pieces within the reference image.

To get started please select the target image you wish.
"""
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

@st.cache_resource(show_spinner=False)
def load_image(file) -> tuple[str, NDArray]:
    if file is None:
        return None, None
    else:
        return file.name, Image.decode(file.getvalue())

image_name, image = load_image(st.file_uploader('Target Image', ['png', 'jpg', 'gif', 'bmp', 'webp']))
hint_name, hint = load_image(st.file_uploader('Hint Image', ['png', 'jpg', 'gif', 'bmp', 'webp']))



#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'### Input Preview'
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

col1, col2 = st.columns(2)

if image is not None:
    col1.image(image, f'Target Grid: "{image_name}"', channels='BGR')
if hint is not None:
    col2.image(hint, f'Hint Image: "{hint_name}"', channels='BGR')

if image is None or hint is None:
    st.stop()


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'## Matching Results'
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#



