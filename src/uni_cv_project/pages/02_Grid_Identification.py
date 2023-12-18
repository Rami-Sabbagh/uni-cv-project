import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

from utils.grid import Grid, GridIdentifier
from utils.image import Image

from typing import *
from numpy.typing import NDArray

st.set_page_config('Grid Identification', '📐')

#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
"""
# 📐 Grid Identification

This tool takes a rectangular image,
which contains a shuffled grid of uniform rectangular pieces
and **detects the shape of the grid (rows and columns count)**.

To get started please select the target image you wish.
"""
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

@st.cache_resource(show_spinner=False)
def load_image(file) -> tuple[str, NDArray]:
    if file is None:
        return 'uni-1_05x04_shuffled_948135.jpg', cv.imread('images/uni-1_05x04_shuffled_948135.jpg')
    else:
        return file.name, Image.decode(file.getvalue())

image_name, image = load_image(st.file_uploader('Target Image', ['png', 'jpg', 'gif', 'bmp']))
st.image(image, f'Target Image: "{image_name}"', channels='BGR')


col1, col2, _ = st.columns([1, 1, 2])
std_threshold = col1.number_input('$\sigma$ Threshold', 1, value=4)
widest_gap = col2.number_input('Widest Gap', 10, value=50)


#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#
'## Detected Shape'
#==---==---==---==---==---==---==---==---==---==---==---==---==---==---==---==#

with st.spinner('Detecting shape...'):
    grid_shape = GridIdentifier.identify_grid_shape(image,
                    std_threshold=std_threshold, widest_gap=widest_gap)
    
    st.write(pd.DataFrame({
        'Rows': { image_name: grid_shape[0] },
        'Columns': { image_name: grid_shape[1] },
    }))


with st.expander('Computed Signals'):
    vertical_signal, horizontal_signal = GridIdentifier.derive_edges_signal(image)

    vertical_cleaned = GridIdentifier.cleanup_edge_signal(vertical_signal,
                        std_threshold=std_threshold, widest_gap=widest_gap)
    
    horizontal_cleaned = GridIdentifier.cleanup_edge_signal(horizontal_signal,
                        std_threshold=std_threshold, widest_gap=widest_gap)

    '##### Horizontal Signal'
    st.line_chart(horizontal_signal)

    '##### Horizontal Edges'
    st.line_chart(horizontal_cleaned)

    '##### Vertical Signal'
    st.line_chart(vertical_signal)
    
    '##### Vertical Edges'
    st.line_chart(vertical_cleaned)


