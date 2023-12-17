import numpy as np
import pandas as pd
import streamlit as st

"""
# Puzzle Solver

A university project built using [OpenCV], [Python] and [Streamlit]!

[OpenCV]: https://opencv.org/
[Python]: https://python.org/
[Streamlit]: https://streamlit.io/
"""

x = st.slider('x')

f'{x} squared is {x ** 2}!'


import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'