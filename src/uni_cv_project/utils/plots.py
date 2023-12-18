import cv2 as cv
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from typing import *
from numpy.typing import NDArray


class Plots:


    @staticmethod
    def image(
        img: NDArray, cmap: str | None = None, *,
        title: str | None = None, format: int = cv.COLOR_BGR2RGB
    ) -> None:
        
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
    

    @staticmethod
    def plot_images(
        images: Sequence[tuple[str, NDArray] | None], columns: int = 3, *,
        cmap: str | None = None, title: str | None = None, format = cv.COLOR_BGR2RGB,
        cell_size: tuple[float, float]=(4., 3.), dpi=80
    ) -> None:
    
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
