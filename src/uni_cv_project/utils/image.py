import cv2 as cv
import numpy as np

from typing import *
from numpy.typing import NDArray


class Image:

    
    @staticmethod
    def decode(raw: bytes) -> NDArray:
        return cv.imdecode(
            np.frombuffer(raw, np.uint8),
            cv.IMREAD_COLOR,
        )
    

    @staticmethod
    def encode(img: NDArray, extension: str = '.jpg') -> bytes:
        return cv.imencode(extension, img)[1].tobytes()