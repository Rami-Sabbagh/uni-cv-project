import skimage

import cv2 as cv
import numpy as np

from scipy.spatial import KDTree

from utils.cell import Direction, directions, directions_opposites

from typing import *
from numpy.typing import NDArray


class PieceFeatureExtractorAbstract:
    
    def __init__(self, /, *, features=None):
        self._features = features

    def __add__(self, other: Self) -> Self:
        features = np.concatenate(
            [self.features, other.features], 
            axis=1)

        return PieceFeatureExtractorAbstract(features=features)

    @property
    def features(self):
        return self._features

    def __getitem__(self, key):
        return self.features[key]


class PieceFeatureExtractorSIFT(PieceFeatureExtractorAbstract):

    def __init__(self, *, reducers=[np.max, np.mean]): 
        super().__init__()

        self._sift: cv.SIFT = cv.SIFT_create()
        self._reducers = reducers

    def fit(self, pieces: list[NDArray]):
        pieces_kps = self._sift.detect(pieces, None)

        self.pieces_sift = [
            np.array([[kp.angle, kp.response, kp.octave] for kp in piece_kps])
            for piece_kps in pieces_kps
        ]

        _features = []
        for reducer in self._reducers:
            pieces_features = np.array([
                reducer(piece_sift, axis=0) 
                if piece_sift.size != 0
                else np.zeros(3)
                for piece_sift in self.pieces_sift
            ])
            _features.append(pieces_features)

        self._features = np.concatenate(_features, axis=1)

        return self


class PieceFeatureExtractor2DPooling(PieceFeatureExtractorAbstract):

    def __init__(self, /, *, reducers=[np.max, np.mean]):
        super().__init__()
        self._reducers = reducers
        
    def fit(self, pieces):
        features = []

        pieces = np.array(pieces)
        for reducer in self._reducers:
            pieces_features = np.array(list(map(
                lambda p: skimage.measure.block_reduce(p, (91, 91), reducer),
                pieces,
            ))).reshape(pieces.shape[0], -1)
            features.append(pieces_features)

        self._features = np.concatenate(features, axis=1)
        return self


class PieceMatcher:

    def fit(self, pieces_features):
        self.kdtree = KDTree(pieces_features)
        return self


    def get(self, piece_features):
        piece = self.kdtree.query(piece_features)
        return piece



class PuzzleSolverRectHint:

    def __init__(self,) -> None:
        pass


    def fit(self, cells: NDArray) -> None:
        self._pieces_shape = cells.shape

        cells_flat = list(cells.flat)
        features = (PieceFeatureExtractorSIFT().fit(cells_flat) +
                    PieceFeatureExtractor2DPooling().fit(cells_flat))

        self._matcher = PieceMatcher().fit(features.features)


    def solve(self, cells: NDArray) -> tuple[list[int], list[int]]:
        pieces = list(cells.flat)

        features = (PieceFeatureExtractorSIFT().fit(pieces) +
                    PieceFeatureExtractor2DPooling().fit(pieces))
        
        forward_solution = list(range(len(pieces)))
        reverse_solution = list(range(len(pieces)))

        for i, f in enumerate(features.features):
            j = self._matcher.get(f)[1]
            forward_solution[j] = i
            reverse_solution[i] = j

        return forward_solution, reverse_solution