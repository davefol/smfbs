from typing import Tuple

import cv2
import numpy as np

from ..fbwriter import FBWriter


class OpenCV(FBWriter):
    def __init__(self, shape, dtype, index):
        self.index = index
        self._shape = shape
        self._dtype = dtype

    def initialize(self):
        self.cap = cv2.VideoCapture(self.index)

    def update(self, buffer, rot90=False):
        success, frame = self.cap.read()
        if rot90:
            frame = np.rot90(frame)
        if not success:
            raise RuntimeError("Failed to get frame")

        if buffer is None:
            return frame
        else:
            np.copyto(buffer, frame)

    def shape(self):
        return self._shape

    def dtype(self):
        return self._dtype

    def close(self):
        self.cap.release()
