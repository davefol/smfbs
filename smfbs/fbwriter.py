from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np


class FBWriter(ABC):
    @abstractmethod
    def initialize(self):
        """
        Runs before first frame buffer update
        """
        pass

    @abstractmethod
    def update(self, buffer) -> Optional[np.ndarray]:
        """
        Called to update the frame buffer. If buffer is None,
        returns the data that would have been copied into buffer.
        """
        pass

    @abstractmethod
    def shape(self) -> Tuple:
        """
        Returns ndarray shape of the buffer
        """

    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Returns ndarray dtype of the buffer
        """

    @abstractmethod
    def close(self) -> np.dtype:
        """
        Called when the stream is closed
        """
