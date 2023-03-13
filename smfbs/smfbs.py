import time
from multiprocessing import Process, Value, shared_memory
from typing import Dict, Tuple, Optional

import numpy as np

from .fbwriter import FBWriter


def _run(
    fbwriter: FBWriter,
    name: str,
    framerate: Value,
):
    fbwriter.initialize()
    template = fbwriter.update(None)
    shm = shared_memory.SharedMemory(name=name, create=True, size=template.nbytes)
    try:
        buffer = np.ndarray(template.shape, template.dtype, buffer=shm.buf)

        while True:
            tic = time.time()
            fbwriter.update(buffer)
            toc = time.time() - tic
            try:
                framerate.value = 1 / toc
            except ZeroDivisionError:
                pass
    finally:
        fbwriter.close()
        shm.close()
        shm.unlink()


class SMFBS:
    def __init__(self):
        self.frame_buffers: Dict[
            str, Tuple[Tuple[int, int, int], np.dtype, Process, np.ndarray, Value]
        ] = dict()

    def add(self, fbwriter: FBWriter, name: Optional[str] = None):
        framerate = Value("d", 0.0)
        process = Process(
            target=_run,
            args=(fbwriter, name, framerate),
        )
        self.frame_buffers[name] = (
            fbwriter.shape(),
            fbwriter.dtype(),
            process,
            framerate,
        )

    def start(self):
        for _, _, process, _ in self.frame_buffers.values():
            process.start()

    def stop(self):
        for _, _, process, _ in self.frame_buffers.values():
            process.terminate()
        for _, _, process, _ in self.frame_buffers.values():
            process.join()
