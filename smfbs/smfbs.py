import time
from multiprocessing import Process, Queue, Value, shared_memory
from typing import Dict, Optional, Tuple
from turbojpeg import TurboJPEG, TJPF_BGR, TJPF_GRAY, TJSAMP_GRAY, TJSAMP_422, TJPF_BGRA

import numpy as np

from .fbwriter import FBWriter


def _run(fbwriter: FBWriter, name: str, framerate: Value, status: Queue, encode_jpeg=False, rot90=False):
    fbwriter.initialize()
    template = fbwriter.update(None)
    if rot90:
        template = np.rot90(template)
    print(f"creating shared memory {name}, size ({template.nbytes})")
    shm = shared_memory.SharedMemory(name=name, create=True, size=template.nbytes)

    if encode_jpeg:
        jpeg = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
        if len(template.shape) == 2:
            pixel_format = TJPF_GRAY
            jpeg_subsample=TJSAMP_GRAY
        else:
            if template.shape[2] == 1:
                pixel_format = TJPF_GRAY
                jpeg_subsample=TJSAMP_GRAY
            elif template.shape[2] == 3:
                pixel_format = TJPF_BGR
                jpeg_subsample=TJSAMP_422
            elif template.shape[2] == 4:
                pixel_format = TJPF_BGRA
                jpeg_subsample=TJSAMP_422
        shm_jpeg = shared_memory.SharedMemory(name=name+"_JPEG", create=True, size=1000000)
        jpeg_buf = shm_jpeg.buf

    try:
        buffer = np.ndarray(template.shape, template.dtype, buffer=shm.buf)

        status.put(True)
        while True:
            fbwriter.update(buffer, rot90=rot90)
            compressed = jpeg.encode(buffer, pixel_format=pixel_format, jpeg_subsample=jpeg_subsample, quality=70)
            if encode_jpeg:
                jpeg_buf[0:4] = len(compressed).to_bytes(4, "big")
                jpeg_buf[4:4+len(compressed)] = compressed

    finally:
        print("closing shm")
        fbwriter.close()
        shm.close()
        shm.unlink()


class SMFBS:
    def __init__(self):
        self.frame_buffers: Dict[
            str, Tuple[Tuple[int, int, int], np.dtype, Process, np.ndarray, Value]
        ] = dict()

    def add(self, fbwriter: FBWriter, name: Optional[str] = None, rot90=False):
        framerate = Value("d", 0.0)
        status = Queue()
        process = Process(
            target=_run,
            args=(fbwriter, name, framerate, status, True, rot90),
        )
        self.frame_buffers[name] = (
            fbwriter.shape(),
            fbwriter.dtype(),
            process,
            framerate,
            status,
        )

    def start(self):
        for _, _, process, _, status in self.frame_buffers.values():
            process.start()
            status.get()
            print("done")

    def stop(self):
        for _, _, process, _, _ in self.frame_buffers.values():
            process.terminate()
        for _, _, process, _, _ in self.frame_buffers.values():
            process.join()
