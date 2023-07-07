import time
from multiprocessing import shared_memory

import cv2
import numpy as np
from flask import Flask, Response
from waitress import serve

from turbojpeg import TurboJPEG, TJPF_BGR, TJPF_GRAY, TJSAMP_GRAY, TJSAMP_422, TJPF_BGRA
from .smfbs import SMFBS


class FrameServer:
    def __init__(self, smfbs: SMFBS, port=8123, threads=20):
        self.smfbs = smfbs
        self.app = Flask(__name__)
        self.initialize_routes()
        serve(self.app, host="0.0.0.0", port=port, threads=threads)

    def serve_frames(self, name: str, encoding: str) -> Response:
        if encoding == "info":
            shm = shared_memory.SharedMemory(name=name)
            shape, dtype, _, framerate, _ = self.smfbs.frame_buffers[name]
            return {
                "framerate": framerate.value,
                "shape": shape,
                "dtype": np.dtype(dtype).char,
            }

        def yield_frames(name: str, encoding: str):
            jpeg = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
            shm = shared_memory.SharedMemory(name=name+"_JPEG")
            buffer = shm.buf

            while True:
                current_time = time.time()
                size = int.from_bytes(buffer[:4], "big")
                img = buffer[4:4+size]
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    #+ cv2.imencode(encoding, buffer)[1].tobytes()
                    #+ jpeg.encode(buffer, pixel_format=pixel_format, jpeg_subsample=jpeg_subsample, quality=70)
                    + img
                    + b"END\r\n"
                )
                time.sleep(max((1/15) - (time.time() - current_time), 0))


        return Response(
            yield_frames(name, encoding),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def index(self):
        return "\n".join([f"{name}" for name in self.smfbs.frame_buffers.keys()])

    def initialize_routes(self):
        self.app.route("/")(self.index)
        self.app.route("/<name>/<encoding>")(self.serve_frames)
