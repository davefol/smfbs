import time
from multiprocessing import shared_memory

import cv2
import numpy as np
from flask import Flask, Response
from waitress import serve

from .smfbs import SMFBS


class FrameServer:
    def __init__(self, smfbs: SMFBS, port=8123, threads=20):
        self.smfbs = smfbs
        self.app = Flask(__name__)
        self.initialize_routes()
        serve(self.app, host="0.0.0.0", port=port, threads=threads)

    def serve_frames(self, name: str, encoding: str) -> Response:
        def yield_frames(name: str, encoding: str):
            shm = shared_memory.SharedMemory(name=name)
            shape, dtype, _, framerate, _ = self.smfbs.frame_buffers[name]
            buffer = np.ndarray(shape, dtype, buffer=shm.buf)
            while True:
                #time.sleep(1 / framerate.value)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + cv2.imencode(encoding, buffer)[1].tobytes()
                    + b"\r\n"
                )

        return Response(
            yield_frames(name, encoding),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def index(self):
        return "\n".join([f"{name}" for name in self.smfbs.frame_buffers.keys()])

    def initialize_routes(self):
        self.app.route("/")(self.index)
        self.app.route("/<name>/<encoding>")(self.serve_frames)
