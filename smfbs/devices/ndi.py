import numpy as np
import NDIlib as ndi
from ..fbwriter import FBWriter

class NDI(FBWriter):
    def __init__(self, ip_address, height=1080, width=1920):
        self.ip_address = ip_address
        self.height = height
        self.width = width

    def initialize(self):
        if not ndi.initialize():
            raise RuntimeError("Failed to initialize NDI")

        finder = ndi.FindCreate(p_extra_ips=self.ip_address)

        ndi_find = ndi.find_create_v2(finder)

        if ndi_find is None:
            raise RuntimeError("Could not find the NDI device")

        sources = []
        while not len(sources) > 0:
            ndi.find_wait_for_sources(ndi_find, 1000)
            sources = ndi.find_get_current_sources(ndi_find)

        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA

        self.ndi_recv = ndi.recv_create_v3(ndi_recv_create)

        if self.ndi_recv is None:
            raise RuntimeError("Could not create the v3 recv handle for NDI")

        ndi.recv_connect(self.ndi_recv, sources[0])

        ndi.find_destroy(ndi_find)


    def update(self, buffer, rot90=False):
        while True:
            t, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 5000)
            if t == ndi.FRAME_TYPE_VIDEO:
                if buffer is None:
                    buffer = np.copy(v.data)
                    ndi.recv_free_video_v2(self.ndi_recv, v)
                    return buffer
                else:
                    np.copyto(buffer, v.data)
                    ndi.recv_free_video_v2(self.ndi_recv, v)
                    return

    def shape(self):
        return (self.height, self.width, 4)

    def dtype(self):
        return np.dtype("B")

    def close(self):
        ndi.recv_destroy(self.ndi_recv)
        ndi.destroy()