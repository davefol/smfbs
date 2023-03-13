import ctypes
from typing import Optional, Tuple

import cv2
import numpy as np
from arena_api.buffer import BufferFactory
from arena_api.system import system

from ..fbwriter import FBWriter


def ip_to_int(ipaddress) -> int:
    ipaddress = [int(x) for x in ipaddress.split(".")]
    ipaddress.reverse()
    out = 0
    for i, b in enumerate(ipaddress):
        out |= b << (8 * i)
    return out


class Lucid(FBWriter):
    def __init__(self, ip_address: str, height: Optional[int], width: Optional[int]):
        self.ip_address = ip_address
        self.height = height
        self.width = width

    def initialize(self):
        devices = system.create_device()
        for device in devices:
            if device.nodemap["GevCurrentIPAddress"].value == ip_to_int(self.ip_address):
                self.device = device
            else:
                system.destroy_device(device)
        nodemap = self.device.nodemap
        tl_stream_nodemap = self.device.tl_stream_nodemap
        nodes = self.device.nodemap.get_node(["Width", "Height"])

        # Set features before streaming.-------------------------------------------
        initial_acquisition_mode = nodemap.get_node("AcquisitionMode").value
        nodemap.get_node("AcquisitionMode").value = "Continuous"
        tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
        tl_stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
        tl_stream_nodemap["StreamPacketResendEnable"].value = True
        self.width = self.width or nodes["Width"].max
        self.height = self.height or nodes["Height"].max
        nodes["Width"].value = self.width
        nodes["Height"].value = self.height
        self.device.start_stream()

    def update(self, buffer):
        d_buffer = self.device.get_buffer()
        item = BufferFactory.copy(d_buffer)
        self.device.requeue_buffer(d_buffer)
        buffer_bytes_per_pixel = int(len(item.data) / (item.width * item.height))
        array = (ctypes.c_ubyte * item.width * item.height).from_address(
            ctypes.addressof(item.pbytes)
        )
        frame = np.ndarray(
            buffer=array,
            dtype=np.uint8,
            shape=(item.height, item.width, buffer_bytes_per_pixel),
        )
        BufferFactory.destroy(item)

        if buffer is None:
            return frame
        else:
            np.copyto(buffer, frame)

    def shape(self):
        return (self.height, self.width)

    def dtype(self):
        return np.dtype("B")

    def close(self):
        system.destroy_device(self.device)
