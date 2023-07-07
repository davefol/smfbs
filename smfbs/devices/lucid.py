import ctypes
from typing import Optional, Tuple

import numpy as np
from arena_api.buffer import BufferFactory
from arena_api.system import system

from ..fbwriter import FBWriter

class DeviceNotFoundError(FileNotFoundError):
    pass

def ip_to_int(ipaddress) -> int:
    ipaddress = [int(x) for x in ipaddress.split(".")]
    ipaddress.reverse()
    out = 0
    for i, b in enumerate(ipaddress):
        out |= b << (8 * i)
    return out


class Lucid(FBWriter):
    def __init__(
        self,
        ip_address: str,
        height: Optional[int],
        width: Optional[int],
        offset_y: Optional[int],
        offset_x: Optional[int],
        exposure_us: float,
        frame_rate: float,
        gain: float,
        gamma: float,
        reverse_x: bool = False,
        reverse_y: bool = False,
    ):
        self.ip_address = ip_address
        self.height = height
        self.width = width
        self.offset_y = offset_y
        self.offset_x = offset_x
        self.exposure_us = exposure_us
        self.gain = gain
        self.gamma = gamma
        self.frame_rate = frame_rate
        self.reverse_x = reverse_x
        self.reverse_y = reverse_y

    def initialize(self):
        device_infos = system.device_infos
        if len(device_infos) == 0:
            raise DeviceNotFoundError("Found zero devices through the Arena API. Please check device connections and try again.")

        for device in device_infos:
            if device["ip"] == self.ip_address:
                self.device = system.create_device(device)[0]
                break

        nodemap = self.device.nodemap
        tl_stream_nodemap = self.device.tl_stream_nodemap
        nodes = self.device.nodemap.get_node(
            [
                "Width",
                "Height",
                "OffsetY",
                "OffsetX",
                "ExposureAuto",
                "ExposureTime",
                "Gain",
                "Gamma",
                "GammaEnable",
                "AcquisitionFrameRate",
                "ReverseX",
                "ReverseY",
            ]
        )

        # Set features before streaming.-------------------------------------------
        initial_acquisition_mode = nodemap.get_node("AcquisitionMode").value
        nodemap.get_node("AcquisitionMode").value = "Continuous"
        tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
        tl_stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
        tl_stream_nodemap["StreamPacketResendEnable"].value = True
        self.width = self.width or nodes["Width"].max
        self.height = self.height or nodes["Height"].max
        self.offset_y = self.offset_y or 0
        self.offset_x = self.offset_x or 0
        self.exposure_us = self.exposure_us or 10000
        self.gamma = self.gamma or 1
        self.frame_rate = self.frame_rate or 15
        nodes["Width"].value = self.width
        nodes["Height"].value = self.height
        nodes["OffsetY"].value = self.offset_y
        nodes["OffsetX"].value = self.offset_x
        nodes["ExposureAuto"].value = "Off"
        nodes["ExposureTime"].value = float(self.exposure_us)
        nodes["Gain"].value = float(self.gain)
        nodes["GammaEnable"].value = True 
        nodes["Gamma"].value = float(self.gamma)
        nodes["AcquisitionFrameRate"] = float(self.frame_rate)
        nodes["ReverseX"].value = self.reverse_x
        nodes["ReverseY"].value = self.reverse_y
        self.device.start_stream()

    def update(self, buffer, rot90=False):
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
        ).copy()
        if rot90:
            frame = np.rot90(frame, k=3)
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
