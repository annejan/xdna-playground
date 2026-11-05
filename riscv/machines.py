"""32 mini RISC-V machines: their RAM, their programs, and the host interpreter (bin/rv_host.so)."""

import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAM, FB, FB_BYTES, MAIL, CPU = 0x8000, 0x5000, 7680, 0x7E00, 0x7F00
W, H = 160, 96
PROGRAMS = ["plasma", "mandel", "life", "stars"]
FRAME, BUDGET, HALT, BAD = 1, 2, 3, 4
# 16 colours (the PICO-8 palette: made for tiny screens)
PALETTE = np.array([(0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81), (171, 82, 54), (95, 87, 79),
                    (194, 195, 199), (255, 241, 232), (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
                    (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)], np.uint8)


def program(name):
    with open(os.path.join(HERE, "bin", name + ".bin"), "rb") as f:
        return f.read()


def boot(n=32):
    """RAM for n machines: machine k runs PROGRAMS[k % 4] with id k."""
    rams = np.zeros((n, RAM), np.uint8)
    for k in range(n):
        code = program(PROGRAMS[k % len(PROGRAMS)])
        rams[k, :len(code)] = np.frombuffer(code, np.uint8)
        rams[k, MAIL + 4:MAIL + 8] = np.frombuffer(np.uint32(k).tobytes(), np.uint8)
    return rams


def screen(ram):
    """The framebuffer as (H, W) colour indices."""
    fb = ram[FB:FB + FB_BYTES]
    px = np.empty(W * H, np.uint8)
    px[0::2], px[1::2] = fb & 15, fb >> 4
    return px.reshape(H, W)


class Host:
    """The interpreter on the CPU: the reference for the NPU."""

    def __init__(self):
        self.lib = ctypes.CDLL(os.path.join(HERE, "bin", "rv_host.so"))

    def frame(self, ram, frame_no, budget=50_000_000):
        ram[MAIL:MAIL + 4] = np.frombuffer(np.uint32(frame_no).tobytes(), np.uint8)
        ran = ctypes.c_uint32()
        why = self.lib.rv_frame(ram.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), budget, ctypes.byref(ran))
        return why, ran.value
