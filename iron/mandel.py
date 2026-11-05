#!/usr/bin/env python3
"""Mandelbrot on the NPU: one AIE core per column, live zoom in the terminal.

  ../env.sh python mandel.py                 zoom into Seahorse Valley
  ../env.sh python mandel.py --still         one frame, then exit
  ../env.sh python mandel.py --png out.png   one big frame as a PNG (no PIL needed)

The host computes c for each pixel (fixed point, Q24 in an int32), the NPU runs the
iterations and gives back the count per pixel. The coordinates are data, not compile-time
values, so the kernel compiles once and every zoom frame reuses it.
"""

import argparse
import shutil
import struct
import sys
import time
import zlib

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, Out

FRAC = 24  # Q24: precision ~6e-8, so the zoom ends near 1e5x
TILE = 512  # pixels per kernel call
MAXIT = 400

KERNEL = f"""
#include <stdint.h>
extern "C" {{
void mandel(int32_t *cx, int32_t *cy, int32_t *out, int32_t n) {{
    const int64_t four = 4LL << {FRAC};
    for (int i = 0; i < n; i++) {{
        int64_t x = 0, y = 0;
        const int64_t a = cx[i], b = cy[i];
        int k = 0;
        for (; k < {MAXIT}; k++) {{
            int64_t xx = (x * x) >> {FRAC};
            int64_t yy = (y * y) >> {FRAC};
            if (xx + yy > four) break;
            int64_t xy = (x * y) >> {FRAC - 1};
            x = xx - yy + a;
            y = xy + b;
        }}
        out[i] = k;
    }}
}}
}}
"""


@iron.jit
def mandel_npu(cx: In, cy: In, out: Out, *, func: CompileTime[ExternalFunction], num_elements: CompileTime[int]):
    ty = np.ndarray[(num_elements,), np.dtype[np.int32]]
    return iron.algorithms.transform_parallel_binary(func, ty, tile_size=TILE)


def kernel():
    t = np.ndarray[(TILE,), np.dtype[np.int32]]
    return ExternalFunction("mandel", source_string=KERNEL, arg_types=[t, t, t, np.dtype(np.int32)])


def grid(w, h, cx, cy, width, pad_to):
    """c for every pixel in Q24, flattened and padded with points far outside the set.

    Pixels are square: in the terminal each cell is two half-block pixels on top of each other."""
    xs = cx + (np.arange(w) / w - 0.5) * width
    ys = cy + (np.arange(h) / h - 0.5) * width * h / w
    gx, gy = np.meshgrid(xs, ys)
    n = w * h
    total = -(-n // pad_to) * pad_to
    out = []
    for g in (gx, gy):
        a = np.full(total, 4 << FRAC, dtype=np.int32)
        a[:n] = np.round(g.ravel() * (1 << FRAC)).astype(np.int32)
        out.append(a)
    return out, n


def cpu(gx, gy):
    """The same fixed-point maths in numpy, for the race."""
    a, b = gx.astype(np.int64), gy.astype(np.int64)
    x, y = np.zeros_like(a), np.zeros_like(a)
    k = np.full(a.shape, MAXIT, dtype=np.int32)
    alive = np.ones(a.shape, bool)
    four = 4 << FRAC
    for i in range(MAXIT):
        xx, yy = (x * x) >> FRAC, (y * y) >> FRAC
        esc = alive & (xx + yy > four)
        k[esc] = i
        alive &= ~esc
        if not alive.any():
            break
        xy = (x * y) >> (FRAC - 1)
        x = np.where(alive, xx - yy + a, x)
        y = np.where(alive, xy + b, y)
    return k


def colors(k):
    t = np.log1p(k) / np.log1p(MAXIT)  # log scale: most points escape in a few steps
    rgb = np.stack([9 * (1 - t) * t**3, 15 * (1 - t) ** 2 * t**2, 8.5 * (1 - t) ** 3 * t], -1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgb[k >= MAXIT] = 0
    return rgb


def draw(rgb, status):
    out = ["\x1b[H"]
    for r in range(0, rgb.shape[0] - 1, 2):
        top, bot = rgb[r], rgb[r + 1]
        out.append("".join(f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m▀"
                           for a, b in zip(top, bot, strict=True)))
        out.append("\x1b[0m\n")
    out.append(f"\x1b[0m\x1b[K{status}")
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\0" + rgb[r].tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--still", action="store_true")
    p.add_argument("--png")
    p.add_argument("--no-cpu", action="store_true", help="skip the CPU race")
    p.add_argument("--x", type=float, default=-0.743643887037151)
    p.add_argument("--y", type=float, default=0.131825904205330)
    args = p.parse_args()

    dev = iron.get_current_device()
    pad_to = TILE * dev.cols
    func = kernel()

    if args.png:
        w, h = 1920, 1080
        (gx, gy), n = grid(w, h, -0.6, 0.0, 3.2, pad_to)
        k = run(func, gx, gy, n)[0]
        png(args.png, colors(k.reshape(h, w)))
        print(f"{args.png}: {w}x{h} on {dev.cols} NPU columns")
        return

    cols, rows = shutil.get_terminal_size()
    w, h = cols, (rows - 2) * 2
    width, zoom = 3.2, 0.94
    cx, cy = -0.6, 0.0
    sys.stdout.write("\x1b[2J\x1b[?25l")
    try:
        frame = 0
        while width > 2e-5:
            # glide from the whole set to the target, then zoom in
            f = min(1.0, frame / 40)
            px, py = cx + (args.x - cx) * f, cy + (args.y - cy) * f
            (gx, gy), n = grid(w, h, px, py, width, pad_to)
            k, npu_ms = run(func, gx, gy, n)
            status = f"NPU {dev.cols} cores: {npu_ms:6.1f} ms"
            if not args.no_cpu and frame % 10 == 0:
                t0 = time.perf_counter()
                cpu(gx[:n], gy[:n])
                cpu_ms = (time.perf_counter() - t0) * 1000
                status += f" | numpy CPU: {cpu_ms:6.1f} ms ({cpu_ms / npu_ms:.1f}x)"
            status += f" | zoom {3.2 / width:,.0f}x | {w}x{h}"
            draw(colors(k[:n].reshape(h, w)), status)
            if args.still:
                break
            width *= zoom if frame >= 20 else 1
            frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\n")


def run(func, gx, gy, n):
    a = iron.tensor(gx, dtype=np.int32, device="npu")
    b = iron.tensor(gy, dtype=np.int32, device="npu")
    out = iron.zeros(gx.shape, dtype=np.int32, device="npu")
    t0 = time.perf_counter()
    mandel_npu(a, b, out, func=func, num_elements=int(gx.shape[0]))
    ms = (time.perf_counter() - t0) * 1000
    return out.numpy().copy(), ms


if __name__ == "__main__":
    main()
