#!/usr/bin/env python3
"""Mandelbrot on all 32 AIE cores, 16 pixels per vector step, int32 in Q28 (zoom to about 1e6).

  ../env.sh python mandel_chip.py --bench               one 960x540 frame, timed, checked against numpy
  ../env.sh python mandel_chip.py --bench --maxit 4096 --zoom 20000

The host sends c for every pixel (cx and cy in Q28), each core iterates z = z^2 + c on 16 pixels at
a time until all 16 have escaped or MAXIT is reached, and sends back the counts. A pixel that
escapes stays escaped (a sticky mask), so the lanes that run on past it cannot count again.
Products go through the vector multiplier (int32 x int32 into a 64-bit accumulator, >> 28, floor,
saturating), so numpy can follow it bit for bit.

Per column the memtile splits the work over the column's 4 cores, like life_chip.py, without the
row exchange: every pixel stands on its own.
"""

import argparse
import time

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

from life_chip import ROWS, snake

FRAC = 28
P = 1024  # pixels per core per tile: input 2 x 4 KB, output 4 KB, double buffered
I32 = np.iinfo(np.int32)

KERNEL = """
#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" {{
/* c in: [cx x P][cy x P] (Q{frac}); out: the iteration count per pixel */
void mandel_{maxit}(int32_t *in, int32_t *out) {{
    aie::set_rounding(aie::rounding_mode::floor);
    aie::set_saturation(aie::saturation_mode::saturate);
    const auto four = aie::broadcast<int32_t, 16>(4 << {frac});
    const auto zero = aie::zeros<int32_t, 16>();
    const auto one = aie::broadcast<int32_t, 16>(1);
    for (int i = 0; i < {p}; i += 16) {{
        auto a = aie::load_unaligned_v<16>(in + i), b = aie::load_unaligned_v<16>(in + {p} + i);
        auto x = zero, y = zero, n = zero;
        auto alive = aie::eq(zero, zero);  /* all lanes */
        for (int k = 0; k < {maxit}; k++) {{
            auto xx = aie::mul(x, x).template to_vector<int32_t>({frac});
            auto yy = aie::mul(y, y).template to_vector<int32_t>({frac});
            /* xx + yy <= 4, written so it cannot overflow: xx and yy saturate at ~8 */
            alive = alive & aie::le(xx, aie::sub(four, yy));
            if (alive.empty()) break;
            n = aie::add(n, aie::select(zero, one, alive));
            auto xy = aie::mul(x, y).template to_vector<int32_t>({frac} - 1);
            x = aie::add(aie::sub(xx, yy), a);
            y = aie::add(xy, b);
        }}
        aie::store_unaligned_v(out + i, n);
    }}
}}
}}
"""


@iron.jit
def mandel_chip(coords: In, counts: Out, *, tiles: CompileTime[int], maxit: CompileTime[int],
                cols: CompileTime[int]):
    cores = cols * ROWS
    n = cores * tiles * P
    in_ty = np.ndarray[(2 * n,), np.dtype[np.int32]]
    out_ty = np.ndarray[(n,), np.dtype[np.int32]]
    col_in_ty = np.ndarray[(ROWS * 2 * P,), np.dtype[np.int32]]
    col_out_ty = np.ndarray[(ROWS * P,), np.dtype[np.int32]]
    tin_ty = np.ndarray[(2 * P,), np.dtype[np.int32]]
    tout_ty = np.ndarray[(P,), np.dtype[np.int32]]

    kernel = ExternalFunction(f"mandel_{maxit}", source_string=KERNEL.format(frac=FRAC, p=P, maxit=maxit),
                              arg_types=[tin_ty, tout_ty])

    col_in = [ObjectFifo(col_in_ty, name=f"in{c}") for c in range(cols)]
    col_out = [ObjectFifo(col_out_ty, name=f"out{c}") for c in range(cols)]
    t_in, t_out = [], []
    for c in range(cols):
        t_in += col_in[c].cons().split([i * 2 * P for i in range(ROWS)], tile=Tile(c, 1),
                                       obj_types=[tin_ty] * ROWS, names=[f"t_in{c}_{i}" for i in range(ROWS)])
        t_out += col_out[c].prod().join([i * P for i in range(ROWS)], tile=Tile(c, 1),
                                        obj_types=[tout_ty] * ROWS, names=[f"t_out{c}_{i}" for i in range(ROWS)])

    def core(f_in, f_out, kernel):
        for _ in range_(tiles):
            a = f_in.acquire(1)
            b = f_out.acquire(1)
            kernel(a, b)
            f_in.release(1)
            f_out.release(1)

    # the core of chain position k gets split/join index k % ROWS of column k // ROWS
    workers = [Worker(core, [t_in[k].cons(), t_out[k].prod(), kernel], tile=Tile(*snake(k))) for k in range(cores)]

    in_taps = TensorAccessPattern.full((2 * n,)).partition(cols)
    out_taps = TensorAccessPattern.full((n,)).partition(cols)
    in_prods = [f.prod() for f in col_in]
    out_conses = [f.cons() for f in col_out]

    def sequence(coords, counts, in_prods, out_conses):
        tg = TaskGroup()
        for c in range(cols):
            in_prods[c].fill(coords, in_taps[c], group=tg)
        for c in range(cols):
            out_conses[c].drain(counts, out_taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [in_ty, out_ty, in_prods, out_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def mandel_np(cx, cy, maxit):
    """The kernel's arithmetic in numpy, bit for bit."""
    def mul(a, b, shift):
        return np.clip((a.astype(np.int64) * b) >> shift, I32.min, I32.max)

    def wrap(v):
        return v.astype(np.int64).astype(np.int32)  # int32 add/sub wrap around, like the vector unit

    a, b = cx.astype(np.int64), cy.astype(np.int64)
    x, y = np.zeros_like(a), np.zeros_like(a)
    n = np.zeros(a.shape, np.int32)
    alive = np.ones(a.shape, bool)
    four = 4 << FRAC
    for _ in range(maxit):
        xx, yy = mul(x, x, FRAC), mul(y, y, FRAC)
        alive &= xx <= four - yy
        if not alive.any():
            break
        n += alive
        xy = mul(x, y, FRAC - 1)
        x = wrap(xx - yy + a).astype(np.int64)
        y = wrap(xy + b).astype(np.int64)
    return n


class Mandel:
    def __init__(self, w, h, cols, maxit=256):
        self.w, self.h, self.cols, self.maxit = w, h, cols, maxit
        self.cores = cols * ROWS
        self.tiles = -(-(w * h) // (self.cores * P))
        self.n = self.cores * self.tiles * P

    def coords(self, cx, cy, width):
        """Q28 c for every pixel of a view centred on (cx, cy), `width` wide, padded with far-away points."""
        if not hasattr(self, "_xi"):
            k = np.arange(self.n)
            real = k < self.w * self.h
            self._xi = np.where(real, k % self.w, 0).astype(np.int64)
            self._yi = np.where(real, k // self.w, 0).astype(np.int64)
            self._pad = ~real
        step = width / self.w
        x0 = round((cx - width / 2) * (1 << FRAC))
        y0 = round((cy - step * self.h / 2) * (1 << FRAC))
        dq = round(step * (1 << FRAC))
        gx = x0 + self._xi * dq
        gy = y0 + self._yi * dq
        gx[self._pad], gy[self._pad] = 4 << FRAC, 0
        return gx.astype(np.int32), gy.astype(np.int32)

    def pack(self, gx, gy):
        """Pixels in order -> per (column, tile, core in column): [cx x P][cy x P]."""
        shape = (self.cols, self.tiles, ROWS, P)
        return np.stack([gx.reshape(shape), gy.reshape(shape)], 3).ravel()

    def run(self, cx, cy, width):
        gx, gy = self.coords(cx, cy, width)
        if not hasattr(self, "_in"):
            self._in = iron.zeros((2 * self.n,), dtype=np.int32, device="npu")
            self._out = iron.zeros((self.n,), dtype=np.int32, device="npu")
        with self._in.overwrite() as a:
            a[...] = self.pack(gx, gy)
        t0 = time.perf_counter()
        mandel_chip(self._in, self._out, tiles=self.tiles, maxit=self.maxit, cols=self.cols)
        ms = (time.perf_counter() - t0) * 1000
        return self._out.numpy()[:self.w * self.h].reshape(self.h, self.w).copy(), ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", action="store_true")
    p.add_argument("--size", nargs=2, type=int, default=(960, 540), metavar=("W", "H"))
    p.add_argument("--maxit", type=int, default=1024)
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--x", type=float, default=-0.743643887037151)
    p.add_argument("--y", type=float, default=0.131825904205330)
    args = p.parse_args()
    dev = iron.get_current_device()
    m = Mandel(*args.size, dev.cols, args.maxit)
    width = 3.2 / args.zoom
    cx, cy = (-0.6, 0.0) if args.zoom == 1 else (args.x, args.y)
    counts, _ = m.run(cx, cy, width)  # compile + warm up
    times = [m.run(cx, cy, width)[1] for _ in range(5)]
    gx, gy = m.coords(cx, cy, width)
    t0 = time.perf_counter()
    want = mandel_np(gx[:m.w * m.h], gy[:m.w * m.h], args.maxit).reshape(m.h, m.w)
    cpu_ms = (time.perf_counter() - t0) * 1000
    iters = int(counts.sum())
    print(f"mandelbrot {m.w}x{m.h}, zoom {args.zoom:g}, maxit {args.maxit}, {m.cores} cores x {m.tiles} tiles of {P}")
    print(f"NPU  : {min(times):8.1f} ms per frame  {iters / min(times) / 1e6:8.2f} G iterations/s")
    print(f"numpy: {cpu_ms:8.1f} ms per frame  {iters / cpu_ms / 1e6:8.2f} G iterations/s")
    print("matches numpy bit for bit" if np.array_equal(counts, want) else
          f"DIFFERENT FROM NUMPY in {int((counts != want).sum())} pixels")


if __name__ == "__main__":
    main()
