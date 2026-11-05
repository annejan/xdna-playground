#!/usr/bin/env python3
"""Lenia on 32 AIE cores: Orbium, the glider of continuous Life, by the dozen.

  ../env.sh python lenia_chip.py --bench

Lenia (Bert Chan, 2018): every cell holds a value 0..1, and every step it looks at a ring of radius
13 around itself: U = the ring's weighted sum (412 weights), then A += dt * G(U), with G a bump around
mu = 0.15 (sigma 0.015). Orbium unicaudatus is a creature that glides through that world for ever.

On the NPU: A in Q12, the 412 ring weights in Q15, U = the 412 multiply-adds (aie::mac on 32 cells
at a time, into 32-bit accumulators) >> 15, and G from a 1024-entry table. Each core holds a band of
rows; every step it sends its top 13 and bottom 13 rows to the cores above and below along the same
snake as life_chip.py, and waits for theirs. Each row carries 16 cells of padding on both sides, a
copy of its own other end, so the ring wraps around left-right without any scalar code.
"""

import argparse
import re
import sys
import time

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

from life_chip import ROWS, snake

R = 13  # the ring's radius
PADX = 16  # cells of wrap-around padding left and right of every row
ONE = 4096  # Q12
MU, SIGMA, DT = 0.15, 0.015, 0.1

# Orbium unicaudatus from Bert Chan's animals.json (Chakazul/Lenia), in Lenia's own RLE
ORBIUM = ("7.MD6.qL$6.pKqEqFURpApBRAqQ$5.VqTrSsBrOpXpWpTpWpUpCrQ$4.CQrQsTsWsApITNPpGqGvL$3.IpIpWrOsGsBqXpJ4.LsFrL$"
          "A.DpKpSpJpDqOqUqSqE5.ExD$qL.pBpTT2.qCrGrVrWqM5.sTpP$.pGpWpD3.qUsMtItQtJ6.tL$.uFqGH3.pXtOuR2vFsK5.sM$"
          ".tUqL4.GuNwAwVxBwNpC4.qXpA$2.uH5.vBxGyEyMyHtW4.qIpL$2.wV5.tIyG3yOxQqW2.FqHpJ$2.tUS4.rM2yOyJyOyHtVpPMpFqNV$"
          "2.HsR4.pUxAyOxLxDxEuVrMqBqGqKJ$3.sLpE3.pEuNxHwRwGvUuLsHrCqTpR$3.TrMS2.pFsLvDvPvEuPtNsGrGqIP$"
          "4.pRqRpNpFpTrNtGtVtStGsMrNqNpF$5.pMqKqLqRrIsCsLsIrTrFqJpHE$6.RpSqJqPqVqWqRqKpRXE$8.OpBpIpJpFTK!")


def decode(rle):
    """Lenia's RLE: '.' is 0, A..X is 1..24, pA..yX is 25..255 (of 255); $ ends a row."""
    rows, row = [], []
    for n, tok in re.findall(r"(\d*)([p-y]?[A-X]|[.$!])", rle):
        k = int(n) if n else 1
        if tok == "$":
            rows.append(row)
            rows += [[]] * (k - 1)
            row = []
        elif tok == "!":
            break
        else:
            v = 0 if tok == "." else (ord(tok) - 64 if len(tok) == 1 else (ord(tok[0]) - ord("p")) * 24 + ord(tok[1]) - 40)
            row += [v / 255] * k
    rows.append(row)
    w = max(map(len, rows))
    return np.array([r + [0] * (w - len(r)) for r in rows])


def ring():
    """The kernel: a smooth ring of radius R, weights in Q15 summing to (almost) 1."""
    y, x = np.mgrid[-R:R + 1, -R:R + 1]
    r = np.sqrt(x * x + y * y) / R
    k = np.where((r > 0) & (r < 1), np.exp(4 - 1 / np.maximum(r * (1 - r), 1e-9)), 0)
    return np.round(k / k.sum() * 32768).astype(np.int64)


WEIGHTS = ring()
TAPS = [(dy - R, dx - R, int(WEIGHTS[dy, dx])) for dy, dx in zip(*np.nonzero(WEIGHTS))]
GROWTH = np.round(DT * (2 * np.exp(-(np.arange(1024) * 4 / ONE - MU) ** 2 / (2 * SIGMA ** 2)) - 1) * ONE).astype(np.int64)

STEP = """
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {w}
#define S ({w} + 2 * {padx})
#define RB {rows}
#define R {r}
#define NT {ntaps}
static const int8_t tdy[NT] = {{ {tdy} }};
static const int8_t tdx[NT] = {{ {tdx} }};
static const int16_t tw[NT] = {{ {tw} }};
static const int16_t growth[1024] = {{ {growth} }};
extern "C" {{
/* one Lenia step of a band: src -> dst; above/below hold the 13 rows next to it (all rows padded) */
void lenia_step(int16_t *src, int16_t *dst, int16_t *above, int16_t *below) {{
    aie::set_rounding(aie::rounding_mode::floor);
    aie::set_saturation(aie::saturation_mode::none);
    alignas(64) int16_t u[32];
    const int16_t *rowp[RB + 2 * R];
    for (int i = 0; i < R; i++) {{ rowp[i] = above + i * S; rowp[RB + R + i] = below + i * S; }}
    for (int i = 0; i < RB; i++) rowp[R + i] = src + i * S;
    for (int r = 0; r < RB; r++) {{
        const int16_t *const *rp = rowp + r + R;    /* rp[dy] is the row dy below row r, dy in -R..R */
        int16_t *o = dst + r * S + {padx};
        const int16_t *a = src + r * S + {padx};
        for (int c = 0; c < W; c += 32) {{
            aie::accum<acc32, 32> acc = aie::zeros<acc32, 32>();
            for (int t = 0; t < NT; t++)
                acc = aie::mac(acc, aie::load_unaligned_v<32>(rp[tdy[t]] + {padx} + c + tdx[t]), tw[t]);
            aie::store_v(u, acc.template to_vector<int16_t>(15));
            for (int i = 0; i < 32; i++) {{
                int k = u[i] >> 2;
                k = k < 0 ? 0 : k > 1023 ? 1023 : k;
                int v = a[c + i] + growth[k];
                o[c + i] = v < 0 ? 0 : v > {one} ? {one} : v;
            }}
        }}
        /* the padding: a copy of the row's other end, so the next step's ring wraps around */
        for (int i = 0; i < {padx}; i++) {{ o[-{padx} + i] = o[W - {padx} + i]; o[W + i] = o[i]; }}
    }}
}}
}}
"""

ROWCOPY = """
#include <stdint.h>
#include <aie_api/aie.hpp>
#define S ({w} + 2 * {padx})
extern "C" {{
/* copy R rows from row `first` of a band into an edge buffer */
void lenia_rows(int16_t *band, int16_t *edge, int32_t first) {{
    for (int i = 0; i < {r} * S; i += 32)
        aie::store_unaligned_v(edge + i, aie::load_unaligned_v<32>(band + first * S + i));
}}
}}
"""


@iron.jit
def lenia_chip(world: In, out: Out, *, w: CompileTime[int], rows: CompileTime[int], steps: CompileTime[int],
               cols: CompileTime[int]):
    cores = cols * ROWS
    stride = w + 2 * PADX
    band = rows * stride
    n = cores * band
    tensor_ty = np.ndarray[(n,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS * band,), np.dtype[np.int16]]
    band_ty = np.ndarray[(band,), np.dtype[np.int16]]
    edge_ty = np.ndarray[(R * stride,), np.dtype[np.int16]]

    fmt = {"w": w, "rows": rows, "padx": PADX, "r": R, "one": ONE, "ntaps": len(TAPS),
           "tdy": ", ".join(str(t[0]) for t in TAPS), "tdx": ", ".join(str(t[1]) for t in TAPS),
           "tw": ", ".join(str(t[2]) for t in TAPS), "growth": ", ".join(str(int(g)) for g in GROWTH)}
    step = ExternalFunction("lenia_step", source_string=STEP.format(**fmt), arg_types=[band_ty, band_ty, edge_ty, edge_ty])
    copy = ExternalFunction("lenia_rows", source_string=ROWCOPY.format(**fmt), arg_types=[band_ty, edge_ty, np.dtype(np.int32)])

    col_in = [ObjectFifo(col_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(col_ty, name=f"out{c}", depth=1) for c in range(cols)]
    band_in, band_out = [], []
    for c in range(cols):
        band_in += col_in[c].cons().split([i * band for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS,
                                          obj_types=[band_ty] * ROWS, names=[f"b_in{c}_{i}" for i in range(ROWS)])
        band_out += col_out[c].prod().join([i * band for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS,
                                           obj_types=[band_ty] * ROWS, names=[f"b_out{c}_{i}" for i in range(ROWS)])
    down = [ObjectFifo(edge_ty, name=f"down{k}", depth=1) for k in range(cores)]
    up = [ObjectFifo(edge_ty, name=f"up{k}", depth=1) for k in range(cores)]

    def core(b_in, b_out, to_up, to_down, from_above, from_below, step, copy):
        a = b_in.acquire(1)
        b = b_out.acquire(1)

        def one_step(src, dst):
            e = to_up.acquire(1)
            copy(src, e, 0)
            to_up.release(1)
            e = to_down.acquire(1)
            copy(src, e, rows - R)
            to_down.release(1)
            above = from_above.acquire(1)
            below = from_below.acquire(1)
            step(src, dst, above, below)
            from_above.release(1)
            from_below.release(1)

        for _ in range_(steps // 2):
            one_step(a, b)
            one_step(b, a)
        one_step(a, b)
        b_in.release(1)
        b_out.release(1)

    workers = [Worker(core, [band_in[k].cons(), band_out[k].prod(), up[k].prod(), down[k].prod(),
                             down[(k - 1) % cores].cons(), up[(k + 1) % cores].cons(), step, copy],
                      tile=Tile(*snake(k))) for k in range(cores)]

    taps = TensorAccessPattern.full((n,)).partition(cols)
    in_prods = [f.prod() for f in col_in]
    out_conses = [f.cons() for f in col_out]

    def sequence(world, out, in_prods, out_conses):
        tg = TaskGroup()
        for c in range(cols):
            in_prods[c].fill(world, taps[c], group=tg)
        for c in range(cols):
            out_conses[c].drain(out, taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [tensor_ty, tensor_ty, in_prods, out_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def lenia_np(a, steps):
    """The kernel's arithmetic in numpy, bit for bit: A (h, w) int16 in Q12 on a torus."""
    a = a.astype(np.int64)
    for _ in range(steps):
        u = np.zeros_like(a)
        for dy, dx, wt in TAPS:
            u += wt * np.roll(np.roll(a, -dy, 0), -dx, 1)
        u >>= 15
        a = np.clip(a + GROWTH[np.clip(u >> 2, 0, 1023)], 0, ONE)
    return a.astype(np.int16)


def creatures(h, w, rng, count=12):
    """A world with `count` Orbia at random places, turned every which way."""
    a = np.zeros((h, w), np.int16)
    o = np.round(decode(ORBIUM) * ONE).astype(np.int16)
    for _ in range(count):
        s = np.rot90(o, int(rng.integers(0, 4)))
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        ys, xs = (np.arange(s.shape[0]) + y) % h, (np.arange(s.shape[1]) + x) % w
        a[np.ix_(ys, xs)] = np.maximum(a[np.ix_(ys, xs)], s)
    return a


class Lenia:
    def __init__(self, w, steps, cols, rows=R):
        self.w = w - w % 32
        self.cores = cols * ROWS
        self.rows = max(R, rows)
        self.h = self.rows * self.cores
        self.steps, self.cols = steps, cols

    def pack(self, a):
        s = self.w + 2 * PADX
        p = np.zeros((self.h, s), np.int16)
        p[:, PADX:PADX + self.w] = a
        p[:, :PADX] = a[:, -PADX:]
        p[:, PADX + self.w:] = a[:, :PADX]
        return p.ravel()

    def run(self, a):
        s = self.w + 2 * PADX
        if not hasattr(self, "_in"):
            self._in = iron.zeros((self.h * s,), dtype=np.int16, device="npu")
            self._out = iron.zeros((self.h * s,), dtype=np.int16, device="npu")
        with self._in.overwrite() as buf:
            buf[...] = self.pack(a)
        t0 = time.perf_counter()
        lenia_chip(self._in, self._out, w=self.w, rows=self.rows, steps=self.steps, cols=self.cols)
        ms = (time.perf_counter() - t0) * 1000
        return self._out.numpy().reshape(self.h, s)[:, PADX:PADX + self.w].copy(), ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", action="store_true")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--rows", type=int, default=R, help="rows per core (at least 13)")
    p.add_argument("--steps", type=int, default=201, help="steps per NPU call, odd")
    p.add_argument("--calls", type=int, default=5)
    args = p.parse_args()
    if args.steps % 2 == 0:
        sys.exit("--steps must be odd")
    dev = iron.get_current_device()
    rng = np.random.default_rng(1)
    check = Lenia(args.width, 21, dev.cols, args.rows)
    a = creatures(check.h, check.w, rng)
    got, _ = check.run(a)
    same = np.array_equal(got, lenia_np(a, 21))
    lenia = Lenia(args.width, args.steps, dev.cols, args.rows)
    lenia.run(a)  # compile + warm up
    t0 = time.perf_counter()
    b = a
    for _ in range(args.calls):
        b, _ = lenia.run(b)
    dt = time.perf_counter() - t0
    n = args.steps * args.calls
    t1 = time.perf_counter()
    lenia_np(a, 2)
    cpu_dt = (time.perf_counter() - t1) / 2
    cells = lenia.w * lenia.h
    print(f"lenia: world {lenia.w}x{lenia.h}, {lenia.cores} cores x {lenia.rows} rows, {len(TAPS)} taps, "
          f"{args.steps} steps per call")
    print(f"NPU  : {n / dt:9.0f} steps/s  {cells * n / dt / 1e6:9.1f} Mcell/s  {cells * len(TAPS) * n / dt / 1e9:7.1f} GMAC/s")
    print(f"numpy: {1 / cpu_dt:9.1f} steps/s  {cells / cpu_dt / 1e6:9.1f} Mcell/s")
    print("21 steps match numpy bit for bit" if same else f"DIFFERENT FROM NUMPY in {int((got != lenia_np(a, 21)).sum())} cells")
    print(f"mass {a.sum() / ONE:.1f} at the start, {b.sum() / ONE:.1f} after {n} steps")


if __name__ == "__main__":
    main()
