#!/usr/bin/env python3
"""Gray-Scott reaction-diffusion on 32 AIE cores: coral, spots and dividing cells.

  ../env.sh python gray_chip.py --bench                 1001 steps per call, race numpy
  ../env.sh python gray_chip.py --bench --preset mitosis

Two chemicals per cell, U and V, as int16 in Q12 (4096 = 1.0). Every step:

  U += Du * lap(U) - U*V*V + F * (1 - U)
  V += Dv * lap(V) + U*V*V - (F + k) * V

with a 5-point Laplacian, Du = 3/16 and Dv = 3/32 (shifts and adds only), and the products
through the vector multiplier (int16 x int16 into a 32-bit accumulator, then >> 12, rounding to
floor so numpy can follow it bit for bit). F and k pick the pattern, see PRESETS.

The world is cut in bands, one per core, along the same snake as life_chip.py: every step a core
sends its top and bottom rows (of both U and V) to its neighbours and waits for theirs. A band
is [U rows][V rows]; an edge buffer is [U row][V row], each with PAD cells either side.
"""

import argparse
import sys
import time

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

from life_chip import PAD, ROWS, snake

ONE = 4096  # Q12
LANES = 32  # int16 lanes in a 512-bit vector
CELLS_MAX = 5120  # cells per core: in + out buffers of 2 x int16 per cell = 40 KB, plus the edge buffers

# (F, k) from Pearson's map; Karl Sims' names
PRESETS = {
    "coral": (0.0545, 0.062),
    "mitosis": (0.0367, 0.0649),
    "worms": (0.078, 0.061),
    "spots": (0.03, 0.062),
    "waves": (0.014, 0.045),
}


def fixed(preset):
    f, k = PRESETS[preset]
    return round(f * ONE), round((f + k) * ONE)


STEP = """
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {w}
#define R {rows}
#define E (W + 2 * {pad})
extern "C" {{
/* one Gray-Scott step of a band: src -> dst; above/below hold the neighbours' rows (U row, V row) */
void gs_step_{preset}(int16_t *src, int16_t *dst, int16_t *above, int16_t *below) {{
    aie::set_rounding(aie::rounding_mode::floor);
    aie::set_saturation(aie::saturation_mode::none);
    const auto one = aie::broadcast<int16_t, 32>({one});
    const auto F = aie::broadcast<int16_t, 32>({f});
    const auto FK = aie::broadcast<int16_t, 32>({fk});
    int16_t *U = src, *V = src + R * W, *U2 = dst, *V2 = dst + R * W;
    for (int r = 0; r < R; r++) {{
        const int16_t *uu = r ? U + (r - 1) * W : above + {pad};
        const int16_t *vu = r ? V + (r - 1) * W : above + E + {pad};
        const int16_t *ud = r < R - 1 ? U + (r + 1) * W : below + {pad};
        const int16_t *vd = r < R - 1 ? V + (r + 1) * W : below + E + {pad};
        const int16_t *um = U + r * W, *vm = V + r * W;
        int16_t *uo = U2 + r * W, *vo = V2 + r * W;
        for (int c = 0; c < W; c += 32) {{
            auto u = aie::load_unaligned_v<32>(um + c), v = aie::load_unaligned_v<32>(vm + c);
            auto lu = aie::add(aie::add(aie::load_unaligned_v<32>(uu + c), aie::load_unaligned_v<32>(ud + c)),
                               aie::add(aie::load_unaligned_v<32>(um + c - 1), aie::load_unaligned_v<32>(um + c + 1)));
            auto lv = aie::add(aie::add(aie::load_unaligned_v<32>(vu + c), aie::load_unaligned_v<32>(vd + c)),
                               aie::add(aie::load_unaligned_v<32>(vm + c - 1), aie::load_unaligned_v<32>(vm + c + 1)));
            lu = aie::sub(lu, aie::upshift(u, 2));
            lv = aie::sub(lv, aie::upshift(v, 2));
            auto du = aie::add(aie::downshift(lu, 3), aie::downshift(lu, 4));   /* 3/16 */
            auto dv = aie::add(aie::downshift(lv, 4), aie::downshift(lv, 5));   /* 3/32 */
            auto uv = aie::mul(u, v).template to_vector<int16_t>(12);
            auto uvv = aie::mul(uv, v).template to_vector<int16_t>(12);
            auto fu = aie::mul(F, aie::sub(one, u)).template to_vector<int16_t>(12);
            auto kv = aie::mul(FK, v).template to_vector<int16_t>(12);
            aie::store_unaligned_v(uo + c, aie::add(aie::sub(aie::add(u, du), uvv), fu));
            aie::store_unaligned_v(vo + c, aie::sub(aie::add(aie::add(v, dv), uvv), kv));
        }}
        /* columns 0 and W-1 wrap around: redo them in scalar code */
        for (int k = 0; k < 2; k++) {{
            int cc = k ? W - 1 : 0, l = cc ? cc - 1 : W - 1, rt = cc == W - 1 ? 0 : cc + 1;
            int u = um[cc], v = vm[cc];
            int lu = uu[cc] + ud[cc] + um[l] + um[rt] - 4 * u;
            int lv = vu[cc] + vd[cc] + vm[l] + vm[rt] - 4 * v;
            int uv = (u * v) >> 12, uvv = (uv * v) >> 12;
            int fu = ({f} * ({one} - u)) >> 12, kv = ({fk} * v) >> 12;
            uo[cc] = (int16_t)(u + (lu >> 3) + (lu >> 4) - uvv + fu);
            vo[cc] = (int16_t)(v + (lv >> 4) + (lv >> 5) + uvv - kv);
        }}
    }}
}}
}}
"""

ROW = """
#include <stdint.h>
#include <aie_api/aie.hpp>
#define W {w}
#define R {rows}
#define E (W + 2 * {pad})
extern "C" {{
/* copy row r of U and of V into an edge buffer */
void gs_row_{preset}(int16_t *band, int16_t *edge, int32_t r) {{
    for (int c = 0; c < W; c += 32) {{
        aie::store_unaligned_v(edge + {pad} + c, aie::load_unaligned_v<32>(band + r * W + c));
        aie::store_unaligned_v(edge + E + {pad} + c, aie::load_unaligned_v<32>(band + R * W + r * W + c));
    }}
}}
}}
"""


@iron.jit
def gray_chip(world: In, out: Out, *, w: CompileTime[int], rows: CompileTime[int], steps: CompileTime[int],
              cols: CompileTime[int], preset: CompileTime[str] = "coral"):
    cores = cols * ROWS
    band = 2 * rows * w
    n = cores * band
    tensor_ty = np.ndarray[(n,), np.dtype[np.int16]]
    col_ty = np.ndarray[(ROWS * band,), np.dtype[np.int16]]
    band_ty = np.ndarray[(band,), np.dtype[np.int16]]
    edge_ty = np.ndarray[(2 * (w + 2 * PAD),), np.dtype[np.int16]]

    f, fk = fixed(preset)
    fmt = {"w": w, "rows": rows, "pad": PAD, "one": ONE, "f": f, "fk": fk, "preset": preset}
    step = ExternalFunction(f"gs_step_{preset}", source_string=STEP.format(**fmt),
                            arg_types=[band_ty, band_ty, edge_ty, edge_ty])
    row = ExternalFunction(f"gs_row_{preset}", source_string=ROW.format(**fmt),
                           arg_types=[band_ty, edge_ty, np.dtype(np.int32)])

    col_in = [ObjectFifo(col_ty, name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(col_ty, name=f"out{c}", depth=1) for c in range(cols)]
    band_in, band_out = [], []
    for c in range(cols):
        band_in += col_in[c].cons().split([i * band for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS,
                                          obj_types=[band_ty] * ROWS, names=[f"b_in{c}_{i}" for i in range(ROWS)])
        band_out += col_out[c].prod().join([i * band for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS,
                                           obj_types=[band_ty] * ROWS, names=[f"b_out{c}_{i}" for i in range(ROWS)])

    # depth 1: two int16 rows per buffer, and memory is tight with two fields per cell
    down = [ObjectFifo(edge_ty, name=f"down{k}", depth=1) for k in range(cores)]
    up = [ObjectFifo(edge_ty, name=f"up{k}", depth=1) for k in range(cores)]

    def core(b_in, b_out, to_up, to_down, from_above, from_below, step, row):
        a = b_in.acquire(1)
        b = b_out.acquire(1)

        def one_step(src, dst):
            e = to_up.acquire(1)
            row(src, e, 0)
            to_up.release(1)
            e = to_down.acquire(1)
            row(src, e, rows - 1)
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
                             down[(k - 1) % cores].cons(), up[(k + 1) % cores].cons(), step, row],
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


def gs_np(u, v, preset, steps):
    """The kernel's arithmetic in numpy, bit for bit (int32 maths, int16 results, >> is floor)."""
    f, fk = fixed(preset)
    u, v = u.astype(np.int32), v.astype(np.int32)
    for _ in range(steps):
        lu = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
        lv = np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v
        uv = (u * v) >> 12
        uvv = (uv * v) >> 12
        fu = (f * (ONE - u)) >> 12
        kv = (fk * v) >> 12
        u, v = (u + (lu >> 3) + (lu >> 4) - uvv + fu).astype(np.int16).astype(np.int32), \
               (v + (lv >> 4) + (lv >> 5) + uvv - kv).astype(np.int16).astype(np.int32)
    return u.astype(np.int16), v.astype(np.int16)


def seed(h, w, rng, blobs=None):
    """U = 1 everywhere, V = 0, with a few square drops of V to start the reaction."""
    u = np.full((h, w), ONE, np.int16)
    v = np.zeros((h, w), np.int16)
    for _ in range(blobs or max(4, h * w // 4000)):
        y, x, s = int(rng.integers(0, h)), int(rng.integers(0, w)), int(rng.integers(3, 8))
        ys, xs = np.arange(y - s, y + s) % h, np.arange(x - s, x + s) % w
        u[np.ix_(ys, xs)] = ONE // 2
        v[np.ix_(ys, xs)] = ONE // 4
    return u, v


class Gray:
    def __init__(self, w, h, steps, cols, preset="coral"):
        self.w = max(LANES, w - w % LANES)
        self.cores = cols * ROWS
        self.rows = max(1, min(-(-h // self.cores), CELLS_MAX // self.w))
        self.h = self.rows * self.cores
        self.steps, self.cols, self.preset = steps, cols, preset

    def pack(self, u, v):
        """(h, w) U and V -> per core [U band][V band], cores in chain order."""
        k, r = self.cores, self.rows
        return np.stack([u.reshape(k, r, self.w), v.reshape(k, r, self.w)], 1).ravel()

    def unpack(self, flat):
        a = flat.reshape(self.cores, 2, self.rows, self.w)
        return a[:, 0].reshape(self.h, self.w), a[:, 1].reshape(self.h, self.w)

    def run(self, u, v):
        if not hasattr(self, "_in"):
            self._in = iron.zeros((2 * u.size,), dtype=np.int16, device="npu")
            self._out = iron.zeros((2 * u.size,), dtype=np.int16, device="npu")
        with self._in.overwrite() as a:
            a[...] = self.pack(u, v)
        t0 = time.perf_counter()
        gray_chip(self._in, self._out, w=self.w, rows=self.rows, steps=self.steps, cols=self.cols,
                  preset=self.preset)
        ms = (time.perf_counter() - t0) * 1000
        u2, v2 = self.unpack(self._out.numpy())
        return u2.copy(), v2.copy(), ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", action="store_true")
    p.add_argument("--steps", type=int, default=1001, help="steps per NPU call, odd")
    p.add_argument("--size", nargs=2, type=int, default=(512, 320), metavar=("W", "H"))
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--preset", choices=sorted(PRESETS), default="coral")
    args = p.parse_args()
    if args.steps % 2 == 0:
        sys.exit("--steps must be odd")
    dev = iron.get_current_device()
    g = Gray(*args.size, args.steps, dev.cols, args.preset)
    u, v = seed(g.h, g.w, np.random.default_rng(1))
    # check a short run against numpy first
    check = Gray(g.w, g.h, 101, dev.cols, args.preset)
    cu, cv, _ = check.run(u, v)
    wu, wv = gs_np(u, v, args.preset, 101)
    same = np.array_equal(cu, wu) and np.array_equal(cv, wv)
    g.run(u, v)  # compile + warm up
    t0 = time.perf_counter()
    for _ in range(args.calls):
        u, v, _ = g.run(u, v)
    dt = time.perf_counter() - t0
    n = args.steps * args.calls
    t1 = time.perf_counter()
    gs_np(u, v, args.preset, 50)
    cpu_dt = (time.perf_counter() - t1) / 50
    cells = g.w * g.h
    print(f"gray-scott {args.preset}: world {g.w}x{g.h}, {g.cores} cores x {g.rows} rows, "
          f"{args.steps} steps per call, {args.calls} calls")
    print(f"NPU  : {n / dt:9.0f} steps/s  {cells * n / dt / 1e6:9.1f} Mcell/s")
    print(f"numpy: {1 / cpu_dt:9.0f} steps/s  {cells / cpu_dt / 1e6:9.1f} Mcell/s")
    print("101 steps match numpy bit for bit" if same else
          f"DIFFERENT FROM NUMPY: U {int((cu != wu).sum())} cells, V {int((cv != wv).sum())} cells")
    print(f"V after the run: min {int(v.min())}, max {int(v.max())}, mean {float(v.mean()):.0f} (of {ONE})")


if __name__ == "__main__":
    main()
