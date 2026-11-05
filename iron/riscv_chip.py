#!/usr/bin/env python3
"""32 RISC-V computers in the NPU: an RV32IM interpreter on every AIE core, each with 32 KB of RAM.

  ../riscv/build.sh                           build the demo programs (clang, bare metal RV32IM)
  ../env.sh python riscv_chip.py --check      run every machine on the NPU and on the host, compare
  ../env.sh python riscv_chip.py --bench      how many MIPS the 32 cores manage together

Every core runs ../riscv/rv32.h, the same interpreter the host runs, on its own 32 KB of RAM. The RAM
is a buffer on the core itself, so it stays there between calls: a call only brings a command (load a
program, or run up to N instructions) and takes the framebuffer (160 x 96, 16 colours) back. A program
ends its frame with `ecall`, like waiting for the vertical blank.
"""

import argparse
import os
import sys
import time

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.runtime import TaskGroup

from life_chip import ROWS, snake

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "riscv"))
import machines as M  # noqa: E402

CMD = 64  # command header: words cmd, id or frame, budget, length
PAYLOAD = 4096  # a program, when loading
IN = CMD + PAYLOAD
OUT = 64 + M.FB_BYTES  # status header (why, instructions run), then the framebuffer
LOAD, RUN = 1, 2

KERNEL = """
extern "C" {
/* one call: load a program into this core's machine, or run it for a while; framebuffer out */
void rv_core(uint8_t *ram, uint8_t *cmd, uint8_t *out) {
    uint32_t c = rv_ld32(cmd, 0), arg = rv_ld32(cmd, 4), budget = rv_ld32(cmd, 8), len = rv_ld32(cmd, 12);
    uint32_t why = 0, ran = 0;
    if (c == 1) {
        for (uint32_t i = 0; i < RV_RAM; i++) ram[i] = 0;
        for (uint32_t i = 0; i < len && i < PAYLOAD; i++) ram[i] = cmd[CMD + i];
        rv_st32(ram, RV_MAIL + 4, arg);              /* the machine's id */
    } else if (c == 2) {
        rv_st32(ram, RV_MAIL, arg);                  /* the frame number */
        why = (uint32_t)rv_run(ram, budget, &ran);
    }
    for (uint32_t i = 0; i < 64; i++) out[i] = 0;
    rv_st32(out, 0, why);
    rv_st32(out, 4, ran);
    rv_st32(out, 8, rv_ld32(ram, RV_CPU + 128));     /* pc */
    for (uint32_t i = 0; i < RV_FB_BYTES; i++) out[64 + i] = ram[RV_FB + i];
}
}
"""


def source():
    with open(os.path.join(HERE, "..", "riscv", "rv32.h")) as f:
        rv = f.read()
    return f"#define CMD {CMD}\n#define PAYLOAD {PAYLOAD}\n" + rv + KERNEL


@iron.jit
def riscv_chip(cmds: In, outs: Out, *, cols: CompileTime[int]):
    cores = cols * ROWS
    in_ty = np.ndarray[(cores * IN,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(cores * OUT,), np.dtype[np.uint8]]
    cin_ty = np.ndarray[(IN,), np.dtype[np.uint8]]
    cout_ty = np.ndarray[(OUT,), np.dtype[np.uint8]]
    ram_ty = np.ndarray[(M.RAM,), np.dtype[np.uint8]]

    kernel = ExternalFunction("rv_core", source_string=source(), arg_types=[ram_ty, cin_ty, cout_ty])

    col_in = [ObjectFifo(np.ndarray[(ROWS * IN,), np.dtype[np.uint8]], name=f"in{c}", depth=1) for c in range(cols)]
    col_out = [ObjectFifo(np.ndarray[(ROWS * OUT,), np.dtype[np.uint8]], name=f"out{c}", depth=1) for c in range(cols)]
    c_in, c_out = [], []
    for c in range(cols):
        c_in += col_in[c].cons().split([i * IN for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS,
                                       obj_types=[cin_ty] * ROWS, names=[f"c_in{c}_{i}" for i in range(ROWS)])
        c_out += col_out[c].prod().join([i * OUT for i in range(ROWS)], tile=Tile(c, 1), depths=[1] * ROWS,
                                        obj_types=[cout_ty] * ROWS, names=[f"c_out{c}_{i}" for i in range(ROWS)])

    # the machines' RAM: a buffer on each core, kept between calls
    rams = [Buffer(ram_ty, name=f"ram{k}") for k in range(cores)]

    def core(f_in, f_out, ram, kernel):
        a = f_in.acquire(1)
        b = f_out.acquire(1)
        kernel(ram, a, b)
        f_in.release(1)
        f_out.release(1)

    workers = [Worker(core, [c_in[k].cons(), c_out[k].prod(), rams[k], kernel], tile=Tile(*snake(k)))
               for k in range(cores)]

    in_taps = TensorAccessPattern.full((cores * IN,)).partition(cols)
    out_taps = TensorAccessPattern.full((cores * OUT,)).partition(cols)
    in_prods = [f.prod() for f in col_in]
    out_conses = [f.cons() for f in col_out]

    def sequence(cmds, outs, in_prods, out_conses):
        tg = TaskGroup()
        for c in range(cols):
            in_prods[c].fill(cmds, in_taps[c], group=tg)
        for c in range(cols):
            out_conses[c].drain(outs, out_taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [in_ty, out_ty, in_prods, out_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class Machines:
    """32 RISC-V machines on the NPU. Core k (in snake order) is machine k."""

    def __init__(self, cols, budget=1_000_000):
        self.cols, self.n, self.budget = cols, cols * ROWS, budget
        self.frames = np.zeros(self.n, np.uint32)  # the frame each machine is working on
        self._in = iron.zeros((self.n * IN,), dtype=np.uint8, device="npu")
        self._out = iron.zeros((self.n * OUT,), dtype=np.uint8, device="npu")

    def _call(self, cmds):
        with self._in.overwrite() as buf:
            buf[...] = cmds.ravel()
        t0 = time.perf_counter()
        riscv_chip(self._in, self._out, cols=self.cols)
        ms = (time.perf_counter() - t0) * 1000
        return self._out.numpy().reshape(self.n, OUT).copy(), ms

    def load(self, programs):
        """programs[k]: the bytes for machine k."""
        cmds = np.zeros((self.n, IN), np.uint8)
        for k, code in enumerate(programs):
            hdr = np.array([LOAD, k, 0, len(code)], np.uint32).view(np.uint8)
            cmds[k, :16] = hdr
            cmds[k, CMD:CMD + len(code)] = np.frombuffer(code, np.uint8)
        self.frames[:] = 0
        return self._call(cmds)

    def run(self):
        """Every machine runs until it ends a frame or the budget is spent. Returns the outputs and ms."""
        cmds = np.zeros((self.n, IN), np.uint8)
        hdr = np.zeros((self.n, 4), np.uint32)
        hdr[:, 0], hdr[:, 1], hdr[:, 2] = RUN, self.frames, self.budget
        cmds[:, :16] = hdr.view(np.uint8)
        outs, ms = self._call(cmds)
        why = outs[:, :4].copy().view(np.uint32)[:, 0]
        self.frames[why == M.FRAME] += 1
        return outs, ms

    @staticmethod
    def status(outs):
        h = outs[:, :12].copy().view(np.uint32)
        return h[:, 0], h[:, 1], h[:, 2]  # why, instructions run, pc

    @staticmethod
    def screens(outs):
        fb = outs[:, 64:]
        px = np.empty((len(outs), M.W * M.H), np.uint8)
        px[:, 0::2], px[:, 1::2] = fb & 15, fb >> 4
        return px.reshape(len(outs), M.H, M.W)


def programs(n):
    return [M.program(M.PROGRAMS[k % len(M.PROGRAMS)]) for k in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="compare every machine with the host interpreter")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--budget", type=int, default=1_000_000, help="instructions per machine per call")
    args = p.parse_args()
    dev = iron.get_current_device()
    mach = Machines(dev.cols, args.budget)
    mach.load(programs(mach.n))

    host = M.Host()
    rams = M.boot(mach.n)
    frames = np.zeros(mach.n, np.uint32)
    same, total, t_npu = True, 0, 0.0
    for call in range(args.calls):
        outs, ms = mach.run()
        t_npu += ms
        why, ran, pc = Machines.status(outs)
        total += int(ran.sum())
        if args.check or call < 3:
            for k in range(mach.n):
                hw, hran = host.frame(rams[k], int(frames[k]), args.budget)
                if hw == M.FRAME:
                    frames[k] += 1
                if hw != why[k] or hran != ran[k] or not np.array_equal(M.screen(rams[k]), Machines.screens(outs[k:k + 1])[0]):
                    same = False
                    print(f"machine {k} ({M.PROGRAMS[k % 4]}) differs at call {call}: NPU why {why[k]} ran {ran[k]} "
                          f"pc {pc[k]:#x}, host why {hw} ran {hran}")
    print(f"{mach.n} RISC-V machines on the NPU, {args.calls} calls of up to {args.budget:,} instructions each")
    print(f"ran {total / 1e6:.1f} M instructions in {t_npu / 1000:.2f} s of NPU calls: {total / t_npu / 1000:.0f} MIPS together, "
          f"{total / t_npu / 1000 / mach.n:.1f} MIPS per core")
    print(f"frames done: {', '.join(f'{p} {int(mach.frames[i::4].mean())}' for i, p in enumerate(M.PROGRAMS))}")
    print(("every call matches the host interpreter" if args.check else "the first 3 calls match the host interpreter")
          if same else "DIFFERENT FROM THE HOST")


if __name__ == "__main__":
    main()
