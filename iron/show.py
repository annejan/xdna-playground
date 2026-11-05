#!/usr/bin/env python3
"""The show: everything on 32 NPU cores in one window, made for a party table.

  ../env.sh python show.py --fit --demo        fullscreen, a new scene every 25 seconds
  ../env.sh python show.py                     in a window, start with Life

  tab        next effect: Life, Brian's Brain, Wireworld, Gray-Scott, Mandelbrot
  1 .. 5     speed (what a step is depends on the effect; the first use of a speed compiles it)
  m          demo mode on / off
  space      pause          r  new start          c  clear          f  fullscreen
  h          help           q, esc  quit

  Life & co    mouse draws; g n p a d l w x v u stamp patterns (shift: 12); e: electron (Wireworld)
  Gray-Scott   mouse left drops chemical, right wipes; p: next preset (coral, mitosis, worms, ...)
  Mandelbrot   mouse left: zoom towards there; right: zoom out; n: next place to dive into
"""

import argparse
import math
import time

import numpy as np
import pygame

import aie.iron as iron
from gray_chip import ONE, PRESETS, Gray
from gray_chip import seed as gray_seed
from life_chip import ROWS, Chip
from life_window import SPEEDS, TITLES, World, colorize, fit_world, hud, lut, place
from mandel_chip import FRAC, Mandel
from patterns import SHAPES, stamp
from rules import feed

# ---------------------------------------------------------------------------------------------
# effects: each knows its own world size, speeds, how to step on the NPU and how to look


class Life:
    """Life, Brian's Brain and Wireworld: the cellular automata of life_window.py."""

    smooth = False
    unit = "gen"

    def __init__(self, sw, sh, cols, rule, fade):
        self.vw, self.vh = fit_world(sw, sh, cols * ROWS)
        self.chip = Chip(self.vw, self.vh, SPEEDS[1], cols, rule)
        self.world = World(self.chip, self.vw, self.vh, 0.3, np.random.default_rng())
        self.speeds, self.fade, self.turn = SPEEDS, fade, 0
        self.title = TITLES[rule]

    def set_speed(self, i):
        self.chip.gens = self.speeds[i]

    def key(self):
        return (self.chip.rule, self.chip.gens)

    def step(self):
        self.world.cells, ms = self.chip.run(self.world.cells)
        self.gen = getattr(self, "gen", 0) + self.chip.gens
        if self.world.sources:
            feed(self.world.cells, self.world.sources, self.gen, self.chip.gens)
        return ms, self.chip.gens

    def cells(self):
        return self.chip.w * self.chip.h

    def image(self):
        return colorize(self.chip.rule, self.world.cells, self.world.heat, self.fade)[:self.vh, :self.vw]

    def paint(self, x, y, left):
        rule = self.chip.rule
        radius = 0 if rule == "wire" else 3
        ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disc = xs * xs + ys * ys <= radius * radius
        yy, xx = (ys + y) % self.chip.h, (xs + x) % self.chip.w
        pen = (3 if rule == "wire" else 1) if left else 0
        self.world.cells[yy, xx] = np.where(disc, pen, self.world.cells[yy, xx])

    def press(self, e, x, y):
        rule = self.chip.rule
        if e.key == pygame.K_r:
            self.world.reset()
        elif e.key == pygame.K_c:
            self.world.clear()
        elif e.key == pygame.K_t:
            self.turn = (self.turn + 1) % 4
            return f"stamps turned {self.turn * 90} degrees"
        elif e.key == pygame.K_e and rule == "wire":
            self.world.cells[y % self.chip.h, x % self.chip.w] = 1
            self.world.cells[y % self.chip.h, (x - 1) % self.chip.w] = 2
        elif rule == "life" and e.unicode.lower() in SHAPES:
            k = e.unicode.lower()
            if e.mod & pygame.KMOD_SHIFT:
                return self.world.fleet(k, x, y, self.turn)
            stamp(self.world.cells, SHAPES[k][1], x, y, self.turn)
            return SHAPES[k][0]
        return None

    def start(self, what):
        """Demo scene start: 'soup' or a pattern key for a fleet of 12."""
        self.world.reset()
        if what != "soup":
            self.world.clear()
            return self.world.fleet(what)
        return None

    def status(self):
        return ""


PALETTES = {
    "coral": lut([(0.0, (4, 8, 28)), (0.3, (10, 50, 100)), (0.55, (30, 150, 170)), (0.8, (200, 235, 210)),
                  (1.0, (255, 255, 255))]),
    "mitosis": lut([(0.0, (10, 2, 20)), (0.35, (80, 10, 90)), (0.6, (210, 50, 140)), (0.85, (255, 190, 210)),
                    (1.0, (255, 255, 255))]),
    "worms": lut([(0.0, (5, 15, 5)), (0.35, (20, 80, 30)), (0.6, (120, 190, 40)), (0.85, (240, 240, 140)),
                  (1.0, (255, 255, 255))]),
    "spots": lut([(0.0, (20, 10, 0)), (0.35, (110, 50, 0)), (0.6, (230, 140, 20)), (0.85, (255, 230, 150)),
                  (1.0, (255, 255, 255))]),
    "waves": lut([(0.0, (0, 0, 10)), (0.3, (30, 20, 120)), (0.6, (120, 80, 230)), (0.85, (220, 200, 255)),
                  (1.0, (255, 255, 255))]),
}


def fit_gray(sw, sh, cores):
    """The biggest Gray-Scott world with the screen's shape: W a multiple of 32, at most 5120 cells
    per core (two int16 fields, double buffered)."""
    best = (32, 32)
    for w in range(32, 2049, 32):
        h = round(w * sh / sw)
        if math.ceil(h / cores) * w <= 5120:
            best = (w, h)
    return best


class Reaction:
    """Gray-Scott reaction-diffusion: hundreds of steps per frame, which is what needs the NPU."""

    smooth = True
    unit = "step"
    title = "Gray-Scott reaction-diffusion"

    def __init__(self, sw, sh, cols):
        self.vw, self.vh = fit_gray(sw, sh, cols * ROWS)
        self.speeds = [11, 51, 151, 401, 801]
        self.g = Gray(self.vw, self.vh, self.speeds[2], cols, "coral")
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.u, self.v = gray_seed(self.g.h, self.g.w, self.rng)

    def set_speed(self, i):
        self.g.steps = self.speeds[i]

    def key(self):
        return ("gray", self.g.preset, self.g.steps)

    def step(self):
        self.u, self.v, ms = self.g.run(self.u, self.v)
        return ms, self.g.steps

    def cells(self):
        return self.g.w * self.g.h

    def image(self):
        x = np.clip(self.v[:self.vh, :self.vw].astype(np.int32) * 255 // 1800, 0, 255)
        return PALETTES[self.g.preset][x]

    def paint(self, x, y, left):
        r = 5 if left else 10
        ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
        disc = xs * xs + ys * ys <= r * r
        yy, xx = (ys + y) % self.g.h, (xs + x) % self.g.w
        self.u[yy, xx] = np.where(disc, ONE // 2 if left else ONE, self.u[yy, xx])
        self.v[yy, xx] = np.where(disc, ONE // 4 if left else 0, self.v[yy, xx])

    def press(self, e, x, y):
        if e.key == pygame.K_p:
            names = list(PRESETS)
            self.g.preset = names[(names.index(self.g.preset) + 1) % len(names)]
            return f"preset: {self.g.preset}"
        if e.key == pygame.K_r:
            self.reset()
        elif e.key == pygame.K_c:
            self.u[:], self.v[:] = ONE, 0
            return "cleared: drop some chemical with the mouse"
        return None

    def start(self, preset):
        self.g.preset = preset
        self.reset()
        return f"preset: {preset}"

    def status(self):
        return f"  |  {self.g.preset}"


TARGETS = [(-0.743643887037151, 0.131825904205330), (-0.761574, -0.0847596), (-1.768778833, -0.001738996),
           (-0.77568377, 0.13646737), (-0.7746806106269039, -0.1374168856037867)]


class Fractal:
    """Mandelbrot: dive into one of TARGETS, palette cycling like it is 1992."""

    smooth = False
    unit = "frame"
    title = "Mandelbrot"

    def __init__(self, sw, sh, cols):
        s = max(1, math.ceil(math.sqrt(sw * sh / 600_000)))  # about 600k pixels: a frame in ~20 ms
        self.vw, self.vh = sw // s, sh // s
        self.m = Mandel(self.vw, self.vh, cols)
        self.speeds = [0.995, 0.99, 0.985, 0.975, 0.96]  # zoom per frame
        self.zoom_step = self.speeds[2]
        self.t = 0
        self.target = 0
        self.start(0)

    def start(self, i):
        self.target = i % len(TARGETS)
        self.cx, self.cy, self.width = -0.6, 0.0, 3.2
        self.tx, self.ty = TARGETS[self.target]
        return f"diving into place {self.target + 1} of {len(TARGETS)}"

    def reset(self):
        self.start(self.target + 1)

    def set_speed(self, i):
        self.zoom_step = self.speeds[i]

    def maxit(self):
        z = 3.2 / self.width
        return 256 if z < 50 else 1024 if z < 5000 else 4096

    def key(self):
        return ("mandel", self.maxit())

    def step(self):
        self.m.maxit = self.maxit()
        self.counts, ms = self.m.run(self.cx, self.cy, self.width)
        self.cx += (self.tx - self.cx) * 0.04
        self.cy += (self.ty - self.cy) * 0.04
        self.width *= self.zoom_step
        if self.width < self.vw * 2 ** -FRAC * 4:  # Q28 runs out of bits: next place
            self.reset()
        self.t += 1
        return ms, 1

    def cells(self):
        return self.vw * self.vh

    def image(self):
        n = self.counts
        x = (np.log1p(n) * 40 + self.t * 2).astype(np.int64) % 256
        rgb = RAINBOW[x]
        rgb[n >= self.m.maxit] = 0
        return rgb

    def plane(self, x, y):
        return (self.cx + (x / self.vw - 0.5) * self.width,
                self.cy + (y / self.vh - 0.5) * self.width * self.vh / self.vw)

    def paint(self, x, y, left):
        if left:
            self.tx, self.ty = self.plane(x, y)
        else:
            self.width = min(3.2, self.width * 1.05)

    def press(self, e, x, y):
        if e.key in (pygame.K_n, pygame.K_r):
            return self.start(self.target + 1)
        return None

    def status(self):
        return f"  |  zoom {3.2 / self.width:,.0f}x  |  maxit {self.m.maxit}"


x = np.linspace(0, 2 * np.pi, 256, endpoint=False)
RAINBOW = (np.stack([0.5 + 0.5 * np.sin(x), 0.5 + 0.5 * np.sin(x + 2.1), 0.5 + 0.5 * np.sin(x + 4.2)], -1)
           * 255).astype(np.uint8)
del x

# ---------------------------------------------------------------------------------------------

EFFECTS = ["life", "brain", "wire", "gray", "mandel"]
# demo scenes: (effect, start, speed index)
SCENES = [("life", "soup", 1), ("gray", "coral", 2), ("life", "g", 2), ("mandel", 0, 2), ("brain", "soup", 0),
          ("gray", "mitosis", 3), ("life", "a", 3), ("wire", "soup", 0), ("mandel", 1, 2), ("gray", "worms", 2),
          ("life", "l", 3), ("mandel", 2, 2), ("gray", "spots", 3), ("life", "p", 4), ("mandel", 3, 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit", action="store_true", help="fullscreen")
    p.add_argument("--demo", type=float, nargs="?", const=25, metavar="SECONDS",
                   help="show mode: a new scene every SECONDS (default 25)")
    p.add_argument("--effect", choices=EFFECTS, default="life")
    p.add_argument("--fade", type=float, default=0.93)
    args = p.parse_args()

    dev = iron.get_current_device()
    pygame.init()
    dw, dh = pygame.display.get_desktop_sizes()[0]
    if args.fit:
        screen = pygame.display.set_mode((dw, dh), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((int(dw * 0.85), int(dh * 0.85)), pygame.RESIZABLE)
    sw, sh = screen.get_size()
    pygame.display.set_caption(f"xdna-playground: {dev.cols * ROWS} NPU cores")
    font = pygame.font.SysFont("monospace", 18, bold=True)
    big = pygame.font.SysFont("monospace", 40, bold=True)
    clock = pygame.time.Clock()

    made = {}

    def effect(name):
        if name not in made:
            if name in ("life", "brain", "wire"):
                made[name] = Life(sw, sh, dev.cols, name, args.fade)
            elif name == "gray":
                made[name] = Reaction(sw, sh, dev.cols)
            else:
                made[name] = Fractal(sw, sh, dev.cols)
        return made[name]

    cur, cur_name = effect(args.effect), args.effect
    speed = 1
    cur.set_speed(speed)
    note = {"text": "h: help   tab: next effect   m: demo", "until": time.perf_counter() + 6,
            "title_until": time.perf_counter() + 4}

    def say(text, seconds=2.5):
        if text:
            note["text"], note["until"] = text, time.perf_counter() + seconds

    def switch(name, what=None, sp=None):
        nonlocal cur, cur_name, speed
        cur, cur_name = effect(name), name
        if sp is not None:
            speed = sp
        cur.set_speed(speed)
        if what is not None:
            say(cur.start(what))
        note["title_until"] = time.perf_counter() + 4

    def mouse_cell():
        r = place(screen, cur.vw, cur.vh)
        mx, my = pygame.mouse.get_pos()
        return (mx - r.x) * cur.vw // max(1, r.w), (my - r.y) * cur.vh // max(1, r.h)

    paused, single, show_help = False, False, False
    compiled = set()
    demo, demo_i, next_scene = args.demo, 0, time.perf_counter()
    t_rate, steps_done, npu_ms, rate, busy = time.perf_counter(), 0, 0.0, 0.0, 0.0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                running = False
            elif e.type == pygame.KEYDOWN:
                x, y = mouse_cell()
                if e.key == pygame.K_TAB:
                    switch(EFFECTS[(EFFECTS.index(cur_name) + 1) % len(EFFECTS)])
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_s:
                    single = True
                elif e.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_h:
                    show_help = not show_help
                elif e.key == pygame.K_m:
                    demo = None if demo else (args.demo or 25)
                    next_scene = time.perf_counter()
                    say("demo mode " + ("on" if demo else "off"))
                elif pygame.K_1 <= e.key <= pygame.K_5:
                    speed = e.key - pygame.K_1
                    cur.set_speed(speed)
                else:
                    say(cur.press(e, x, y))

        if demo and time.perf_counter() >= next_scene:
            name, what, sp = SCENES[demo_i % len(SCENES)]
            switch(name, what, sp)
            demo_i += 1
            next_scene = time.perf_counter() + demo

        buttons = pygame.mouse.get_pressed()
        if buttons[0] or buttons[2]:
            x, y = mouse_cell()
            cur.paint(x, y, bool(buttons[0]))

        if not paused or single:
            if cur.key() not in compiled:
                hud(screen, font, "compiling a kernel for this effect and speed ...")
                pygame.display.flip()
                compiled.add(cur.key())
            ms, steps = cur.step()
            npu_ms += ms
            steps_done += steps
            single = False

        now = time.perf_counter()
        if now - t_rate > 0.5:
            rate, busy = steps_done / (now - t_rate), npu_ms / 1000 / (now - t_rate)
            steps_done, npu_ms, t_rate = 0, 0.0, now

        img = cur.image()
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        r = place(screen, cur.vw, cur.vh)
        screen.fill((0, 0, 0))
        scale = pygame.transform.smoothscale if cur.smooth else pygame.transform.scale
        screen.blit(scale(surf, r.size), r.topleft)

        cells = f"  |  {cur.cells() * rate / 1e9:.2f} Gcell/s" if cur.unit != "frame" else ""
        hud(screen, font, f"{cur.title}  |  {rate:,.0f} {cur.unit}s/s  |  NPU busy {busy:4.0%}{cells}"
                          f"{cur.status()}  |  {cur.vw}x{cur.vh} on {dev.cols * ROWS} cores"
                          + ("  |  PAUSED" if paused else "") + ("  |  DEMO" if demo else ""))
        if now < note["until"]:
            hud(screen, font, note["text"], y=44)
        if now < note["title_until"]:
            text = f"{cur.title} on {dev.cols * ROWS} NPU cores"
            w, h = big.size(text)
            hud(screen, big, text, x=(screen.get_width() - w) // 2 - 10, y=screen.get_height() - h - 50)
        if show_help:
            for i, line in enumerate((__doc__ or "").strip().splitlines()[2:]):
                hud(screen, font, line, y=80 + i * 28)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
