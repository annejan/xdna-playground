#!/usr/bin/env python3
"""The show: the NPU computes, the Radeon iGPU draws it, in one window made for a party table.

  ../env.sh python show.py --fit --demo        fullscreen, a new scene every 25 seconds
  ../env.sh python show.py                     in a window, start with Life
  ../env.sh python show.py --cpu-render        draw with numpy + pygame instead of the GPU

  tab        next effect: Life, Brian's Brain, Wireworld, Gray-Scott, Mandelbrot
  1 .. 5     speed (what a step is depends on the effect; the first use of a speed compiles it)
  m          demo mode on / off
  space      pause          r  new start          c  clear          f  fullscreen
  h          help           q, esc  quit

  Life & co    mouse draws; g n p a d l w x v u stamp patterns (shift: 12); e: electron (Wireworld)
  Gray-Scott   mouse left drops chemical, right wipes; p: next preset (coral, mitosis, worms, ...)
  Mandelbrot   mouse left: zoom towards there; right: zoom out; n: next place to dive into
  Julia        mouse left: take c in your hand and drag it; right: let go; n: somewhere else on the circle
  Lenia        mouse left: drop an Orbium; right: wipe a spot; r: 14 new ones
"""

import argparse
import math
import time

import numpy as np
import pygame

import aie.iron as iron
from gl import Renderer
from gray_chip import ONE, PRESETS, Gray
from gray_chip import seed as gray_seed
from lenia_chip import ONE as LONE
from lenia_chip import ORBIUM
from lenia_chip import Lenia as LeniaChip
from lenia_chip import creatures, decode
from life_chip import ROWS, Chip
from life_window import FIRE, ICE, SPEEDS, TITLES, World, colorize, fit_world, hud, lut, place
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
        self.speed_i, self.bloom = 2, 1.0
        self.lut = {"life": FIRE, "brain": ICE, "wire": FIRE}[rule]

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

    def draw(self, r, rect):
        r.ca(self.world.cells[:self.vh, :self.vw], self.chip.rule, self.lut, self.fade, rect, key=self.chip.rule)


PALETTES = {
    "spirals": lut([(0.0, (4, 8, 28)), (0.3, (10, 50, 100)), (0.55, (30, 150, 170)), (0.8, (200, 235, 210)),
                    (1.0, (255, 255, 255))]),
    "waves": lut([(0.0, (0, 0, 10)), (0.3, (30, 20, 120)), (0.6, (120, 80, 230)), (0.85, (220, 200, 255)),
                  (1.0, (255, 255, 255))]),
    "bubbles": lut([(0.0, (10, 2, 20)), (0.35, (80, 10, 90)), (0.6, (210, 50, 140)), (0.85, (255, 190, 210)),
                    (1.0, (255, 255, 255))]),
    "pulse": lut([(0.0, (20, 10, 0)), (0.35, (110, 50, 0)), (0.6, (230, 140, 20)), (0.85, (255, 230, 150)),
                  (1.0, (255, 255, 255))]),
    "coral": lut([(0.0, (5, 15, 5)), (0.35, (20, 80, 30)), (0.6, (120, 190, 40)), (0.85, (240, 240, 140)),
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
        self.speed_i, self.bloom = 3, 0.45
        self.g = Gray(self.vw, self.vh, self.speeds[self.speed_i], cols, "spirals")
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

    def draw(self, r, rect):
        r.gray(self.v[:self.vh, :self.vw], self.g.preset, PALETTES[self.g.preset], rect, top=0.45)


LENIA_LUT = lut([(0.0, (2, 4, 16)), (0.25, (20, 30, 110)), (0.5, (40, 140, 200)), (0.75, (190, 240, 160)),
                 (1.0, (255, 255, 230))])


class LeniaFx:
    """Lenia: Orbium gliders in continuous Life; 412 multiply-adds per cell per step on the NPU.
    The world is 13 rows per core tall and 256 wide, so it is drawn lying down. Click: a new Orbium."""

    smooth = True
    unit = "step"
    title = "Lenia"

    def __init__(self, sw, sh, cols):
        self.l = LeniaChip(256, 7, cols)
        self.vw, self.vh = self.l.h, self.l.w  # lying down: world rows run left to right
        self.speeds = [1, 3, 7, 15, 31]
        self.speed_i, self.bloom = 2, 0.6
        self.rng = np.random.default_rng()
        self.orbium = np.round(decode(ORBIUM) * LONE).astype(np.int16)
        self.last_drop = 0.0
        self.reset()

    def reset(self):
        self.a = creatures(self.l.h, self.l.w, self.rng, count=14)

    def start(self, what=None):
        self.reset()
        return "14 Orbia, gliding"

    def set_speed(self, i):
        self.l.steps = self.speeds[i]

    def key(self):
        return ("lenia", self.l.steps)

    def step(self):
        self.a, ms = self.l.run(self.a)
        return ms, self.l.steps

    def cells(self):
        return self.l.w * self.l.h

    def image(self):
        return LENIA_LUT[np.clip(self.a.T.astype(np.int32) * 255 // LONE, 0, 255)]

    def drop(self, row, col):
        s = np.rot90(self.orbium, int(self.rng.integers(0, 4)))
        ys = (np.arange(s.shape[0]) + row - s.shape[0] // 2) % self.l.h
        xs = (np.arange(s.shape[1]) + col - s.shape[1] // 2) % self.l.w
        self.a[np.ix_(ys, xs)] = np.maximum(self.a[np.ix_(ys, xs)], s)

    def paint(self, x, y, left):
        now = time.perf_counter()
        if now - self.last_drop < 0.3:
            return
        self.last_drop = now
        if left:
            self.drop(x, y)  # screen x is the world's row, screen y its column
        else:
            ys = (np.arange(-15, 16) + x) % self.l.h
            xs = (np.arange(-15, 16) + y) % self.l.w
            self.a[np.ix_(ys, xs)] = 0

    def press(self, e, x, y):
        if e.key == pygame.K_r:
            return self.start()
        if e.key == pygame.K_c:
            self.a[:] = 0
            return "empty: click to drop Orbia"
        return None

    def status(self):
        return f"  |  mass {self.a.sum() / LONE:.0f}"

    def draw(self, r, rect):
        r.gray(self.a, "lenia", LENIA_LUT, rect, top=1.0, rotate=True, key="lenia:a")


TARGETS = [(-0.743643887037151, 0.131825904205330), (-0.761574, -0.0847596), (-1.768778833, -0.001738996),
           (-0.77568377, 0.13646737), (-0.7746806106269039, -0.1374168856037867)]


class Fractal:
    """Mandelbrot: dive into one of TARGETS, palette cycling like it is 1992."""

    smooth = False
    unit = "frame"
    title = "Mandelbrot"

    def __init__(self, sw, sh, cols):
        s = max(1, math.ceil(math.sqrt(sw * sh / 450_000)))  # about 450k pixels: a frame in ~15 ms
        self.vw, self.vh = sw // s, sh // s
        self.m = Mandel(self.vw, self.vh, cols)
        self.speeds = [0.75, 0.6, 0.45, 0.3, 0.15]  # width left after a second: zoom by time, not by frame
        self.speed_i, self.bloom = 2, 0.5
        self.zoom_step = self.speeds[self.speed_i]
        self.last = None
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

    LEVELS = (512, 1024, 2048)

    def maxit(self):
        z = 3.2 / self.width
        return self.LEVELS[0] if z < 50 else self.LEVELS[1] if z < 5000 else self.LEVELS[2]

    def warm(self):
        """Compile every maxit level now, not in the middle of a dive."""
        for m in self.LEVELS:
            self.m.maxit = m
            self.m.run(-0.6, 0.0, 3.2)
        return [("mandel", m) for m in self.LEVELS]

    def key(self):
        return ("mandel", self.maxit())

    def step(self):
        self.m.maxit = self.maxit()
        self.counts, ms = self.m.run(self.cx, self.cy, self.width)
        now = time.perf_counter()
        dt = min(0.1, now - self.last) if self.last else 1 / 60
        self.last = now
        ease = 1 - 0.08 ** dt  # glide to the target, the same speed whatever the frame rate
        self.cx += (self.tx - self.cx) * ease
        self.cy += (self.ty - self.cy) * ease
        self.width *= self.zoom_step ** dt
        if self.width < self.vw * 2 ** -FRAC * 4:  # Q28 runs out of bits: next place
            self.reset()
        self.t += dt * 60
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
            self.width = min(3.2, self.width * 1.03)

    def press(self, e, x, y):
        if e.key in (pygame.K_n, pygame.K_r):
            return self.start(self.target + 1)
        return None

    def status(self):
        return f"  |  zoom {3.2 / self.width:,.0f}x  |  maxit {self.m.maxit}"

    def draw(self, r, rect):
        r.mandel(self.counts, self.m.maxit, self.t * 0.004, rect)


class Julia:
    """Julia sets that morph: c walks just outside the edge of the Mandelbrot set's main cardioid,
    where the Julia sets are intricate (measured: 25-35% of the picture is filigree), and rushes past
    the cusp at c = 1/4, where they turn to dust. Every frame a new fractal.
    Mouse left: take c in your hand (it follows the mouse over the complex plane); right: let go."""

    smooth = False
    unit = "frame"
    title = "Julia sets"
    MAXIT = 256

    def __init__(self, sw, sh, cols):
        s = max(1, math.ceil(math.sqrt(sw * sh / 450_000)))
        self.vw, self.vh = sw // s, sh // s
        self.m = Mandel(self.vw, self.vh, cols, self.MAXIT, julia=True)
        self.speeds = [0.04, 0.08, 0.15, 0.3, 0.6]  # radians per second along the circle
        self.speed_i, self.bloom = 2, 0.55
        self.angle, self.t, self.last, self.hand = 0.0, 0.0, None, None
        self.width = 3.3

    def start(self, what=None):
        self.angle, self.hand = float(np.random.default_rng().uniform(0.5, 2 * np.pi - 0.5)), None
        return "c walks along the edge of the Mandelbrot set"

    def reset(self):
        self.start()

    def set_speed(self, i):
        self.speed = self.speeds[i]

    def key(self):
        return ("julia", self.MAXIT)

    def warm(self):
        self.m.run(0.0, 0.0, self.width)
        return [self.key()]

    def c(self):
        if self.hand is not None:
            return self.hand
        t = self.angle
        p = complex(math.cos(t), math.sin(t)) / 2 - complex(math.cos(2 * t), math.sin(2 * t)) / 4
        out = p - 0.25
        p += 0.04 * out / abs(out) if abs(out) > 1e-9 else 0
        return p.real, p.imag

    def step(self):
        now = time.perf_counter()
        dt = min(0.1, now - self.last) if self.last else 1 / 60
        self.last = now
        if self.hand is None:
            self.angle += self.speed * dt * (5 if math.cos(self.angle) > 0.95 else 1)  # hurry past the cusp
        cx, cy = self.c()
        self.m.c = (round(cx * (1 << FRAC)), round(cy * (1 << FRAC)))
        self.counts, ms = self.m.run(0.0, 0.0, self.width)
        self.t += dt * 60
        return ms, 1

    def cells(self):
        return self.vw * self.vh

    def image(self):
        n = self.counts
        rgb = RAINBOW[(np.log1p(n) * 40 + self.t * 2).astype(np.int64) % 256]
        rgb[n >= self.MAXIT] = 0
        return rgb

    def paint(self, x, y, left):
        if left:  # the mouse picks c, on a plane 3 wide
            self.hand = ((x / self.vw - 0.5) * 3.0, (y / self.vh - 0.5) * 3.0 * self.vh / self.vw)
        else:
            self.hand = None

    def press(self, e, x, y):
        if e.key in (pygame.K_n, pygame.K_r):
            return self.start()
        return None

    def status(self):
        cx, cy = self.c()
        return f"  |  c = {cx:+.4f} {cy:+.4f}i" + ("  (in your hand)" if self.hand else "")

    def draw(self, r, rect):
        r.mandel(self.counts, self.MAXIT, self.t * 0.004, rect)


x = np.linspace(0, 2 * np.pi, 256, endpoint=False)
RAINBOW = (np.stack([0.5 + 0.5 * np.sin(x), 0.5 + 0.5 * np.sin(x + 2.1), 0.5 + 0.5 * np.sin(x + 4.2)], -1)
           * 255).astype(np.uint8)
del x

# ---------------------------------------------------------------------------------------------

EFFECTS = ["life", "brain", "wire", "gray", "lenia", "mandel", "julia"]
# demo scenes: (effect, start, speed index)
SCENES = [("lenia", None, 2), ("julia", None, 2), ("gray", "spirals", 4), ("life", "soup", 2), ("mandel", 0, 2), ("gray", "waves", 4),
          ("life", "g", 3), ("brain", "soup", 1), ("julia", None, 3), ("gray", "bubbles", 4), ("mandel", 1, 2),
          ("life", "a", 4), ("wire", "soup", 1), ("gray", "spirals", 3), ("julia", None, 2), ("mandel", 2, 3),
          ("lenia", None, 3), ("life", "l", 4), ("gray", "waves", 3), ("mandel", 3, 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit", action="store_true", help="fullscreen")
    p.add_argument("--demo", type=float, nargs="?", const=25, metavar="SECONDS",
                   help="show mode: a new scene every SECONDS (default 25)")
    p.add_argument("--effect", choices=EFFECTS, default="life")
    p.add_argument("--fade", type=float, default=0.93)
    p.add_argument("--cpu-render", action="store_true", help="draw with numpy + pygame, not the GPU")
    args = p.parse_args()
    use_gl = not args.cpu_render

    dev = iron.get_current_device()
    pygame.init()
    dw, dh = pygame.display.get_desktop_sizes()[0]
    size = (dw, dh) if args.fit else (int(dw * 0.85), int(dh * 0.85))
    flags = pygame.FULLSCREEN if args.fit else (0 if use_gl else pygame.RESIZABLE)
    if use_gl:
        import moderngl
        screen = pygame.display.set_mode(size, flags | pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
        ctx = moderngl.create_context()
        gpu = Renderer(ctx, size)
        gpu_name = ctx.info["GL_RENDERER"].split(" (")[0].replace("AMD ", "").replace(" Graphics", "")
    else:
        screen = pygame.display.set_mode(size, flags)
    sw, sh = screen.get_size()

    def text(f, line, y=8, x=8):
        """HUD line: on the screen surface, or as a texture over the GL picture."""
        if not use_gl:
            return hud(screen, f, line, y=y, x=x)
        label = f.render(line, True, (255, 240, 200))
        box = pygame.Surface((label.get_width() + 20, label.get_height() + 10), pygame.SRCALPHA)
        box.fill((0, 0, 0, 170))
        box.blit(label, (10, 5))
        gpu.text(ctx.screen, box, x, y)
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
            elif name == "julia":
                made[name] = Julia(sw, sh, dev.cols)
            elif name == "lenia":
                made[name] = LeniaFx(sw, sh, dev.cols)
            else:
                made[name] = Fractal(sw, sh, dev.cols)
        return made[name]

    cur, cur_name = effect(args.effect), args.effect
    cur.set_speed(cur.speed_i)
    note = {"text": "h: help   tab: next effect   m: demo", "until": time.perf_counter() + 6,
            "title_until": time.perf_counter() + 4}

    def say(text, seconds=2.5):
        if text:
            note["text"], note["until"] = text, time.perf_counter() + seconds

    def switch(name, what=None, sp=None):
        nonlocal cur, cur_name
        cur, cur_name = effect(name), name
        if sp is not None:
            cur.speed_i = sp
        cur.set_speed(cur.speed_i)
        if what is not None or name in ("julia", "lenia"):
            say(cur.start(what))
        note["title_until"] = time.perf_counter() + 4

    def mouse_cell():
        r = place(screen, cur.vw, cur.vh)
        mx, my = pygame.mouse.get_pos()
        return (mx - r.x) * cur.vw // max(1, r.w), (my - r.y) * cur.vh // max(1, r.h)

    paused, single, show_help = False, False, False
    compiled = set()

    def warm_up():
        """Compile every kernel the demo needs before it starts, so the show never stops to compile."""
        todo = list(dict.fromkeys((n, w, sp) for n, w, sp in SCENES))
        for i, (n, w, sp) in enumerate(todo):
            if use_gl:
                ctx.screen.use()
                ctx.clear(0.0, 0.0, 0.0)
            else:
                screen.fill((0, 0, 0))
            text(big, f"warming up the NPU: {i + 1} / {len(todo)}", x=40, y=sh // 2 - 30)
            pygame.display.flip()
            pygame.event.pump()
            switch(n, w, sp)
            if n in ("mandel", "julia"):
                compiled.update(cur.warm())
            elif cur.key() not in compiled:
                cur.step()
                compiled.add(cur.key())

    if args.demo:
        warm_up()
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
                    if use_gl:
                        say("fullscreen: start with --fit (the GPU window cannot switch)")
                    else:
                        pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_h:
                    show_help = not show_help
                elif e.key == pygame.K_m:
                    demo = None if demo else (args.demo or 25)
                    next_scene = time.perf_counter()
                    say("demo mode " + ("on" if demo else "off"))
                elif pygame.K_1 <= e.key <= pygame.K_5:
                    cur.speed_i = e.key - pygame.K_1
                    cur.set_speed(cur.speed_i)
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
                if use_gl:
                    ctx.screen.use()
                    ctx.clear(0.0, 0.0, 0.0)
                text(font, "compiling a kernel for this effect and speed ...")
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

        r = place(screen, cur.vw, cur.vh)
        if use_gl:
            gpu.begin()
            cur.draw(gpu, tuple(r))
            gpu.finish(ctx.screen, strength=cur.bloom)
        else:
            img = cur.image()
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
            screen.fill((0, 0, 0))
            scale = pygame.transform.smoothscale if cur.smooth else pygame.transform.scale
            screen.blit(scale(surf, r.size), r.topleft)

        cells = f"  |  {cur.cells() * rate / 1e9:.2f} Gcell/s" if cur.unit != "frame" else ""
        drawn = f"  |  drawn by {gpu_name}" if use_gl else ""
        text(font, f"{cur.title}  |  {rate:,.0f} {cur.unit}s/s  |  NPU busy {busy:4.0%}{cells}"
                   f"{cur.status()}  |  {cur.vw}x{cur.vh} on {dev.cols * ROWS} NPU cores{drawn}"
                   + ("  |  PAUSED" if paused else "") + ("  |  DEMO" if demo else ""))
        if now < note["until"]:
            text(font, note["text"], y=44)
        if now < note["title_until"]:
            line = f"{cur.title}  ·  {dev.cols * ROWS} NPU cores" + (" + GPU" if use_gl else "")
            w, h = big.size(line)
            text(big, line, x=(sw - w) // 2 - 10, y=sh - h - 50)
        if show_help:
            for i, line in enumerate((__doc__ or "").strip().splitlines()[2:]):
                text(font, line, y=80 + i * 28)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
