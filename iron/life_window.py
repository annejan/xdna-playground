#!/usr/bin/env python3
"""Cellular automata on 32 NPU cores, in a window: Life, Brian's Brain and Wireworld.

  ../env.sh python life_window.py              1024x768 world
  ../env.sh python life_window.py --fit        fullscreen, the world sized to fill the screen
  ../env.sh python life_window.py --demo 25    show mode: a new scene every 25 seconds
  ../env.sh python life_window.py --rule wire

  tab                  next rule: life, brain, wire
  mouse left / right   draw / erase (wire: draws wire)
  e                    wire: an electron at the mouse
  space                pause, s = one step while paused
  1 2 3 4 5            1, 3, 9, 33 or 101 generations per frame (the first use compiles it)
  r / c                new start / clear
  g n p a d l w x v u  life: stamp a pattern at the mouse (h lists them); with shift: a fleet of 12
  t                    turn the next stamps by 90 degrees
  m                    demo mode on / off
  f                    fullscreen
  q, esc               quit
"""

import argparse
import math
import time

import numpy as np
import pygame

import aie.iron as iron
from life_chip import BAND_MAX, ROWS, Chip
from patterns import SHAPES, stamp
from rules import seed

SPEEDS = [1, 3, 9, 33, 101]
ORDER = ["life", "brain", "wire"]
TITLES = {"life": "Conway's Life", "brain": "Brian's Brain", "wire": "Wireworld"}


def lut(stops):
    x = np.linspace(0, 1, 256)
    pos = [s[0] for s in stops]
    return np.stack([np.interp(x, pos, [s[1][i] for s in stops]) for i in range(3)], -1).astype(np.uint8)


# inferno-ish fire for Life, ice for Brian's Brain
FIRE = lut([(0.0, (0, 0, 4)), (0.15, (31, 12, 72)), (0.35, (85, 15, 109)), (0.55, (136, 34, 106)),
            (0.7, (186, 54, 85)), (0.82, (227, 89, 51)), (0.92, (249, 140, 10)), (1.0, (249, 201, 50))])
ICE = lut([(0.0, (0, 0, 6)), (0.3, (8, 20, 70)), (0.6, (20, 70, 170)), (0.85, (60, 160, 240)), (1.0, (150, 230, 255))])


def colorize(rule, cells, heat, fade):
    heat *= fade
    heat[cells == 1] = 1.0
    if rule == "life":
        rgb = FIRE[(heat * 255).astype(np.uint8)]
        rgb[cells == 1] = (252, 255, 164)
    elif rule == "brain":
        rgb = ICE[(heat * 255).astype(np.uint8)]
        rgb[cells == 2] = (70, 110, 255)
        rgb[cells == 1] = (235, 250, 255)
    else:
        # copper wire that glows where electrons went by; heads electric blue, tails red
        rgb = np.zeros(cells.shape + (3,), np.uint8)
        wire = cells == 3
        glow = heat[wire][:, None]
        rgb[wire] = np.clip(np.array([110, 62, 22]) + glow * np.array([145, 150, 120]), 0, 255)
        rgb[cells == 2] = (255, 70, 40)
        rgb[cells == 1] = (140, 220, 255)
    return rgb


def fit_world(dw, dh, cores):
    """The biggest world that fills a dw x dh screen at a whole scale and still fits on the chip:
    W a multiple of 64 (the vector kernel), and a band of ceil(H / cores) rows within BAND_MAX."""
    for s in range(1, 17):
        w, h = (dw // s) // 64 * 64, dh // s
        if w >= 64 and math.ceil(h / cores) * w <= BAND_MAX:
            return w, h
    raise SystemExit(f"no world fits a {dw}x{dh} screen")


def place(screen, vw, vh):
    """Where the vw x vh world goes on the screen: as big as fits, same aspect ratio, centred.
    A whole scale when it is at least 2x, so cells stay square and sharp."""
    sw, sh = screen.get_size()
    f = min(sw / vw, sh / vh)
    if f >= 2:
        f = math.floor(f)
    w, h = int(vw * f), int(vh * f)
    return pygame.Rect((sw - w) // 2, (sh - h) // 2, w, h)


class World:
    """The cells on the host and what the window shows of them (vw x vh, top left)."""

    def __init__(self, chip, vw, vh, fill, rng):
        self.chip, self.vw, self.vh, self.fill, self.rng = chip, vw, vh, fill, rng
        self.cells = seed(chip.rule, (chip.h, chip.w), rng, fill)
        self.heat = np.zeros(self.cells.shape, np.float32)

    def set_rule(self, rule):
        self.chip.rule = rule
        self.reset()

    def reset(self):
        self.cells = seed(self.chip.rule, self.cells.shape, self.rng, self.fill)
        self.heat[:] = 0

    def clear(self):
        self.cells[:] = 0
        self.heat[:] = 0

    def fleet(self, key, x=None, y=None, turn=0):
        name, shape = SHAPES[key]
        x = self.vw // 8 if x is None else x
        y = self.vh // 6 if y is None else y
        for i in range(4):
            for j in range(3):
                stamp(self.cells, shape, x + i * self.vw // 4, y + j * self.vh // 3, turn)
        return f"12 x {name}"


# demo scenes: (rule, start: "soup" or a pattern key for a fleet of 12, speed index)
SCENES = [("life", "soup", 1), ("life", "g", 2), ("brain", "soup", 0), ("life", "a", 3),
          ("wire", "soup", 0), ("life", "l", 3), ("brain", "soup", 1), ("life", "n", 2), ("life", "p", 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", nargs=2, type=int, default=(1024, 768), metavar=("W", "H"))
    p.add_argument("--fill", type=float, default=0.3)
    p.add_argument("--fade", type=float, default=0.93, help="how slowly trails cool down (0..1)")
    p.add_argument("--fit", action="store_true", help="fullscreen, with the world sized to fill the screen")
    p.add_argument("--rule", choices=ORDER, default="life")
    p.add_argument("--demo", type=float, nargs="?", const=25, metavar="SECONDS",
                   help="show mode: a new scene every SECONDS (default 25)")
    args = p.parse_args()

    dev = iron.get_current_device()
    pygame.init()
    dw, dh = pygame.display.get_desktop_sizes()[0]
    speed = 0
    if args.fit:
        vw, vh = fit_world(dw, dh, dev.cols * ROWS)
        chip = Chip(vw, vh, SPEEDS[speed], dev.cols, args.rule)  # chip.h may be a few rows more: off screen
    else:
        chip = Chip(*args.size, SPEEDS[speed], dev.cols, args.rule)
        vw, vh = chip.w, chip.h
    world = World(chip, vw, vh, args.fill, np.random.default_rng())

    if args.fit:
        screen = pygame.display.set_mode((dw, dh), pygame.FULLSCREEN)
    else:
        scale = max(1, min(int(dw * 0.9) // vw, int(dh * 0.85) // vh))
        screen = pygame.display.set_mode((vw * scale, vh * scale), pygame.RESIZABLE)
    pygame.display.set_caption(f"Cellular automata on {chip.cores} NPU cores — {vw}x{vh}")
    font = pygame.font.SysFont("monospace", 18, bold=True)
    big = pygame.font.SysFont("monospace", 40, bold=True)
    clock = pygame.time.Clock()

    state = {"message": "h: help   tab: next rule", "message_until": time.perf_counter() + 5,
             "title_until": time.perf_counter() + 4}

    def say(text, seconds=2.5):
        state["message"], state["message_until"] = text, time.perf_counter() + seconds

    def mouse_cell():
        r = place(screen, vw, vh)
        mx, my = pygame.mouse.get_pos()
        return (mx - r.x) * vw // max(1, r.w), (my - r.y) * vh // max(1, r.h)

    def scene(i):
        rule, start, sp = SCENES[i % len(SCENES)]
        chip.gens = SPEEDS[sp]
        world.set_rule(rule)
        if start != "soup":
            world.clear()
            say(world.fleet(start))
        state["title_until"] = time.perf_counter() + 4

    paused, step, gen, gens_done, t_rate, ms, rate = False, False, 0, 0, time.perf_counter(), 0.0, 0.0
    compiled = set()
    turn, show_help = 0, False
    demo, demo_i = args.demo, 0
    next_scene = time.perf_counter()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_TAB:
                    world.set_rule(ORDER[(ORDER.index(chip.rule) + 1) % len(ORDER)])
                    state["title_until"] = time.perf_counter() + 3
                elif e.key == pygame.K_s:
                    step = True
                elif e.key == pygame.K_r:
                    world.reset()
                elif e.key == pygame.K_c:
                    world.clear()
                elif e.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_h:
                    show_help = not show_help
                elif e.key == pygame.K_m:
                    demo = None if demo else (args.demo or 25)
                    next_scene = time.perf_counter()
                    say("demo mode " + ("on" if demo else "off"))
                elif e.key == pygame.K_t:
                    turn = (turn + 1) % 4
                    say(f"stamps turned {turn * 90} degrees")
                elif e.key == pygame.K_e and chip.rule == "wire":
                    x, y = mouse_cell()
                    world.cells[y % chip.h, x % chip.w] = 1
                    world.cells[y % chip.h, (x - 1) % chip.w] = 2
                elif chip.rule == "life" and e.unicode.lower() in SHAPES:
                    key = e.unicode.lower()
                    x, y = mouse_cell()
                    if e.mod & pygame.KMOD_SHIFT:
                        say(world.fleet(key, x, y, turn))
                    else:
                        stamp(world.cells, SHAPES[key][1], x, y, turn)
                        say(SHAPES[key][0])
                elif pygame.K_1 <= e.key <= pygame.K_5:
                    chip.gens = SPEEDS[e.key - pygame.K_1]

        if demo and time.perf_counter() >= next_scene:
            scene(demo_i)
            demo_i += 1
            next_scene = time.perf_counter() + demo

        # draw with the mouse: live cells, or wire in Wireworld
        buttons = pygame.mouse.get_pressed()
        if buttons[0] or buttons[2]:
            x, y = mouse_cell()
            radius = 0 if chip.rule == "wire" else 3
            ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            disc = xs * xs + ys * ys <= radius * radius
            yy, xx = (ys + y) % chip.h, (xs + x) % chip.w
            pen = (3 if chip.rule == "wire" else 1) if buttons[0] else 0
            world.cells[yy, xx] = np.where(disc, pen, world.cells[yy, xx])

        if not paused or step:
            if (chip.rule, chip.gens) not in compiled:
                hud(screen, font, f"compiling the {chip.rule} kernel for {chip.gens} generations per call ...")
                pygame.display.flip()
                compiled.add((chip.rule, chip.gens))
            world.cells, ms = chip.run(world.cells)
            gen += chip.gens
            gens_done += chip.gens
            step = False

        now = time.perf_counter()
        if now - t_rate > 0.5:
            rate, gens_done, t_rate = gens_done / (now - t_rate), 0, now

        rgb = colorize(chip.rule, world.cells, world.heat, args.fade)
        surf = pygame.surfarray.make_surface(rgb[:vh, :vw].swapaxes(0, 1))
        r = place(screen, vw, vh)
        screen.fill((0, 0, 0))
        screen.blit(pygame.transform.scale(surf, r.size), r.topleft)
        hud(screen, font,
            f"{TITLES[chip.rule]}  |  gen {gen:,}  |  {rate:,.0f} gen/s  |  {chip.gens} per frame  |  "
            f"NPU {ms:.1f} ms/call  |  {chip.w * chip.h * rate / 1e9:.2f} Gcell/s  |  {vw}x{vh}"
            + ("  |  PAUSED" if paused else "") + ("  |  DEMO" if demo else ""))
        if now < state["message_until"]:
            hud(screen, font, state["message"], y=44)
        if now < state["title_until"]:
            text = f"{TITLES[chip.rule]} on {chip.cores} NPU cores"
            w, h = big.size(text)
            hud(screen, big, text, x=(screen.get_width() - w) // 2 - 10, y=screen.get_height() - h - 50)
        if show_help:
            lines = [f"{k}  {name}  ({shape.shape[1]}x{shape.shape[0]})" for k, (name, shape) in SHAPES.items()]
            lines += ["tab  next rule", "e  electron (wire)", "m  demo mode", "1-5  speed"]
            for i, line in enumerate(lines):
                hud(screen, font, line, y=80 + i * 30)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def hud(screen, font, text, y=8, x=8):
    label = font.render(text, True, (255, 240, 200))
    box = pygame.Surface((label.get_width() + 20, label.get_height() + 10), pygame.SRCALPHA)
    box.fill((0, 0, 0, 170))
    screen.blit(box, (x, y))
    screen.blit(label, (x + 10, y + 5))


if __name__ == "__main__":
    main()
