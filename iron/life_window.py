#!/usr/bin/env python3
"""Game of Life on 32 NPU cores, in a window.

  ../env.sh python life_window.py              1024x768 world
  ../env.sh python life_window.py --size 1536 512
  ../env.sh python life_window.py --fit          fullscreen, the world sized to fill the screen

  mouse left / right   draw / erase cells
  space                pause, s = one step while paused
  1 2 3 4 5            1, 3, 9, 33 or 101 generations per frame (the first press of a speed compiles it)
  r / c                random soup / clear
  g n p a d l w x v u  stamp a pattern at the mouse (h lists them); with shift: a fleet of 12
  t                    turn the next stamps by 90 degrees
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

SPEEDS = [1, 3, 9, 33, 101]
# inferno-ish: black, purple, red, orange, yellow; live cells pale yellow
STOPS = [(0.0, (0, 0, 4)), (0.15, (31, 12, 72)), (0.35, (85, 15, 109)), (0.55, (136, 34, 106)),
         (0.7, (186, 54, 85)), (0.82, (227, 89, 51)), (0.92, (249, 140, 10)), (1.0, (249, 201, 50))]
ALIVE = (252, 255, 164)


def lut():
    x = np.linspace(0, 1, 256)
    pos = [s[0] for s in STOPS]
    return np.stack([np.interp(x, pos, [s[1][i] for s in STOPS]) for i in range(3)], -1).astype(np.uint8)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", nargs=2, type=int, default=(1024, 768), metavar=("W", "H"))
    p.add_argument("--fill", type=float, default=0.3)
    p.add_argument("--fade", type=float, default=0.93, help="how slowly dead cells cool down (0..1)")
    p.add_argument("--fit", action="store_true", help="fullscreen, with the world sized to fill the screen")
    args = p.parse_args()

    dev = iron.get_current_device()
    pygame.init()
    dw, dh = pygame.display.get_desktop_sizes()[0]
    speed = 0
    if args.fit:
        vw, vh = fit_world(dw, dh, dev.cols * ROWS)
        chip = Chip(vw, vh, SPEEDS[speed], dev.cols)  # chip.h may be a few rows more: those stay off screen
    else:
        chip = Chip(*args.size, SPEEDS[speed], dev.cols)
        vw, vh = chip.w, chip.h
    rng = np.random.default_rng()
    cells = (rng.random((chip.h, chip.w)) < args.fill).astype(np.uint8)
    heat = np.zeros(cells.shape, np.float32)
    colors = lut()

    if args.fit:
        screen = pygame.display.set_mode((dw, dh), pygame.FULLSCREEN)
    else:
        scale = max(1, min(int(dw * 0.9) // vw, int(dh * 0.85) // vh))
        screen = pygame.display.set_mode((vw * scale, vh * scale), pygame.RESIZABLE)
    pygame.display.set_caption(f"Life on {chip.cores} NPU cores — {vw}x{vh}")

    def mouse_cell():
        r = place(screen, vw, vh)
        mx, my = pygame.mouse.get_pos()
        return (mx - r.x) * vw // max(1, r.w), (my - r.y) * vh // max(1, r.h)

    font = pygame.font.SysFont("monospace", 18, bold=True)
    clock = pygame.time.Clock()

    paused, step, gen, gens_done, t_rate, ms, rate = False, False, 0, 0, time.perf_counter(), 0.0, 0.0
    compiled = set()
    turn, message, message_until, show_help = 0, "h: help", time.perf_counter() + 5, False
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_s:
                    step = True
                elif e.key == pygame.K_r:
                    cells = (rng.random(cells.shape) < args.fill).astype(np.uint8)
                elif e.key == pygame.K_c:
                    cells[:] = 0
                    heat[:] = 0
                elif e.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_h:
                    show_help = not show_help
                elif e.key == pygame.K_t:
                    turn = (turn + 1) % 4
                    message, message_until = f"stamps turned {turn * 90} degrees", time.perf_counter() + 2
                elif e.unicode.lower() in SHAPES:
                    name, shape = SHAPES[e.unicode.lower()]
                    x, y = mouse_cell()
                    if e.mod & pygame.KMOD_SHIFT:  # a fleet: 4 x 3 spread over the world
                        for i in range(4):
                            for j in range(3):
                                stamp(cells, shape, x + i * vw // 4, y + j * vh // 3, turn)
                        name = f"12 x {name}"
                    else:
                        stamp(cells, shape, x, y, turn)
                    message, message_until = name, time.perf_counter() + 2
                elif pygame.K_1 <= e.key <= pygame.K_5:
                    speed = e.key - pygame.K_1
                    chip.gens = SPEEDS[speed]

        # draw with the mouse
        buttons = pygame.mouse.get_pressed()
        if buttons[0] or buttons[2]:
            x, y = mouse_cell()
            ys, xs = np.ogrid[-3:4, -3:4]
            disc = xs * xs + ys * ys <= 9
            yy, xx = (ys + y) % chip.h, (xs + x) % chip.w
            if buttons[0]:
                cells[yy, xx] |= disc.astype(np.uint8)
            else:
                cells[yy, xx] &= (~disc).astype(np.uint8)

        if not paused or step:
            if chip.gens not in compiled:
                hud(screen, font, f"compiling the kernel for {chip.gens} generations per call ...")
                pygame.display.flip()
                compiled.add(chip.gens)
            cells, ms = chip.run(cells)
            gen += chip.gens
            gens_done += chip.gens
            step = False

        now = time.perf_counter()
        if now - t_rate > 0.5:
            rate, gens_done, t_rate = gens_done / (now - t_rate), 0, now

        heat *= args.fade
        heat[cells == 1] = 1.0
        rgb = colors[(heat * 255).astype(np.uint8)]
        rgb[cells == 1] = ALIVE
        surf = pygame.surfarray.make_surface(rgb[:vh, :vw].swapaxes(0, 1))
        r = place(screen, vw, vh)
        screen.fill((0, 0, 0))
        screen.blit(pygame.transform.scale(surf, r.size), r.topleft)
        hud(screen, font,
            f"gen {gen:,}  |  {rate:,.0f} gen/s  |  {chip.gens} per frame  |  NPU {ms:.1f} ms/call  |  "
            f"{chip.w * chip.h * rate / 1e9:.2f} Gcell/s  |  {vw}x{vh}  |  alive {int(cells.sum()):,}"
            + ("  |  PAUSED" if paused else ""))
        if now < message_until:
            hud(screen, font, message, y=44)
        if show_help:
            for i, (k, (name, shape)) in enumerate(SHAPES.items()):
                hud(screen, font, f"{k}  {name}  ({shape.shape[1]}x{shape.shape[0]})", y=80 + i * 30)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def hud(screen, font, text, y=8):
    label = font.render(text, True, (255, 240, 200))
    box = pygame.Surface((label.get_width() + 20, label.get_height() + 10), pygame.SRCALPHA)
    box.fill((0, 0, 0, 170))
    screen.blit(box, (8, y))
    screen.blit(label, (18, y + 5))


if __name__ == "__main__":
    main()
