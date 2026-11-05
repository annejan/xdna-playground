"""Cellular automata for the on-chip engine (life_chip.py): one byte per cell, 8 neighbours.

Each rule has three versions of the same thing:
  - C macros for the scalar code (the two columns that wrap around),
  - an aie_api vector body (64 cells per step) for everything else,
  - a numpy version, to check the NPU against.

The vector body gets the 9 loads of the 3x3 neighbourhood as ul uc ur / ml mc mr / dl dc dr
(uint8 x 64) and the constants zero one two three, and leaves the new cells in `res`.
"""

import numpy as np


def neighbours(c, on):
    """Per cell: how many of the 8 neighbours are 'on' (torus)."""
    x = on(c).astype(np.uint8)
    return sum(np.roll(np.roll(x, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def life_np(c):
    s = neighbours(c, lambda v: v)
    return ((s == 3) | ((s == 2) & (c == 1))).astype(np.uint8)


def brain_np(c):
    s = neighbours(c, lambda v: v == 1)
    out = np.where(c == 1, 2, 0).astype(np.uint8)
    out[(c == 0) & (s == 2)] = 1
    return out


def wire_np(c):
    s = neighbours(c, lambda v: v == 1)
    out = c.copy()
    out[c == 1] = 2
    out[c == 2] = 3
    out[(c == 3) & ((s == 1) | (s == 2))] = 1
    return out


def avg(a, b):
    """floor((a + b) / 2) without leaving 8 bits, the same way the kernel does it."""
    return (a >> 1) + (b >> 1) + (a & b & 1)


FIRE_COOL = 8  # every 8th column cools down by 1 per row


def fire_np(c):
    dn = np.roll(c, -1, 0)
    h = avg(dn, avg(np.roll(dn, 1, 1), np.roll(dn, -1, 1)))
    cool = (np.arange(c.shape[1]) % FIRE_COOL == 0).astype(np.uint8)
    return (np.maximum(h, cool) - cool).astype(np.uint8)


RULES = {
    # Conway's Life: born with 3 neighbours, survives with 2 or 3
    "life": {
        "states": 2,
        "on": "(v)",
        "next": "((s) == 3 || ((s) == 2 && (m)))",
        "vector": """
            auto s = aie::add(ul, uc); s = aie::add(s, ur); s = aie::add(s, ml); s = aie::add(s, mr);
            s = aie::add(s, dl); s = aie::add(s, dc); s = aie::add(s, dr);
            auto res = aie::select(zero, one, aie::eq(aie::bit_or(s, mc), three));""",
        "numpy": life_np,
    },
    # Brian's Brain: 0 off, 1 firing, 2 dying. Off fires with exactly 2 firing neighbours,
    # firing always starts dying, dying always goes off. Nothing is stable: it keeps exploding.
    "brain": {
        "states": 3,
        "on": "((v) == 1)",
        "next": "((m) == 1 ? 2 : ((m) == 0 && (s) == 2) ? 1 : 0)",
        "vector": """
            /* firing is 1 and dying is 2, so (v & 1) is 1 exactly for the firing cells */
            auto s = aie::add(aie::bit_and(ul, one), aie::bit_and(uc, one));
            s = aie::add(s, aie::bit_and(ur, one)); s = aie::add(s, aie::bit_and(ml, one));
            s = aie::add(s, aie::bit_and(mr, one)); s = aie::add(s, aie::bit_and(dl, one));
            s = aie::add(s, aie::bit_and(dc, one)); s = aie::add(s, aie::bit_and(dr, one));
            auto born = aie::eq(s, two) & aie::eq(mc, zero);
            auto res = aie::select(zero, one, born);
            res = aie::select(res, two, aie::eq(mc, one));""",
        "numpy": brain_np,
    },
    # Wireworld: 0 empty, 1 electron head, 2 electron tail, 3 wire. Head -> tail -> wire;
    # wire becomes a head with 1 or 2 heads around it. Enough to build diodes, clocks and logic.
    "wire": {
        "states": 4,
        "on": "((v) == 1)",
        "next": "((m) == 1 ? 2 : (m) == 2 ? 3 : (m) == 3 ? (((s) == 1 || (s) == 2) ? 1 : 3) : 0)",
        "vector": """
            auto h = [&](auto v) { return aie::select(zero, one, aie::eq(v, one)); };
            auto s = aie::add(h(ul), h(uc)); s = aie::add(s, h(ur)); s = aie::add(s, h(ml));
            s = aie::add(s, h(mr)); s = aie::add(s, h(dl)); s = aie::add(s, h(dc)); s = aie::add(s, h(dr));
            auto fire = aie::eq(mc, three) & (aie::eq(s, one) | aie::eq(s, two));
            auto res = aie::select(mc, two, aie::eq(mc, one));
            res = aie::select(res, three, aie::eq(mc, two));
            res = aie::select(res, one, fire);""",
        "numpy": wire_np,
    },
}


# Demo fire: a byte of heat per cell, each generation the heat of the 3 cells below moves up one
# row (half from straight below, a quarter from each side), and every 8th column cools down by 1.
# The host keeps putting fuel on the bottom rows.
RULES["fire"] = {
    "states": 256,
    "prelude": """
    aie::vector<uint8_t, 64> cool;
    for (int i = 0; i < 64; i++) cool[i] = (i % 8) == 0;""",
    "vector": """
            auto avg = [&](auto a, auto b) {
                return aie::add(aie::add(aie::downshift(a, 1), aie::downshift(b, 1)), aie::bit_and(aie::bit_and(a, b), one));
            };
            auto heat = avg(dc, avg(dl, dr));
            auto res = aie::sub(aie::max(heat, cool), cool);
            (void)ul; (void)uc; (void)ur; (void)ml; (void)mc; (void)mr;""",
    "cell": ("int a = (dn[l] >> 1) + (dn[rt] >> 1) + (dn[l] & dn[rt] & 1); "
             "int h = (dn[cc] >> 1) + (a >> 1) + (dn[cc] & a & 1); "
             "int k = (cc % 8) == 0; o[cc] = h > k ? h - k : 0;"),
    "numpy": fire_np,
}


def cpu(cells, gens, rule="life"):
    f = RULES[rule]["numpy"]
    for _ in range(gens):
        cells = f(cells)
    return cells


def maze(h, w, rng, spacing=3, roots=None, cover=0.9):
    """Wireworld start: a forest of random trees made of wire, grown from a few roots at once.

    Nodes sit every `spacing` cells and branches run between them, so wires never touch except
    where they join. A tree has no loops: a pulse sent in at a root splits at every fork, runs out to
    the dead ends and is gone. Feed the roots now and then (`sources`) and you get lightning.
    Returns the cells and the sources: (y, x, period in generations) per root."""
    ny, nx = (h - 2) // spacing, (w - 2) // spacing
    roots = roots or max(4, ny * nx // 1600)
    owner = np.full((ny, nx), -1)
    c = np.zeros((h, w), np.uint8)
    frontier, sources = [], []
    for k in range(roots):
        y, x = int(rng.integers(0, ny)), int(rng.integers(0, nx))
        if owner[y, x] >= 0:
            continue
        owner[y, x] = k
        c[1 + y * spacing, 1 + x * spacing] = 3
        sources.append((1 + y * spacing, 1 + x * spacing, int(rng.integers(6, 30))))
        frontier.append((y, x))
    target, grown = cover * ny * nx, len(frontier)
    while frontier and grown < target:
        i = int(rng.integers(0, len(frontier)))
        y, x = frontier[i]
        free = [(y + dy, x + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= y + dy < ny and 0 <= x + dx < nx and owner[y + dy, x + dx] < 0]
        if not free:
            frontier[i] = frontier[-1]
            frontier.pop()
            continue
        ty, tx = free[int(rng.integers(0, len(free)))]
        owner[ty, tx] = owner[y, x]
        y0, x0, y1, x1 = 1 + y * spacing, 1 + x * spacing, 1 + ty * spacing, 1 + tx * spacing
        c[min(y0, y1):max(y0, y1) + 1, min(x0, x1):max(x0, x1) + 1] = 3
        frontier.append((ty, tx))
        grown += 1
    return c, sources


def feed(cells, sources, gen, gens):
    """Send an electron in at every source whose period came round in the last `gens` generations."""
    for y, x, period in sources:
        if gen // period != (gen - gens) // period and cells[y, x] == 3:
            cells[y, x] = 1


def seed(rule, shape, rng, fill=0.3):
    if rule == "fire":
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if rule == "wire":
        cells, sources = maze(*shape, rng)
        cells[[y for y, _, _ in sources], [x for _, x, _ in sources]] = 1  # one pulse per root to start
        return cells
    return (rng.random(shape) < fill).astype(np.uint8)
