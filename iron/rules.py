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


def cpu(cells, gens, rule="life"):
    f = RULES[rule]["numpy"]
    for _ in range(gens):
        cells = f(cells)
    return cells


def circuit(h, w, rng):
    """Wireworld start: rows of clocks. Each clock is a loop of wire with electrons going round; its
    size sets the rhythm. Each sends its pulses down an output wire that ends in nothing, some with
    a fork. Nothing leads back into a loop, so the circuit never fills up with electrons."""
    c = np.zeros((h, w), np.uint8)
    for r in range(2, h - 10, 10):
        x0 = 2 + int(rng.integers(0, 12))
        lw, lh = int(rng.integers(5, 26)), 4
        c[r, x0:x0 + lw + 1] = c[r + lh, x0:x0 + lw + 1] = 3
        c[r:r + lh + 1, x0] = c[r:r + lh + 1, x0 + lw] = 3
        for _ in range(int(rng.integers(1, 3))):  # electrons on the bottom edge, moving right
            x = x0 + 2 + int(rng.integers(0, max(1, lw - 3)))
            c[r + lh, x], c[r + lh, x - 1] = 1, 2
        y = r + lh // 2  # output wire from the right side of the loop
        end = min(w - 3, x0 + lw + 1 + int(rng.integers(w // 3, w)))
        c[y, x0 + lw + 1:end] = 3
        if rng.random() < 0.6:  # a fork 3 rows down, into the gap before the next row of clocks
            fx = int(rng.integers(x0 + lw + 6, max(x0 + lw + 7, end - 10)))
            c[y + 1:y + 4, fx] = 3
            c[y + 3, fx:min(w - 3, fx + int(rng.integers(20, w // 2)))] = 3
    return c


def seed(rule, shape, rng, fill=0.3):
    if rule == "wire":
        return circuit(*shape, rng)
    return (rng.random(shape) < fill).astype(np.uint8)
