#include "tm.h"
/* Life on the whole screen, a cell per pixel; the colour is the cell's age. The ages live in the
   framebuffer itself (4 bits per cell); the next generation is built in a packed buffer of the same size. */
static uint8_t next[W * H / 2];
static uint32_t rng = 1;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static inline int alive(int x, int y) { return pget((x + W) % W, (y + H) % H) != 0; }
int main(void) {
    rng = 0x9e3779b9u * (machine_id() + 1);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) pset(x, y, (rnd() & 3) == 0 ? 15 : 0);
    for (int f = 0;; f++) {
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int s = alive(x - 1, y - 1) + alive(x, y - 1) + alive(x + 1, y - 1) + alive(x - 1, y) +
                        alive(x + 1, y) + alive(x - 1, y + 1) + alive(x, y + 1) + alive(x + 1, y + 1);
                int c = pget(x, y), n = (s == 3 || (s == 2 && c)) ? (c > 2 ? c - 1 : 2) : 0;  /* 15 = newborn, fades to 2 */
                uint8_t *p = &next[(y * W + x) / 2];
                *p = (x & 1) ? (uint8_t)((*p & 0x0f) | (n << 4)) : (uint8_t)((*p & 0xf0) | n);
            }
        for (int i = 0; i < W * H / 2; i++) FB[i] = next[i];
        if ((f & 127) == 127) for (int i = 0; i < 300; i++) pset(rnd() % W, rnd() % H, 15);  /* a little rain */
        vsync();
    }
}
