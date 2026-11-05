#include "tm.h"
/* a starfield flying towards you */
#define N 200
static int sx[N], sy[N], sz[N];
static uint32_t rng = 7;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
int main(void) {
    rng += machine_id() * 7919;
    for (int i = 0; i < N; i++) { sx[i] = (int)(rnd() % 2000) - 1000; sy[i] = (int)(rnd() % 1200) - 600; sz[i] = 1 + rnd() % 1000; }
    int speed = 4 + machine_id() % 12;
    for (;;) {
        for (int i = 0; i < (int)(W * H / 2); i++) FB[i] = 0;
        for (int i = 0; i < N; i++) {
            sz[i] -= speed;
            if (sz[i] <= 1) { sx[i] = (int)(rnd() % 2000) - 1000; sy[i] = (int)(rnd() % 1200) - 600; sz[i] = 1000; }
            int x = W / 2 + sx[i] * 64 / sz[i], y = H / 2 + sy[i] * 64 / sz[i];
            if (x >= 0 && x < W && y >= 0 && y < H) pset(x, y, 15 - sz[i] * 14 / 1000);
        }
        vsync();
    }
}
