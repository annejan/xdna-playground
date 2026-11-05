#include "tm.h"
/* a Mandelbrot dive in Q24 fixed point; every machine dives into its own spot */
static const int spots[8][2] = {{-12477490, 2211683}, {-12761088, -1421504}, {-29670000, -29170}, {-13013480, 2289480},
                                {-12997060, -2305440}, {4781800, 2516582}, {-1680000, 17340000}, {-26080000, 0}};
int main(void) {
    int id = machine_id();
    int cx = spots[id & 7][0], cy = spots[id & 7][1];
    int scale = 1 << 17;  /* Q24 per pixel: the view is 160 * scale wide */
    for (;;) {
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int ca = cx + (x - W / 2) * scale, cb = cy + (y - H / 2) * scale, a = 0, b = 0, n = 0;
                for (; n < 64; n++) {
                    int64_t aa = (int64_t)a * a >> 24, bb = (int64_t)b * b >> 24;
                    if (aa + bb > (4 << 24)) break;
                    b = (int)((int64_t)a * b >> 23) + cb;
                    a = (int)(aa - bb) + ca;
                }
                pset(x, y, n == 64 ? 0 : 1 + n % 15);
            }
        scale = scale * 15 / 16;
        if (scale < 64) scale = 1 << 17;
        vsync();
    }
}
