#include "tm.h"
int main(void) {
    make_sin();
    int id = machine_id(), k = 2 + id % 5;
    for (;;) {
        int t = frame_no() * (2 + id % 3);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int v = isin(x * k + t) + isin(y * (k + 1) - t) + isin((x + y) * 2 + t * 2) + isin((x * x + y * y) / (16 + id));
                pset(x, y, (v >> 5) & 15);
            }
        vsync();
    }
}
