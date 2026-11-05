/* The mini machine's "SDK": a 160 x 96 screen of 16 colours, a frame call, the machine's id. */
#include <stdint.h>
#define W 160
#define H 96
#define FB ((volatile uint8_t *)0x5000)
#define MAIL ((volatile uint32_t *)0x7E00)
static inline uint32_t frame_no(void) { return MAIL[0]; }
static inline uint32_t machine_id(void) { return MAIL[1]; }
static inline void pset(int x, int y, int c) {
    volatile uint8_t *p = FB + (y * W + x) / 2;
    *p = (x & 1) ? (uint8_t)((*p & 0x0f) | (c << 4)) : (uint8_t)((*p & 0xf0) | (c & 15));
}
static inline int pget(int x, int y) { uint8_t v = FB[(y * W + x) / 2]; return (x & 1) ? v >> 4 : v & 15; }
/* end the frame: the host shows the screen and lets the program run on next frame */
static inline void vsync(void) { register int a7 asm("a7") = 1; asm volatile("ecall" : : "r"(a7) : "memory"); }
/* sine in 1/256 steps of a turn, result -127..127, from a table made at start */
static int8_t sintab[256];
static void make_sin(void) {
    /* a parabola per half turn is close enough for demo effects */
    for (int i = 0; i < 256; i++) {
        int t = i & 127, v = t * (128 - t) * 127 / 4096;
        sintab[i] = (int8_t)(i < 128 ? v : -v);
    }
}
static inline int isin(int a) { return sintab[a & 255]; }
static inline int icos(int a) { return sintab[(a + 64) & 255]; }
