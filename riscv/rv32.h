/* A small RV32IM interpreter: one RISC-V machine with 32 KB of RAM.
 *
 * The same code runs on an AIE core (compiled by Peano) and on the host (gcc), so the host can check
 * the NPU byte for byte. No libc, no globals: the whole machine lives in its RAM.
 *
 * Memory map (32 KB, addresses wrap at 32 KB):
 *   0x0000  program (code, data, bss), loaded by the host
 *   0x5000  stack top (grows down)
 *   0x5000  framebuffer: 160 x 96 pixels, 4 bits each, two per byte (7680 bytes)
 *   0x7E00  mailbox: frame number, machine id, buttons (written by the host)
 *   0x7F00  the CPU: x0..x31, pc (kept here between calls)
 *
 * A program ends every frame with `ecall` (a7 = 1): the interpreter stops and the host shows the
 * framebuffer. a7 = 93 (exit) stops the machine for good.
 */
#ifndef RV32_H
#define RV32_H
#include <stdint.h>

#define RV_RAM 0x8000u
#define RV_MASK (RV_RAM - 1u)
#define RV_FB 0x5000u
#define RV_FB_BYTES 7680u
#define RV_MAIL 0x7E00u
#define RV_CPU 0x7F00u   /* 32 registers + pc: 132 bytes, up to 0x7F84 */

enum { RV_FRAME = 1, RV_BUDGET = 2, RV_HALT = 3, RV_BAD = 4 };

/* a 32-bit word that may alias the byte RAM: aligned accesses become one load or store */
typedef uint32_t __attribute__((may_alias)) rv_word;

static inline uint32_t rv_ld32(const uint8_t *m, uint32_t a) {
    a &= RV_MASK;
    if ((a & 3) == 0)
        return *(const rv_word *)(m + a);
    return (uint32_t)m[a] | (uint32_t)m[(a + 1) & RV_MASK] << 8 | (uint32_t)m[(a + 2) & RV_MASK] << 16 |
           (uint32_t)m[(a + 3) & RV_MASK] << 24;
}

static inline void rv_st32(uint8_t *m, uint32_t a, uint32_t v) {
    a &= RV_MASK;
    if ((a & 3) == 0) {
        *(rv_word *)(m + a) = v;
        return;
    }
    m[a] = (uint8_t)v;
    m[(a + 1) & RV_MASK] = (uint8_t)(v >> 8);
    m[(a + 2) & RV_MASK] = (uint8_t)(v >> 16);
    m[(a + 3) & RV_MASK] = (uint8_t)(v >> 24);
}

/* Run until the program ends a frame, halts, hits a bad instruction, or `budget` instructions pass.
 * Returns the reason; the count of instructions run goes to *ran. */
static int rv_run(uint8_t *m, uint32_t budget, uint32_t *ran) {
    uint32_t x[32];
    for (int i = 0; i < 32; i++) x[i] = rv_ld32(m, RV_CPU + 4 * i);
    uint32_t pc = rv_ld32(m, RV_CPU + 128);
    int why = RV_BUDGET;
    uint32_t n = 0;
    while (n < budget) {
        uint32_t in = rv_ld32(m, pc);
        uint32_t op = in & 0x7f, rd = (in >> 7) & 31, f3 = (in >> 12) & 7, rs1 = (in >> 15) & 31,
                 rs2 = (in >> 20) & 31, f7 = in >> 25;
        uint32_t a = x[rs1], b = x[rs2], next = pc + 4, v = 0;
        int32_t imm_i = (int32_t)in >> 20;
        int32_t imm_s = ((int32_t)in >> 25 << 5) | (int32_t)((in >> 7) & 31);
        int wr = 1;
        n++;
        switch (op) {
        case 0x37: v = in & 0xfffff000u; break;                    /* lui */
        case 0x17: v = pc + (in & 0xfffff000u); break;             /* auipc */
        case 0x6f: {                                               /* jal */
            int32_t imm = ((int32_t)(in & 0x80000000u) >> 11) | (int32_t)(in & 0xff000u) |
                          (int32_t)((in >> 9) & 0x800u) | (int32_t)((in >> 20) & 0x7feu);
            v = next;
            next = pc + (uint32_t)imm;
            break;
        }
        case 0x67: v = next; next = (a + (uint32_t)imm_i) & ~1u; break; /* jalr */
        case 0x63: {                                               /* branches */
            int32_t imm = ((int32_t)(in & 0x80000000u) >> 19) | (int32_t)((in << 4) & 0x800u) |
                          (int32_t)((in >> 20) & 0x7e0u) | (int32_t)((in >> 7) & 0x1eu);
            int t = 0;
            switch (f3) {
            case 0: t = a == b; break;
            case 1: t = a != b; break;
            case 4: t = (int32_t)a < (int32_t)b; break;
            case 5: t = (int32_t)a >= (int32_t)b; break;
            case 6: t = a < b; break;
            case 7: t = a >= b; break;
            default: why = RV_BAD; goto out;
            }
            if (t) next = pc + (uint32_t)imm;
            wr = 0;
            break;
        }
        case 0x03: {                                               /* loads */
            uint32_t ad = (a + (uint32_t)imm_i) & RV_MASK;
            switch (f3) {
            case 0: v = (uint32_t)(int32_t)(int8_t)m[ad]; break;
            case 1: v = (uint32_t)(int32_t)(int16_t)(m[ad] | m[(ad + 1) & RV_MASK] << 8); break;
            case 2: v = rv_ld32(m, ad); break;
            case 4: v = m[ad]; break;
            case 5: v = m[ad] | (uint32_t)m[(ad + 1) & RV_MASK] << 8; break;
            default: why = RV_BAD; goto out;
            }
            break;
        }
        case 0x23: {                                               /* stores */
            uint32_t ad = (a + (uint32_t)imm_s) & RV_MASK;
            switch (f3) {
            case 0: m[ad] = (uint8_t)b; break;
            case 1: m[ad] = (uint8_t)b; m[(ad + 1) & RV_MASK] = (uint8_t)(b >> 8); break;
            case 2: rv_st32(m, ad, b); break;
            default: why = RV_BAD; goto out;
            }
            wr = 0;
            break;
        }
        case 0x13: {                                               /* op-imm */
            uint32_t sh = rs2;
            switch (f3) {
            case 0: v = a + (uint32_t)imm_i; break;
            case 1: v = a << sh; break;
            case 2: v = (int32_t)a < imm_i; break;
            case 3: v = a < (uint32_t)imm_i; break;
            case 4: v = a ^ (uint32_t)imm_i; break;
            case 5: v = (in & 0x40000000u) ? (uint32_t)((int32_t)a >> sh) : a >> sh; break;
            case 6: v = a | (uint32_t)imm_i; break;
            case 7: v = a & (uint32_t)imm_i; break;
            }
            break;
        }
        case 0x33:                                                 /* op */
            if (f7 == 1) {                                         /* M extension */
                switch (f3) {
                case 0: v = a * b; break;
                case 1: v = (uint32_t)(((int64_t)(int32_t)a * (int64_t)(int32_t)b) >> 32); break;
                case 2: v = (uint32_t)(((int64_t)(int32_t)a * (int64_t)(uint64_t)b) >> 32); break;
                case 3: v = (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32); break;
                case 4: v = b == 0 ? 0xffffffffu : ((int32_t)a == INT32_MIN && (int32_t)b == -1) ? a
                                                 : (uint32_t)((int32_t)a / (int32_t)b); break;
                case 5: v = b == 0 ? 0xffffffffu : a / b; break;
                case 6: v = b == 0 ? a : ((int32_t)a == INT32_MIN && (int32_t)b == -1) ? 0
                                       : (uint32_t)((int32_t)a % (int32_t)b); break;
                case 7: v = b == 0 ? a : a % b; break;
                }
            } else {
                switch (f3) {
                case 0: v = (in & 0x40000000u) ? a - b : a + b; break;
                case 1: v = a << (b & 31); break;
                case 2: v = (int32_t)a < (int32_t)b; break;
                case 3: v = a < b; break;
                case 4: v = a ^ b; break;
                case 5: v = (in & 0x40000000u) ? (uint32_t)((int32_t)a >> (b & 31)) : a >> (b & 31); break;
                case 6: v = a | b; break;
                case 7: v = a & b; break;
                }
            }
            break;
        case 0x0f: wr = 0; break;                                  /* fence */
        case 0x73:                                                 /* ecall */
            wr = 0;
            pc = next;
            why = x[17] == 93 ? RV_HALT : RV_FRAME;
            goto out;
        default:
            why = RV_BAD;
            goto out;
        }
        if (wr && rd) x[rd] = v;
        pc = next;
    }
out:
    for (int i = 0; i < 32; i++) rv_st32(m, RV_CPU + 4 * i, x[i]);
    rv_st32(m, RV_CPU + 128, pc);
    *ran = n;
    return why;
}

#endif
