/* The interpreter for the host: the reference the NPU has to match, byte for byte. */
#include "rv32.h"
int rv_frame(uint8_t *ram, uint32_t budget, uint32_t *ran) { return rv_run(ram, budget, ran); }
