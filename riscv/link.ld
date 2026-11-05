/* A mini machine: program from 0, stack down from 0x5000, framebuffer at 0x5000. */
ENTRY(_start)
SECTIONS {
    . = 0;
    .text : { *(.text.start) *(.text*) }
    .rodata : { *(.rodata*) *(.srodata*) }
    .data : { *(.data*) *(.sdata*) }
    __bss_start = .;
    .bss : { *(.bss*) *(.sbss*) *(COMMON) }
    . = ALIGN(4);
    __bss_end = .;
    ASSERT(. < 0x4000, "program too big: it must leave room for the stack below 0x5000")
}
