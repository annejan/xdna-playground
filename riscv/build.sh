#!/bin/bash
# Build the demo programs for the mini RISC-V machines: bare metal RV32IM, no libc.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
for src in demos/*.c; do
    name=$(basename "$src" .c)
    clang --target=riscv32 -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -nostdlib -Idemos \
        -T link.ld -fuse-ld=lld -o "bin/$name.elf" crt0.S "$src"
    llvm-objcopy -O binary "bin/$name.elf" "bin/$name.bin"
    echo "$name: $(stat -c %s "bin/$name.bin") bytes"
done
