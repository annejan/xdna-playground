# xdna-playground

Fun with the AMD XDNA 2 NPU in a Ryzen AI laptop, on Linux, programmed directly with
[IRON](https://github.com/Xilinx/mlir-aie) (AMD's Python toolkit for the AI Engine array).
No Ubuntu, no root, no Ryzen AI SDK: XRT is unpacked from AMD's .deb files and IRON comes from pip.

Made on an AMD Ryzen AI 7 PRO 350 (Krackan Point: 8 columns x 4 rows = 32 AI Engine cores) with
openSUSE Tumbleweed and the `amdxdna` driver in the mainline kernel.

## What is in here

| program | what it does |
|---|---|
| `iron/life_window.py` | Life, Brian's Brain, Wireworld and the old demo fire on all 32 cores, in a window: trails, draw with the mouse, stamp patterns, a demo mode, up to 101 generations per frame |
| `iron/life_chip.py` | the engine: the world stays on the chip, the cores swap their edge rows with their neighbours every generation |
| `iron/rules.py` | the automata: each as a 64-cells-per-step vector body, scalar C for the edges, and numpy to check against |
| `iron/life.py` | the first version: 8 cores, the host rebuilds the halos between calls; scalar and vector kernels |
| `iron/mandel.py` | Mandelbrot zoom in the terminal, fixed point on 8 cores, raced against numpy |
| `iron/patterns.py` | Gosper and Simkin glider guns, R-pentomino, acorn, diehard, spaceships, oscillators (RLE) |

## How fast

Game of Life, 1 bit per cell in a byte, measured on the laptop against numpy on its Zen 5 cores:

| version | Mcell/s |
|---|---:|
| numpy (CPU) | 820 - 1,180 |
| `life.py --scalar`: 8 cores, plain C | 141 |
| `life.py`: 8 cores, 64 cells per vector instruction, halos from the host | 2,079 |
| `life_chip.py --gens 101`: 32 cores, world on the chip | 16,017 |
| `life_chip.py --gens 1001`: 32 cores, world on the chip | **29,890** |

The last line is a 512 x 1536 world at 38,000 generations per second, 36 times numpy. Every
benchmark compares its result with numpy, cell for cell.

The same engine with other rules (`life_chip.py --bench --gens 101 --rule ...`):

| rule | NPU | numpy |
|---|---:|---:|
| Brian's Brain: off, firing, dying; fires with exactly 2 firing neighbours | 29,361 Mcell/s | 518 |
| Wireworld: empty, electron head, tail, wire; a wire fires with 1 or 2 heads around | 27,858 Mcell/s | 959 |
| Demo fire: a byte of heat, the 3 cells below move up a row, every 8th column cools | 30,654 Mcell/s | 1,079 |

## How the on-chip Life works

```
 column:   0     1     2         7
 row 5    k=3   k=4   k=11  ..  k=28      the 32 cores form one chain (a snake):
 row 4    k=2   k=5   k=10  ..  k=29      down column 0, up column 1, ...
 row 3    k=1   k=6   k=9   ..  k=30
 row 2    k=0   k=7   k=8   ..  k=31  ->  back to k=0 (the world is a torus)
```

- Core k holds band k of the world (for example 48 rows of 512 cells) in its own 64 KB memory.
- Every generation, core k sends its top row to core k-1 and its bottom row to core k+1, and waits
  for theirs. With those two rows it can compute its whole band.
- On AIE2 a core can read the memory of its north, south and west neighbours, so neighbours in the
  chain swap rows through shared memory. Only the wrap from core 31 to core 0 uses a DMA stream.
- The kernel does 64 cells per step: 8 shifted loads, 7 adds, and the rule without branches:
  `(neighbours | self) == 3` is true for 3 neighbours, or for 2 neighbours and a live cell.
- The host sends the world once, the cores run N generations on their own, the host reads it back.

## Set up

You need: an XDNA 2 NPU (Ryzen AI 300 series: Strix Point, Krackan; Ryzen AI Max: Strix Halo), Linux
6.14 or newer (the `amdxdna` driver), the NPU firmware from `linux-firmware`, Python 3.12, `curl`,
`unzip`, `ar` (binutils).

An XDNA 1 NPU (Ryzen 7040 / 8040: Phoenix, Hawk Point) should work too, on 16 cores instead of 32:
IRON supports it and the code takes the number of columns from the device. Not tested yet; if you
try it, please open an issue with what happened.

```
./xrt.sh        # XRT 2.26 from AMD's Ubuntu .deb files, unpacked into ./xrt (about 100 MB)
./setup.sh      # mlir-aie v1.4.5 + IRON + the Peano compiler in ./ironenv (about 1.5 GB)
./env.sh xrt-smi examine
```

`xrt-smi examine` must show your NPU. On Ubuntu with AMD's XRT installed you can skip `xrt.sh`;
`env.sh` then uses `/opt/xilinx/xrt`.

## Run

```
cd iron
../env.sh python life_window.py                     # 1024 x 768 world in a window
../env.sh python life_window.py --fit               # fullscreen, world sized to the screen (960 x 600 at 2x on 1920 x 1200)
../env.sh python life_window.py --fit --demo        # show mode: a new scene every 25 seconds
../env.sh python life_window.py --rule fire         # start with the fire (or brain, wire)
../env.sh python life_chip.py --bench --gens 1001   # the race against numpy
../env.sh python life.py --bench 512 2048 --gens 7  # the 8-core version
../env.sh python mandel.py                          # Mandelbrot zoom in the terminal
```

The first run of each kernel compiles it (some seconds); after that it comes from the cache.

Keys in `life_window.py`:

| key | |
|---|---|
| `tab` | next rule: Life, Brian's Brain, Wireworld, fire |
| mouse left / right | draw / erase (Wireworld: wire, fire: paint fire) |
| `e` | Wireworld: an electron at the mouse |
| `m` | demo mode on / off |
| `1` .. `5` | 1, 3, 9, 33 or 101 generations per frame |
| `g n p a d l w x v u` | stamp a pattern at the mouse; with shift a fleet of 12; `h` lists them |
| `t` | turn the next stamps by 90 degrees |
| `space`, `s` | pause, one step |
| `r`, `c` | new start, clear |
| `f`, `q` | fullscreen, quit |

## Licence

MIT, see [LICENSE](LICENSE). IRON, mlir-aie and the `aie_api` headers the kernels compile against
are AMD's (Apache 2.0 with LLVM exception); XRT is AMD's. `xrt.sh` downloads them from AMD and
Ubuntu, nothing of theirs is in this repository.
