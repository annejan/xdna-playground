#!/bin/bash
# IRON (mlir-aie v1.4.5) in a venv: the wheels for mlir-aie and the Peano compiler, no system install.
#   ./setup.sh          one time, ~1.5 GB
#   ./env.sh CMD...     every time: run CMD with XRT + IRON in the environment
set -euo pipefail
cd "$(dirname "$0")"
MLIR_AIE=v1.4.5
[ -d mlir-aie ] || git clone --depth 1 -b "$MLIR_AIE" https://github.com/Xilinx/mlir-aie.git
python3.12 -m venv ironenv
ironenv/bin/pip install -q --upgrade pip
ironenv/bin/pip install "mlir_aie==${MLIR_AIE#v}" -f "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/$MLIR_AIE"
ironenv/bin/pip install -r mlir-aie/utils/peano-requirements.txt
ironenv/bin/pip install -r mlir-aie/python/requirements.txt
ironenv/bin/pip install pygame-ce
du -sh ironenv
