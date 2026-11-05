#!/bin/bash
# Run a command with XRT and IRON set up.
#   ./env.sh xrt-smi examine
#   ./env.sh bash                                 a shell to play in
#   cd iron && ../env.sh python life_window.py
#
# XRT comes from ./xrt (made by xrt.sh), or from XILINX_XRT if you set it (/opt/xilinx/xrt on Ubuntu).
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -z "${XILINX_XRT:-}" ]; then
    if [ -d "$HERE/xrt" ]; then XILINX_XRT=$HERE/xrt; else XILINX_XRT=/opt/xilinx/xrt; fi
fi
export XILINX_XRT
export PATH=$XILINX_XRT/bin:$PATH
LIBS=$XILINX_XRT/lib
[ -d "$HERE/boost/usr/lib/x86_64-linux-gnu" ] && LIBS=$LIBS:$HERE/boost/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$XILINX_XRT/python${PYTHONPATH:+:$PYTHONPATH}
source "$HERE/ironenv/bin/activate"
cd "$HERE/mlir-aie" && source utils/env_setup.sh >/dev/null && cd - >/dev/null
exec "$@"
