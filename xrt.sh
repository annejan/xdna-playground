#!/bin/bash
# XRT 2.26 for the AMD XDNA NPU, unpacked into ./xrt -- no root, nothing in /opt, any distro.
#
# AMD ships XRT for Linux only as Ubuntu 24.04 .deb files. They are plain binaries, so on another
# distro (this was made on openSUSE Tumbleweed) we unpack them instead of installing them, and add
# the two Ubuntu boost 1.83 libraries that xrt-smi links to. The kernel driver is not in here:
# amdxdna is in mainline Linux since 6.14, and the NPU firmware comes with linux-firmware.
#
#   ./xrt.sh              download, check, unpack into ./xrt and ./boost
#   ./env.sh xrt-smi examine
set -euo pipefail
cd "$(dirname "$0")"

AMD=https://download.amd.com/opendownload/RyzenAI/1.9.0/RAI_1.9_Linux_NPU_XRT.zip
UBUNTU=http://archive.ubuntu.com/ubuntu/pool/main/b/boost1.83
BOOST=(libboost-filesystem1.83.0_1.83.0-2.1ubuntu3_amd64.deb libboost-program-options1.83.0_1.83.0-2.1ubuntu3_amd64.deb)

mkdir -p dl
[ -f dl/RAI_1.9_Linux_NPU_XRT.zip ] || curl -fL -o dl/RAI_1.9_Linux_NPU_XRT.zip "$AMD"
for d in "${BOOST[@]}"; do
    [ -f "dl/$d" ] || curl -fL -o "dl/$d" "$UBUNTU/$d"
done
(cd dl && sha256sum -c - <<'EOF'
074c981fad839ea4b7623337f63ed772e3469f5212d16c220b0f6f8b002b7877  RAI_1.9_Linux_NPU_XRT.zip
d36329139070037ce08aad7f12ecbf6ad14ae764b5dd824915fdc7e319205369  libboost-filesystem1.83.0_1.83.0-2.1ubuntu3_amd64.deb
a72f5c27dc06d7cc5fffbbd818d71dec4316871adb1815394d716fd17c150088  libboost-program-options1.83.0_1.83.0-2.1ubuntu3_amd64.deb
EOF
)
(cd dl && unzip -oq RAI_1.9_Linux_NPU_XRT.zip)

# a .deb is an ar archive with data.tar.* inside: unpack each into its own directory
unpack() {
    local deb tmp
    deb=$(realpath "$1")
    tmp=$(mktemp -d -p dl)
    (cd "$tmp" && ar x "$deb")
    mkdir -p "$2"
    tar -xf "$tmp"/data.tar.* -C "$2"
    rm -rf "$tmp"
}

rm -rf x xrt boost
for deb in dl/xrt_*.deb; do
    unpack "$deb" "x/$(basename "$deb" .deb)"
done
# base + base-dev + npu + the amdxdna plugin all live in opt/xilinx/xrt: merge them (hard links)
mkdir -p xrt
for d in x/*/opt/xilinx/xrt; do
    cp -al "$d"/. xrt/
done
for d in "${BOOST[@]}"; do
    unpack "dl/$d" boost
done
echo "XRT $(sed -n 's/.*"BUILD_VERSION" *: *"\([^"]*\)".*/\1/p' xrt/version.json 2>/dev/null) in ./xrt"
echo "try: ./env.sh xrt-smi examine"
