#!/bin/bash
# Compile an RDNA probe on the pod. usage: build.sh <file.hip> [arch] [--asm]
set -e
src="${1:-empty.hip}"
arch="${2:-gfx1100}"
mode="${3:-}"
case "$arch" in
    gfx1100) def=-DKITTENS_RDNA3 ;;
    *) echo "unknown arch $arch" >&2; exit 1 ;;
esac
cd "$(dirname "$0")"
FLAGS="-std=c++20 -O3 --offload-arch=$arch $def
    -I${ROCM_PATH:-/opt/rocm}/include/hip
    -I../../include -Wno-pass-failed"
if [ "$mode" = "--asm" ]; then
    hipcc $FLAGS --cuda-device-only -S "$src" -o "${src%.hip}.s"
    echo "built ${src%.hip}.s"
else
    hipcc $FLAGS "$src" -o "${src%.hip}"
    echo "built ${src%.hip}"
fi
