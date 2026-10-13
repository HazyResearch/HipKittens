#!/bin/bash
# The MLIR the hk dialect is compiled against.
#
# MLIR only, the host target only -- hk lowers to HipKittens C++ and hipcc does
# the GPU code generation, so nothing here needs AMDGPU -- Release with
# assertions, and the Python bindings for $PYTHON. Bindings are per CPython
# version, so a machine that runs two interpreters builds twice.
#
#   scripts/build_mlir.sh
#   HK_LLVM_ROOT=/workspace/llvm PYTHON=/opt/venv/bin/python3 scripts/build_mlir.sh
#
# With no network (the dev pods), put a checkout at $HK_LLVM_ROOT/llvm-project
# first; the clone below is skipped when one is there.
#
# JOBS defaults to nproc, which inside a container is the node's core count
# and not the cgroup's: on the 8-CPU, 48 GiB dev pod, nproc compile jobs got
# the whole pod OOM-killed. Pass JOBS=6 LINK_JOBS=1 there.
set -euo pipefail

LLVM_TAG="${LLVM_TAG:-llvmorg-23.1.3}"
PYTHON="${PYTHON:-$(command -v python3)}"
ROOT="${HK_LLVM_ROOT:-/dev/shm/hk-llvm}"
SRC="${LLVM_SRC:-$ROOT/llvm-project}"
PYTAG="$("$PYTHON" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"
BUILD="${LLVM_BUILD:-$ROOT/build-$PYTAG}"
INSTALL="${LLVM_INSTALL:-$ROOT/install-$PYTAG}"
JOBS="${JOBS:-$(nproc)}"
LINK_JOBS="${LINK_JOBS:-$JOBS}"

if [ ! -d "$SRC/mlir" ]; then
    git clone --depth 1 --branch "$LLVM_TAG" https://github.com/llvm/llvm-project.git "$SRC"
fi

"$PYTHON" -c 'import nanobind, numpy, yaml' || {
    echo "need nanobind>=2.9,<3.0, numpy and PyYAML for $PYTHON" >&2; exit 1; }

cmake -G Ninja -S "$SRC/llvm" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DMLIR_ENABLE_EXECUTION_ENGINE=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_PARALLEL_LINK_JOBS="$LINK_JOBS" \
    -DPython3_EXECUTABLE="$PYTHON" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL"

cmake --build "$BUILD" -j "$JOBS"
cmake --install "$BUILD" > /dev/null
echo "MLIR $LLVM_TAG for $PYTAG installed at $INSTALL"
