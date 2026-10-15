#!/usr/bin/env bash
# Build kernels/cdna4/attn/gqa_causal once per sequence length for bench_layer.py, as
#   bench/attn_build/tk_attn_H<H>_KV<H_KV>_D<Dh>_N<M>*.so
# The kernel's shape is fixed at compile time. kernel.cpp names its module tk_kernel; -Dtk_kernel=<name>
# gives each build its own module name, because Python returns the first-loaded module for every later
# load of the same name (measured: the N=4096 load ran the N=2048 kernel).
# Usage: bench/build_attn.sh <H> <H_KV> <Dh> <M> [<M> ...]     e.g. bench/build_attn.sh 32 8 128 2048 4096 8192
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo "usage: $0 <H> <H_KV> <Dh> <M> [<M> ...]" >&2
  exit 2
fi
H=$1; H_KV=$2; DH=$3; shift 3
BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$BENCH/../../../../../.." && pwd)
export THUNDERKITTENS_ROOT="$ROOT"      # kreb-activate may export a different root
mkdir -p "$BENCH/attn_build"
for M in "$@"; do
  NAME="tk_attn_H${H}_KV${H_KV}_D${DH}_N${M}"
  make -C "$ROOT/kernels/cdna4/attn/gqa_causal" GPU_TARGET=CDNA4 ATTN_B=1 ATTN_N="$M" ATTN_H="$H" \
       ATTN_H_KV="$H_KV" ATTN_D="$DH" EXTRA_CPPFLAGS="-Dtk_kernel=$NAME" TARGET="$BENCH/attn_build/$NAME"
done
