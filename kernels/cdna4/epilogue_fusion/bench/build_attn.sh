#!/usr/bin/env bash
# Builds gqa_causal per sequence length into bench/attn_build/tk_attn_H<H>_KV<H_KV>_D<Dh>_N<M>*.so.
# Each build needs a unique module name: Python reuses the first-loaded module of a given name.
# Usage: bench/build_attn.sh <H> <H_KV> <Dh> <M> [<M> ...]   e.g. 32 8 128 2048 4096 8192
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo "usage: $0 <H> <H_KV> <Dh> <M> [<M> ...]" >&2
  exit 2
fi
H=$1; H_KV=$2; DH=$3; shift 3
BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$BENCH/../../../.." && pwd)
export THUNDERKITTENS_ROOT="$ROOT"
mkdir -p "$BENCH/attn_build"
for M in "$@"; do
  NAME="tk_attn_H${H}_KV${H_KV}_D${DH}_N${M}"
  make -C "$ROOT/kernels/cdna4/attn/gqa_causal" GPU_TARGET=CDNA4 ATTN_B=1 ATTN_N="$M" ATTN_H="$H" \
       ATTN_H_KV="$H_KV" ATTN_D="$DH" EXTRA_CPPFLAGS="-Dtk_kernel=$NAME" TARGET="$BENCH/attn_build/$NAME"
done
