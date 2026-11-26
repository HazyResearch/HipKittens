#!/bin/bash
# Build gemm.cpp once per tiling configuration and report register usage plus
# TFLOPs, so the tile shape can be picked from measurements rather than a
# register-pressure estimate. Each line is one configuration.
#
#   ./sweep.sh "BLOCK_M=256 BLOCK_N=128 WARP_ROWS=4" ...
#
# With no arguments it runs the default grid below.
set -u
QB_ARGS="${QB_ARGS:-}"

run() {
    local desc="$1"
    # HK_MULTI_CONFIG=0 pins the kernel to exactly the tiling named here; the
    # shipped dispatch picks between two of them by shape, which would make a
    # sweep measure the selection rule instead of the configuration.
    local flags="-DHK_MULTI_CONFIG=0"
    for kv in $desc; do flags="$flags -D${kv/=/=}"; done
    make clean >/dev/null 2>&1
    if ! make EXTRA_HIPFLAGS="$flags" >/tmp/build.log 2>&1; then
        printf '%-58s | BUILD FAILED: %s\n' "$desc" "$(grep -m1 -E 'error|static_assert|Error' /tmp/build.log | cut -c1-90)"
        return
    fi
    local vgpr spill occ out
    # The remark lines are prefixed with file:line:col, so anchor on the label.
    vgpr=$(sed -n 's/.*remark: *VGPRs: \([0-9]*\).*/\1/p'       /tmp/build.log | head -1)
    spill=$(sed -n 's/.*remark: *VGPRs Spill: \([0-9]*\).*/\1/p' /tmp/build.log | head -1)
    occ=$(sed -n 's/.*remark: *Occupancy \[waves\/SIMD\]: \([0-9]*\).*/\1/p' /tmp/build.log | head -1)
    out=$(timeout 600 python3 quickbench.py $QB_ARGS 2>&1 | tail -1)
    printf '%-58s | vgpr=%-3s spill=%-2s occ=%-2s | %s\n' "$desc" "$vgpr" "$spill" "$occ" "$out"
}

if [ $# -gt 0 ]; then
    for cfg in "$@"; do run "$cfg"; done
    exit 0
fi

echo "config                                                     | registers            | TFLOPS 4096^3 / 8192x8192x4096"
run "BLOCK_M=128 BLOCK_N=128 K_STEP=32 DOT_SLICE=16 WARP_ROWS=4"
run "BLOCK_M=128 BLOCK_N=128 K_STEP=32 DOT_SLICE=32 WARP_ROWS=4"
run "BLOCK_M=128 BLOCK_N=128 K_STEP=64 DOT_SLICE=16 WARP_ROWS=4"
run "BLOCK_M=256 BLOCK_N=128 K_STEP=32 DOT_SLICE=16 WARP_ROWS=4"
run "BLOCK_M=128 BLOCK_N=256 K_STEP=32 DOT_SLICE=16 WARP_ROWS=2"
run "BLOCK_M=256 BLOCK_N=128 K_STEP=32 DOT_SLICE=16 WARP_ROWS=8 NUM_WARPS=16"
run "BLOCK_M=256 BLOCK_N=256 K_STEP=32 DOT_SLICE=16 WARP_ROWS=4 NUM_WARPS=16"
run "BLOCK_M=256 BLOCK_N=256 K_STEP=32 DOT_SLICE=16 WARP_ROWS=8 NUM_WARPS=16"
