#!/bin/bash
# Compile-probe each hardware facility for a given arch.
ARCH=${1:-gfx1100}
cd "$(dirname "$0")"
FEATS="F_BUFFER_LOAD_LDS F_GLOBAL_LOAD_LDS F_SCHED_BARRIER F_SCHED_GROUP_BARRIER F_IGLP_OPT
       F_SETPRIO F_PERMLANE16 F_PERMLANE32_SWAP F_DS_BPERMUTE F_DOT2 F_WMMA_BF16
       F_WMMA_F16_ACC F_SWMMAC F_DS_READ_TR"
echo "### arch=$ARCH"
for f in $FEATS; do
  if err=$(hipcc --offload-arch=$ARCH -D$f -O2 -c features.hip -o /dev/null 2>&1); then
    printf "  %-24s YES\n" "$f"
  else
    reason=$(echo "$err" | grep -oE "needs target feature [a-z0-9,+-]*" | head -1)
    printf "  %-24s no   %s\n" "$f" "$reason"
  fi
done
echo "  --- max static LDS ---"
for n in 16384 32768 49152 65536 81920 131072; do
  if hipcc --offload-arch=$ARCH -DF_LDS_SIZE -DLDS_FLOATS=$((n/4)) -O2 -c features.hip -o /dev/null 2>/dev/null; then
    printf "  %8d bytes  OK\n" $n
  else
    printf "  %8d bytes  FAIL\n" $n
  fi
done
