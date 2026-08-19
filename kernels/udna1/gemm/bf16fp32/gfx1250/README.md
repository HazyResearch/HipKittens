# gfx1250 GEMM optimization ladder

`bf16` -> `fp32` GEMM rungs, ordered upward from a naive kernel to the fastest. Each rung adds one
hardware or algorithmic feature -- plus whatever that feature's API or latency structure forces
along with it -- and changes nothing else. Accumulation is fp32 throughout, and each rung is named
for the feature it adds.

Every rung shares `common.h`, `harness.h`, the TN operand contract (`a` is `[M, K]`, `b` is
`[N, K]`, both K-contiguous, so the kernel computes `C = A . B^T` via `mma_ABt`) and a **column-major
`c`** -- allocate it as `torch.empty_strided((M, N), (1, M))`, since a row-major `c` is refused
rather than silently transposed. They are written against
`include/udna1/ops/warp/{sync,sched,cluster}/` and the gfx1250-only extensions to
`memory/tile/global_to_shared.cuh`, `memory/tile/shared_to_register.cuh` and `register/tile/mma.cuh`.


## gfx1250 A0 (MI450) multicast limitation

**Multicast TDM loads are not supported on gfx1250 A0 silicon.** The original rungs 10–12
(`gemm_wgc_multicast`, `gemm_epilogue`, `gemm_one_wave`) use multicast operand fills inside a
4×4 workgroup cluster and **must not be run on A0** — they wedge the GPU.

The A0-safe ladder path forks at rung 9 (`gemm_split_bar`) into three non-multicast cluster
rungs that keep cluster barriers and launch attributes but use per-workgroup `tdm::load_async`
with mask 0 (same fills as rung 9):

| Rung | Kernel | Adds over previous |
|------|--------|-------------------|
| 10 | `gemm_wgc_cluster` | 4×4 cluster + split cluster/workgroup barriers (no multicast) |
| 11 | `gemm_epilogue_nomc` | Staged C through LDS, `lock_simd` |
| 12 | `gemm_one_wave_nomc` | 1 wave/SIMD, 128×128 warp tile, pipelined operand ring |

The `_nomc` suffix marks the A0-safe non-multicast variants of `gemm_epilogue` and `gemm_one_wave`
(the multicast modules keep the bare names for A1+ silicon).

The multicast rungs remain in the tree for future A1+ silicon where multicast is supported.

## Tile geometry

Each rung declares its own `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `K_STEP`, `WARPS_M` and `WARPS_N` at the
top of its file, with the values derived from them, so a rung's shape is visible without opening
`common.h`.

The macro tile is `BLOCK_M` x `BLOCK_N` of output, split across a `WARPS_M` x `WARPS_N` grid of
warps. Bigger tiles do more math per byte of operand fetched -- arithmetic intensity for a square
tile is `M*N/(M+N)` FLOP per byte -- which is why the ladder climbs through 64x64, 128x128 and
256x256. 256x256 is the largest the register file allows: at four waves per SIMD a warp's 64x64
accumulator takes 128 VGPRs of the 256 a lane gets, and doubling the tile again would need 512.

`K_STEP` is the matrix instruction's K depth, fixed at 32. `BLOCK_K` is how much K one LDS stage
holds, so `BLOCK_K / K_STEP` sub-steps run per fill. Deepening it amortises the per-block rendezvous
over more math and puts the LDS row at 256 B, which is eight lanes per cache line and half the fill
requests.

## Synchronization

The rungs collectively use ten sync calls, which is a lot to meet at once -- but no rung meets more
than one new one. Reading upward, `gemm_naive` and `gemm_double_buf` use only the full barrier
`sync::sync`. `gemm_async` splits it into `arrive`/`wait` and adds `wait_async`, because the fill is
now a copy engine that has to be waited on separately. `gemm_tdm` swaps that engine, and with it
`wait_async` for `wait_tdm`. `gemm_wgc_cluster` (A0-safe) and `gemm_wgc_multicast` (A1+ multicast) add the cluster barrier. `gemm_one_wave` adds no
new primitive at all -- it only wraps the ones already there in `sched::compiler_fence`.

Each call in a kernel is labelled with the job it does and the resource it does it on: `wait for
data (LDS)`, `wait for data (TDM)`, `wait for everyone (workgroup)`, and so on. The two jobs are
independent -- a barrier orders warps and says nothing about memory, a counter wait orders one
warp's own transfers and says nothing about other warps -- and the four rules that follow from that,
including which drains may be partial and why, are in the header of
`include/udna1/ops/warp/sync/barrier.cuh`.

Every kernel file opens with a `Kernel Specification` block -- tile geometry, occupancy in waves per
SIMD, register and spill counts, LDS footprint, the per-K-block sync inventory and arithmetic
intensity -- so the per-kernel facts sit next to the code rather than here.

## Benchmark

`test.py` is the correctness gate and `gemm_ladder.py` is the measurement campaign; a rung number for publication comes from the latter.

```
make all-kernels                                    # 15 pybind11 modules, no executables
python3 test.py                                     # every rung, four shapes, against torch.matmul
./gemm_ladder.py -r 25 -i 100                       # the table above
./gemm_ladder.py -r 25 -i 100 --torch gemm_tdm gemm_one_wave
```

Rungs are positional and default to the whole ladder. Each cell is 500 warmup iterations then 100
measured, each measured iteration timed by its own event pair with a 512 MB cache flush enqueued
before it, so no iteration reads what the one before it left resident. `--torch` adds `torch.matmul`
as a baseline arm, rotated and timed like a rung but reported below the table since it is not on the
ladder. Set `HIP_VISIBLE_DEVICES` to pick a card.

## Build

The kernels target `gfx1250`. All seven cluster rungs (`gemm_wgc_cluster`, `gemm_epilogue_nomc`,
`gemm_one_wave_nomc`, and the multicast `gemm_wgc_multicast`, `gemm_epilogue`, `gemm_one_wave`) use `__cluster_dims__` and the `hipLaunchAttributeClusterDimension` launch
attribute; ROCm 7.2.4 and earlier have neither and fail those three with a cascade of errors that
read as `shared_allocator` problems. Those seven need **ROCm 7.15 (clang 23)**; the other eight build
on ROCm 7.2+. Correctness needs PyTorch with gfx1250 support, which is **torch 2.11.0+rocm7.14.0**.

```
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    --extra-index-url https://pypi.org/simple "torch[device-gfx1250]==2.11.0+rocm7.14.0" pybind11
```
