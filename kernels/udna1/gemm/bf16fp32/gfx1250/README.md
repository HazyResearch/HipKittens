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


## gfx1250 multicast-free ladder path

Some gfx1250 deployments do not support **multicast TDM loads**. The multicast rungs 09–12
(`09_gemm_wgc_multicast`, `10_gemm_epilogue`, `11_gemm_one_wave`, `12_gemm_two_waves`) use
multicast operand fills inside a 4×4 workgroup cluster and require multicast-capable hardware.

The multicast-free ladder path forks at rung 08 (`08_gemm_split_bar`) into four non-multicast cluster
rungs that keep cluster barriers and launch attributes but use per-workgroup `tdm::load_async`
with mask 0 (same fills as rung 08):

| Rung | Kernel | Adds over previous |
|------|--------|-------------------|
| 09 | `09_gemm_wgc_cluster` | 4×4 cluster + split cluster/workgroup barriers (no multicast) |
| 10 | `10_gemm_epilogue_nomc` | Staged C through LDS, `lock_simd` |
| 11 | `11_gemm_one_wave_nomc` | 1 wave/SIMD, 128×128 warp tile, pipelined operand ring |
| 12 | `12_gemm_two_waves_nomc` | 2 waves/SIMD, 64×128 warp tile, interleaved schedule |

The `_nomc` suffix marks the multicast-free non-multicast variants of `10_gemm_epilogue`,
`11_gemm_one_wave`, and `12_gemm_two_waves` (the multicast modules keep the bare names). Rung 09
keeps `cluster` in the name to parallel `09_gemm_wgc_multicast`.

The multicast rungs remain in the tree for hardware that supports multicast TDM loads.

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

The rungs collectively use eleven sync calls, which is a lot to meet at once -- but no rung meets
more than one new one. Reading upward, `00_gemm_naive` and `01_gemm_double_buf` use the full barrier
`sync::sync` and `wait_load` for their explicit GL -> RT -> ST path. `02_gemm_async` splits the
barrier into `arrive`/`wait` and adds `wait_async`, because the fill is now a copy engine that has to
be waited on separately. `07_gemm_tdm` swaps that engine, and with it `wait_async` for `wait_tdm`.
`09_gemm_wgc_cluster` (multicast-free) and `09_gemm_wgc_multicast` add the cluster barrier.
`11_gemm_one_wave` and `11_gemm_one_wave_nomc` add no new primitive at all -- they only wrap the
ones already there in `sched::compiler_fence`.

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
make all-kernels                                    # 17 pybind11 modules, no executables
python3 test.py                                     # every rung, four shapes, against torch.matmul
./gemm_ladder.py -r 25 -i 100 --json-out results.json
./gemm_ladder.py -r 25 -i 100 --torch 07_gemm_tdm 12_gemm_two_waves_nomc
```

Rungs are positional and default to the whole ladder. Each cell is 500 warmup iterations then 100
measured, each measured iteration timed by its own event pair with a 512 MB cache flush enqueued
before it, so no iteration reads what the one before it left resident. `--torch` adds `torch.matmul`
as a baseline arm, rotated and timed like a rung but reported below the table since it is not on the
ladder. Set `HIP_VISIBLE_DEVICES` to pick a card.

## Build

The kernels target `gfx1250`. The eight cluster kernels (`09_gemm_wgc_cluster` through
`12_gemm_two_waves_nomc` on the multicast-free path, and `09_gemm_wgc_multicast` through
`12_gemm_two_waves` on the multicast path) use static `__cluster_dims__`; ordinary HIP launches pick up their required 4x4
cluster metadata. ROCm 7.2.4 and earlier lack this support and fail those eight with a cascade that
reads as `shared_allocator` problems. Those eight need **ROCm 7.15 (clang 23)**; the nine
non-cluster kernels build on ROCm 7.2+. Correctness needs PyTorch with gfx1250 support, which is
**torch 2.11.0+rocm7.14.0**.

```
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    --extra-index-url https://pypi.org/simple "torch[device-gfx1250]==2.11.0+rocm7.14.0" pybind11
```
