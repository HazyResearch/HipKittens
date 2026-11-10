/**
 * @file moe_compute.cuh
 * @brief Shared core for the fused per-expert MoE block-scaled GEMM (gfx1250): config, globals, the
 *   moe_compute_tile core, load/scale helpers, and the host validation + BLOCK_M router.
 *
 * MXFP8 activation (FP8 E4M3) x MXFP4 weight (FP4 E2M1), both E8M0 per-32 block-scaled, f32 accumulate
 * via mma_ABt_scaled. Weights are consumed PRE-SHUFFLED into WMMA order (-DMOE_PRESHUFFLE).
 *
 * Fragments (not standalone TUs), assembled by moe_gemm.cpp into the pybind module: moe_gemm1 (GEMM1
 * f32), moe_gemm2 (down-proj bf16), moe_swiglu (SwiGLU + SwiGLU/MXFP8 variants). moe_gemm_adaptive.cpp
 * re-includes the chain once per BLOCK_M {16,32,48,64} (MOE_AS_ENGINE) with a density-based router.
 *
 * Two build configs: default DECODE and -DMOE_PREFILL. Requires the fixed LLVM (clang >= a8949ba8):
 * older clang miscompiles the E>1 gemm2 down-proj.
 */

// MOE_AS_ENGINE: the adaptive dispatcher (moe_gemm_adaptive.cpp) #includes this file once per BLOCK_M
// in distinct namespaces and provides the headers + `using namespace kittens;` once; skip them here.
#ifndef MOE_AS_ENGINE
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "kittens.cuh"
#include "moe_tiles.cuh"   // MoE-local flat preshuffle shape (st_flat) + load_b_preshuffled
#include "pyutils/pyutils.cuh"

using namespace kittens;
#endif  // MOE_AS_ENGINE

// Over-read contract: A/B stream global->LDS via tdm::load_async, which reads up to 16B past each
// operand's last K-slice. B is a single packed [E,N,K/2] pointer (no per-expert stride), so dispatch_*
// callers pad the END of the A and B allocations by >=16B, not per expert (a mid-buffer over-read lands
// in the next expert's bytes, loaded but unused). Scales are not over-read.

// Weight preshuffle (default ON): weights pre-reordered offline into WMMA order as contiguous 1024B
// blocks [E][N/16][K/128][1024]; load_b_preshuffled reads lane L at data+L*32. The host upload (gated
// by MOE_PRESHUFFLE) must match.
#ifndef MOE_PRESHUFFLE
#define MOE_PRESHUFFLE 1
#endif

// ======================== Configuration ========================

#ifndef MOE_BLOCK_M
#define MOE_BLOCK_M 64             // token (M) tile height (16/32/48/64); multiple of 16. Narrow = decode.
#endif
constexpr int BLOCK_M    = MOE_BLOCK_M;
#ifndef MOE_BLOCK_N
#define MOE_BLOCK_N 64             // output N tile width (prefill uses 128); requires N % BLOCK_N == 0.
#endif
constexpr int BLOCK_N    = MOE_BLOCK_N;
constexpr int K_STEP     = 128;     // WMMA K-depth (fixed by the 16x16x128 instruction)
#ifndef MOE_KTILE
#define MOE_KTILE 256               // global->LDS load tile K-width (BLOCK_K); override via -DMOE_KTILE
#endif
constexpr int KTILE      = MOE_KTILE;            // must be a multiple of K_STEP
constexpr int KSUB       = KTILE / K_STEP;       // WMMA steps per loaded K-tile
// KTILE must be a positive multiple of K_STEP (else KSUB==0 -> silent zero output); TDM pins 256.
static_assert(KTILE >= K_STEP && KTILE % K_STEP == 0,
              "MOE_KTILE must be a positive multiple of K_STEP (128)");
#ifndef MOE_XCD
#define MOE_XCD 1                  // L2 program-id swizzle domains; 1=off. gfx1250 has 2 AIDs (HBM-bound, off).
#endif
#ifndef MOE_MTILES
#define MOE_MTILES 1               // weight-stationary group: m-tiles/WG sharing one weight stream. 1=off.
// MTILES>1 CONTRACT: each expert's m-tile count in expt_data (incl -1 padding) is a multiple of MTILES,
// so a group never straddles two experts.
#endif

// Two configs, host-selected: default DECODE (single-buffer, per-K-step S2R, per-warp B); -DMOE_PREFILL
// = PREFILL (hoisted S2R, shared 2D-TDM B, double-buffer). Each is a straight-line moe_compute_tile
// body; decode's only switch is MOE_HOIST for BM32.
#ifndef MOE_PREFILL
#if MOE_BLOCK_M == 32
// BM32 decode must also hoist: it is the only WARPS_M==2 & NSUB_M==1 tile, where the non-hoist path's
// single reused rA lets the scheduler overlap its ds_load with the consuming WMMA and miscompile.
#define MOE_HOIST   1
#endif
#endif

// Warp tiling: BLOCK_M % 32 == 0 -> 2x2 split (WARP_M=BLOCK_M/2); else 1x4 (all 4 warps in N), because
// WARP_M % 16 != 0 (BM16/BM48) would be silent no-output. NUM_WARPS=4 either way.
#if MOE_BLOCK_M % 32 != 0
constexpr int WARPS_M    = 1;
constexpr int WARPS_N    = 4;
#else
constexpr int WARPS_M    = 2;
constexpr int WARPS_N    = 2;
#endif
constexpr int WARP_M     = BLOCK_M / WARPS_M;   // 32 at BLOCK_M=64; 16 at BLOCK_M<=16 or 32
constexpr int WARP_N     = BLOCK_N / WARPS_N;   // 32 at BLOCK_N=64/WARPS_N=2; 16 at WARPS_N=4
constexpr int NUM_WARPS  = WARPS_M * WARPS_N;   // 4
constexpr int NUM_THREADS = WARP_THREADS * NUM_WARPS; // 128
constexpr int NSUB_M     = WARP_M / 16;         // 16x16 WMMA output subtiles per warp, M dir
constexpr int NSUB_N     = WARP_N / 16;         // N dir
static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of the 16-row WMMA tile");
static_assert(WARP_M * WARPS_M == BLOCK_M, "BLOCK_M must equal WARP_M * WARPS_M");
static_assert(WARP_N * WARPS_N == BLOCK_N, "BLOCK_N must equal WARP_N * WARPS_N");
static_assert(NSUB_M >= 1 && NSUB_N >= 1,
              "each warp needs >=1 16x16 WMMA subtile; check BLOCK_M/BLOCK_N vs the warp split");
// WARP_M/WARP_N must be whole multiples of the 16x16 WMMA subtile, else NSUB_* truncates and the
// leftover rows/cols are silently never computed (e.g. BLOCK_N=48 -> WARP_N=24 drops 8 columns).
static_assert(WARP_M % 16 == 0, "WARP_M must be a multiple of 16; check BLOCK_M / WARPS_M");
static_assert(WARP_N % 16 == 0, "WARP_N must be a multiple of 16; check BLOCK_N / WARPS_N");
// The MXFP8 re-quant epilogue maps one thread per output row (m = threadIdx.x, if m < BLOCK_M),
// so rows beyond NUM_THREADS would never be quantized.
static_assert(BLOCK_M <= NUM_THREADS, "BLOCK_M must be <= NUM_THREADS (mxfp8 epilogue is one thread/row)");

// __launch_bounds__ resident-block floor (default 1). Raising it can spill VGPRs -> re-check correctness.
#ifndef MOE_MIN_BLOCKS
#define MOE_MIN_BLOCKS 1
#endif
#define MOE_LAUNCH_ATTRS __launch_bounds__(NUM_THREADS, MOE_MIN_BLOCKS)

// Typed rt<float,16,16,col> accumulator (WMMA output tiles[0][0].data is float2[4]), read via C_AT.
using acc_rt = kittens::rt<float, 16, 16, kittens::ducks::rt_layout::col, kittens::ducks::rt_shape::rt_16x16>;
#define C_AT(r, mi, ni, k) C_accum[r][mi][ni].tiles[0][0].data[k]

constexpr int MTILES = MOE_MTILES;                  // weight-stationary group size
static_assert(MTILES >= 1, "MOE_MTILES must be >= 1");
// The compute path uses static __shared__ st tiles (ts_A / ts_Ball), so it needs no dynamic shared.
// The SwiGLU epilogue's raw_lds is a separate dynamic allocation, sized by its dispatcher.
constexpr int TOTAL_LDS_BYTES = 0;

typedef int int4_vec __attribute__((__vector_size__(16)));

// ======================== Scaled WMMA (MXFP8 x MXFP4) ========================
// The scaled WMMA lives in the library (kittens::mma_ABt_scaled, mma.cuh). Its C operand carries the
// running accumulator (D = A*B + C across K-steps): matrix_a = FP8 activation with scaleA, matrix_b =
// FP4 weight with scaleB, both block-scaled.

// ======================== LDS->register loads ========================



// E8M0 block scales as a packed 4x-E8M0 (32-bit) global view: one element is the 4 e8m0 that scale one
// 128-wide WMMA K-step of one row. `gl` unpacks its dtype, so an fp8e8m0 gl would index in bytes and
// return one e8m0; typing the view on a 4-byte element makes gl[{row,gidx}] return the packed u32 the
// WMMA consumes. Columns are K/K_STEP (= (K/32) e8m0 / 4). A: [M, K/K_STEP]; B: [E, N, K/K_STEP].
using scale_a_gl = kittens::gl<int, 1,  1, -1, -1>;
using scale_b_gl = kittens::gl<int, 1, -1, -1, -1>;
__host__ inline scale_a_gl make_scale_a_gl(const uint8_t *sa, int M, int K) {
    return kittens::make_gl<scale_a_gl>(reinterpret_cast<uint64_t>(sa), 1, 1, M, K / K_STEP);
}
__host__ inline scale_b_gl make_scale_b_gl(const uint8_t *sb, int E, int N, int K) {
    return kittens::make_gl<scale_b_gl>(reinterpret_cast<uint64_t>(sb), 1, E, N, K / K_STEP);
}

// Output C as a row-major [M,N] gl so the epilogues store via kittens::store(c_gl, acc_rt, coord).
// gl's ctor/make_gl are __host__-only, so the kernels take the gl as a launch parameter (built here).
using out_gl_f32  = kittens::gl<float, 1, 1, -1, -1>;
using out_gl_bf16 = kittens::gl<kittens::bf16, 1, 1, -1, -1>;
__host__ inline out_gl_f32  make_out_gl_f32(float *c, int M, int N) {
    return kittens::make_gl<out_gl_f32>(reinterpret_cast<uint64_t>(c), 1, 1, M, N);
}
__host__ inline out_gl_bf16 make_out_gl_bf16(kittens::bf16 *c, int M, int N) {
    return kittens::make_gl<out_gl_bf16>(reinterpret_cast<uint64_t>(c), 1, 1, M, N);
}

// Persistence: each workgroup handles PERSIST_NI consecutive tile_n's. Default 1 (off) -- decode
// relies on many small concurrent workgroups for cross-workgroup latency hiding.
constexpr int PERSIST_NI = 1;

// ======================== Kernel globals ========================
// Per-expert grouped GEMM, contiguous pre-grouped (layer-2) layout.
//   Activations are already sorted by expert into one contiguous [M,K] buffer (M = total
//   padded token-rows = grid_m * BLOCK_M). Each m-tile pid_m maps to one expert via
//   expt_data[pid_m] (-1 = padding tile, skipped). Weights carry an expert dim [E,N,K].
// Epilogue params (bias/gammas/swiglu/quant) and the gather (layer-1) path come later.

// Host-side launch configuration (mirrors the gemm-ladder launch_config in kernels/cdna5/gemm). Kept
// out of the *_globals structs so the struct passed to each kernel carries only kernel inputs, not the
// launch geometry/stream. All MoE ops share one grid/block/dynamic-LDS shape, built from any *_globals.
struct moe_launch_config {
    hipStream_t stream;
    dim3 grid;
    dim3 block;
    template <class G>
    explicit moe_launch_config(const G &g, hipStream_t launch_stream = 0)
        : stream(launch_stream),
          grid((g.M / BLOCK_M + MTILES - 1) / MTILES, g.N / BLOCK_N),
          block(NUM_THREADS) {}
    size_t dynamic_shared_memory() const { return TOTAL_LDS_BYTES; }
};

// Per-kernel globals structs and the SwiGLU pointwise helper live with their kernels (each carries
// only kernel inputs; moe_launch_config above is templated over any of them):
//   moe_globals            -> moe_gemm1.cpp
//   moe_gemm2_globals      -> moe_gemm2.cpp
//   moe_swiglu_globals / moe_swiglu_mxfp8_globals / moe_swiglu_apply -> moe_swiglu.cpp



// ======================== Shared GEMM core (weight-stationary over MTILES) ========================
// Computes MTILES output tiles [BLOCK_M, BLOCK_N] for one weight column (tile_n) of one expert,
// streaming the shared weight tile from HBM ONCE per K-tile and reusing it across all MTILES
// activation tiles (cuts weight-HBM by up to MTILES x). The MTILES tiles are the consecutive
// m-tiles [group_m*MTILES .. +MTILES) -- by the MOE_MTILES contract they share one expert (or are
// trailing -1 padding). Returns false if the whole group is padding. Per-tile validity is in
// tile_valid[r]; valid tiles use group_expert + row_base[r].
__device__ __forceinline__ bool moe_compute_tile(
    const uint8_t *a, const uint8_t *b, const scale_a_gl &scale_a, const scale_b_gl &scale_b,
    const int32_t *expt_data, int N, int K, int group_m, int tile_n,
    acc_rt (&C_accum)[MTILES][NSUB_M][NSUB_N],
    int (&row_base)[MTILES], bool (&tile_valid)[MTILES],
    int &warp_row, int &warp_col, int &group_expert)
{
    #pragma unroll
    for (int r = 0; r < MTILES; r++)
        #pragma unroll
        for (int i = 0; i < NSUB_M; i++)
            for (int j = 0; j < NSUB_N; j++)
                for (int k = 0; k < 4; k++) C_AT(r, i, j, k) = {0.f, 0.f};

    // Group's m-tiles: pid = group_m*MTILES + r. By contract every expert's tile-run is a multiple
    // of MTILES, so all valid tiles in a group share one expert; -1 tiles (if any) are the tail.
    const int pid0 = group_m * MTILES;
    group_expert = -1;
    #pragma unroll
    for (int r = 0; r < MTILES; r++) {
        const int e_r = expt_data[pid0 + r];
        tile_valid[r] = (e_r >= 0);
        row_base[r]   = (pid0 + r) * BLOCK_M;
        // MOE_MTILES contract: all valid tiles in a group share one expert (padding -1s are the
        // tail). A straddle silently multiplies a tile by the wrong expert's weights AND scales.
        assert((e_r < 0 || group_expert < 0 || e_r == group_expert) &&
               "MOE_MTILES contract violated: m-tile group straddles two experts");
        if (e_r >= 0 && group_expert < 0) group_expert = e_r;   // first valid expert
    }
    if (group_expert < 0) return false;        // whole group is padding -> skip

    const int wid = warpid();
    warp_row = wid / WARPS_N;
    warp_col = wid % WARPS_N;

    const int a_stride      = K;
    const int b_half_stride = K / 2;
    const int k_tiles       = K / KTILE;       // global load tiles (KTILE=256 K each)

    // Weight (shared) bases use group_expert + tile_n. Activation base is per m-tile. Scales are read
    // straight from the scale_a/scale_b gls by absolute (row, K-group) coord in load_scales_typed.
    const size_t w_expert_off  = (size_t)group_expert * N * (size_t)b_half_stride;
    const uint8_t *const b_tile_base = b + w_expert_off + (size_t)tile_n * BLOCK_N * b_half_stride;
    // Preshuffled weight: same per-expert byte size (N*K/2) reordered to [g=N/16][gk=K/128][1024].
    const uint8_t *const b_ps_expert_base = b + w_expert_off;
    const int b_ps_g0         = tile_n * (BLOCK_N / 16);   // first N-row group of this tile
    const int b_ps_k_over_128 = K / K_STEP;                // K-steps per row (block stride unit)
    const uint8_t *a_tile_base[MTILES];
    #pragma unroll
    for (int r = 0; r < MTILES; r++)
        a_tile_base[r] = a + (size_t)row_base[r] * a_stride;

    // Per K-tile: one BLOCK_M x KTILE A load (issued once by warp_col==0, shared across warp_col) plus
    // each warp's B as NSUB_N chunks of KSUB contiguous 1024B preshuffle blocks. Then per K-step: load
    // the A/B subtiles to registers and mma_ABt_scaled into C_accum. Requires MOE_PRESHUFFLE.
    const int warp_a_row0 = warp_row * WARP_M;
    const int warp_b_row0 = warp_col * WARP_N;
    // A is one shared BLOCK_M x KTILE tile; each warp_row reads its M-slice at warp_a_row0. Decode
    // stages B per warp (ts_B, duplicated across warp_rows); prefill stages one shared all-N-groups
    // tile (ts_Ball) filled by a single 2D TDM load.
    static_assert(WARPS_M == 1 || WARPS_M == 2, "WARPS_M must be 1 or 2");
    constexpr int NW = WARPS_M * WARPS_N;
    using a_st = kittens::st<kittens::fp8e4m3, BLOCK_M, KTILE, kittens::ducks::st_shape::st_16x32_padded<256, 16>>;
    using b_st = kittens::st<kittens::fp8e4m3, 16, KSUB * 64, kittens::ducks::st_shape::st_flat<16, KSUB * 64>>;
    using a_rt = kittens::rt<kittens::fp8e4m3, 16, 128, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x128>;
    using b_rt = kittens::rt<kittens::fp4e2m1, 16, 128, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x128>;
    // The two data paths differ in B staging, S2R issue, and buffering, so they are written out as
    // two straight-line kernels (owner: "two kernels, duplicated code is AOK") rather than one body
    // interleaved with MOE_TDMB/MOE_DB/MOE_HOIST levers. -DMOE_PREFILL selects the prefill body.
    const int wtid = warpid();
#ifdef MOE_PREFILL
    // ===================== PREFILL: hoisted S2R + shared 2D-TDM B + double buffer =====================
    constexpr int NBUF = 2;
    __shared__ a_st ts_A[NBUF][MTILES];         // shared A (WARPS_M==1)
    // Shared all-N-groups B tile (contiguous 1024B preshuffle blocks, block (r_g,s) at (r_g*KSUB+s)*1024),
    // filled by one 2D TDM load per K-tile -- avoids the per-warp-B TDM-engine contention.
    using b_all_st = kittens::st<kittens::fp8e4m3, 16, (BLOCK_N / 16) * KSUB * 64,
                                 kittens::ducks::st_shape::st_flat<16, (BLOCK_N / 16) * KSUB * 64>>;
    __shared__ b_all_st ts_Ball[NBUF];          // all N-groups, contiguous

    // A once by wtid==0 (single issuer: warp_col==0 is 2 warps when WARPS_M==2, which would double-load
    // the shared A tile), then ONE 2D TDM load of all N-groups by wtid==0 (1 op vs 8 per-warp).
    auto load_ktile_typed = [&](int kt_, int buf) {
        if (wtid == 0) {
            #pragma unroll
            for (int r = 0; r < MTILES; r++)
                kittens::tdm::load_async<kittens::ducks::st_shape::st_16x32_padded<256, 16>, BLOCK_M, KTILE>(
                    ts_A[buf][r], a_tile_base[r] + (size_t)kt_ * KTILE, BLOCK_M, KTILE, a_stride);
            constexpr int NGROUP_TDM = BLOCK_N / 16;
            kittens::tdm::load_async<kittens::ducks::st_shape::st_flat<NGROUP_TDM, KSUB * 1024>, NGROUP_TDM, KSUB * 1024>(
                ts_Ball[buf],
                b_ps_expert_base + (size_t)((size_t)b_ps_g0 * b_ps_k_over_128 + (size_t)kt_ * KSUB) * 1024,
                NGROUP_TDM, KSUB * 1024, b_ps_k_over_128 * 1024);
        }
    };
    // Scales are per-lane WMMA register operands (not an LDS tile), so each lane reads its own packed u32
    // (4 e8m0) straight from the gl by coord -- no load_async/LDS hop. laneid%16 maps 16 rows over 32 lanes.
    auto load_scales_typed = [&](int kt, uint32_t (&sA)[MTILES][NSUB_M][KSUB], uint32_t (&sB)[NSUB_N][KSUB]) {
        const int l16 = kittens::laneid() % 16;
        #pragma unroll
        for (int s = 0; s < KSUB; s++) {
            const int gidx = kt * KSUB + s;
            #pragma unroll
            for (int r = 0; r < MTILES; r++)
                #pragma unroll
                for (int mi = 0; mi < NSUB_M; mi++)
                    sA[r][mi][s] = (uint32_t) scale_a[{ 0, 0, row_base[r] + warp_a_row0 + mi * 16 + l16, gidx }];
            #pragma unroll
            for (int ni = 0; ni < NSUB_N; ni++)
                sB[ni][s] = (uint32_t) scale_b[{ 0, group_expert, tile_n * BLOCK_N + warp_b_row0 + ni * 16 + l16, gidx }];
        }
    };
    // Hoist all KSUB S2R loads into register arrays, drain the LDS reads once, then issue the K-tile's
    // WMMAs back-to-back (dense issue). Costs VGPRs (KSUB x live operands).
    auto compute_ktile = [&](int buf, const uint32_t (&sA)[MTILES][NSUB_M][KSUB], const uint32_t (&sB)[NSUB_N][KSUB]) {
        b_rt rB_all[KSUB][NSUB_N];
        a_rt rA_all[KSUB][MTILES][NSUB_M];
        #pragma unroll
        for (int s = 0; s < KSUB; s++) {
            #pragma unroll
            for (int ni = 0; ni < NSUB_N; ni++)
                kittens::load_b_preshuffled(rB_all[s][ni], ts_Ball[buf], ((warp_col * NSUB_N + ni) * KSUB + s) * 1024);
            #pragma unroll
            for (int r = 0; r < MTILES; r++)
                #pragma unroll
                for (int mi = 0; mi < NSUB_M; mi++)
                    kittens::load(rA_all[s][r][mi], kittens::subtile_view(ts_A[buf][r], warp_a_row0 + mi * 16, s * K_STEP));
        }
        kittens::sync::wait_ds<0>();          // single drain for the whole K-tile's hoisted S2R
        #pragma unroll
        for (int s = 0; s < KSUB; s++)
            #pragma unroll
            for (int r = 0; r < MTILES; r++)
                #pragma unroll
                for (int mi = 0; mi < NSUB_M; mi++)
                    #pragma unroll
                    for (int ni = 0; ni < NSUB_N; ni++) {
                        const kittens::fp8e8m0_4 sav{ sA[r][mi][s] }, sbv{ sB[ni][s] };  // e8m0x4 from u32, no cast
                        kittens::mma_ABt_scaled(
                            C_accum[r][mi][ni], rA_all[s][r][mi], rB_all[s][ni], C_accum[r][mi][ni],
                            &sav, &sbv);
                        // Pin the WMMA against the surrounding ds_load schedule; without it the machine
                        // scheduler reorders ds_load/WMMA at high B-operand pressure and miscompiles.
                        __builtin_amdgcn_sched_barrier(0);
                    }
    };
    // Double-buffer: prefetch kt+1 during kt's compute to overlap the weight load.
    load_ktile_typed(0, 0);
    kittens::sync::wait_tdm<0>();
    kittens::sync::sync();
    for (int kt = 0; kt < k_tiles; kt++) {
        const int cur = kt & 1, nxt = (kt + 1) & 1;
        uint32_t sA[MTILES][NSUB_M][KSUB], sB[NSUB_N][KSUB];
        load_scales_typed(kt, sA, sB);
        kittens::sync::wait_load<0>();
        if (kt + 1 < k_tiles) load_ktile_typed(kt + 1, nxt);
        compute_ktile(cur, sA, sB);
        kittens::sync::wait_tdm<0>();              // drain the prefetch; barrier makes nxt visible + WAR on cur
        kittens::sync::sync();
    }
#else
    // ===================== DECODE: per-warp B + single buffer + per-K-step S2R =====================
    constexpr int NBUF = 1;
    __shared__ a_st ts_A[NBUF][MTILES];         // shared A (WARPS_M==1)
    __shared__ b_st ts_B[NBUF][NW][NSUB_N];     // per-warp B

    // A once by wtid==0 (single issuer: warp_col==0 is 2 warps when WARPS_M==2, which would double-load
    // the shared A tile), then each warp loads its NSUB_N B chunks (KSUB contiguous 1024B blocks).
    auto load_ktile_typed = [&](int kt_, int buf) {
        if (wtid == 0) {
            #pragma unroll
            for (int r = 0; r < MTILES; r++)
                kittens::tdm::load_async<kittens::ducks::st_shape::st_16x32_padded<256, 16>, BLOCK_M, KTILE>(
                    ts_A[buf][r], a_tile_base[r] + (size_t)kt_ * KTILE, BLOCK_M, KTILE, a_stride);
        }
        #pragma unroll
        for (int ni = 0; ni < NSUB_N; ni++) {
            const size_t blk0 = ((size_t)(b_ps_g0 + warp_col * NSUB_N + ni) * b_ps_k_over_128
                                 + (size_t)(kt_ * KSUB)) * 1024;
            kittens::tdm::load_async<kittens::ducks::st_shape::st_flat<16, KSUB * 64>, 16, KSUB * 64>(
                ts_B[buf][wtid][ni], b_ps_expert_base + blk0, 16, KSUB * 64, KSUB * 64);
        }
    };
    // Scales are per-lane WMMA register operands (not an LDS tile), so each lane reads its own packed u32
    // (4 e8m0) straight from the gl by coord -- no load_async/LDS hop. laneid%16 maps 16 rows over 32 lanes.
    auto load_scales_typed = [&](int kt, uint32_t (&sA)[MTILES][NSUB_M][KSUB], uint32_t (&sB)[NSUB_N][KSUB]) {
        const int l16 = kittens::laneid() % 16;
        #pragma unroll
        for (int s = 0; s < KSUB; s++) {
            const int gidx = kt * KSUB + s;
            #pragma unroll
            for (int r = 0; r < MTILES; r++)
                #pragma unroll
                for (int mi = 0; mi < NSUB_M; mi++)
                    sA[r][mi][s] = (uint32_t) scale_a[{ 0, 0, row_base[r] + warp_a_row0 + mi * 16 + l16, gidx }];
            #pragma unroll
            for (int ni = 0; ni < NSUB_N; ni++)
                sB[ni][s] = (uint32_t) scale_b[{ 0, group_expert, tile_n * BLOCK_N + warp_b_row0 + ni * 16 + l16, gidx }];
        }
    };
    auto compute_ktile = [&](int buf, const uint32_t (&sA)[MTILES][NSUB_M][KSUB], const uint32_t (&sB)[NSUB_N][KSUB]) {
#ifdef MOE_HOIST
        // BM32 decode must hoist (see the config selector): on the per-K-step path its single reused rA
        // lets the scheduler overlap the ds_load with the consuming WMMA and miscompile. Hoist all KSUB
        // S2R loads, drain once, then issue the WMMAs dense.
        b_rt rB_all[KSUB][NSUB_N];
        a_rt rA_all[KSUB][MTILES][NSUB_M];
        #pragma unroll
        for (int s = 0; s < KSUB; s++) {
            #pragma unroll
            for (int ni = 0; ni < NSUB_N; ni++)
                kittens::load_b_preshuffled(rB_all[s][ni], ts_B[buf][wtid][ni], s * 1024);
            #pragma unroll
            for (int r = 0; r < MTILES; r++)
                #pragma unroll
                for (int mi = 0; mi < NSUB_M; mi++)
                    kittens::load(rA_all[s][r][mi], kittens::subtile_view(ts_A[buf][r], warp_a_row0 + mi * 16, s * K_STEP));
        }
        kittens::sync::wait_ds<0>();          // single drain for the whole K-tile's hoisted S2R
        #pragma unroll
        for (int s = 0; s < KSUB; s++)
            #pragma unroll
            for (int r = 0; r < MTILES; r++)
                #pragma unroll
                for (int mi = 0; mi < NSUB_M; mi++)
                    #pragma unroll
                    for (int ni = 0; ni < NSUB_N; ni++) {
                        const kittens::fp8e8m0_4 sav{ sA[r][mi][s] }, sbv{ sB[ni][s] };  // e8m0x4 from u32, no cast
                        kittens::mma_ABt_scaled(
                            C_accum[r][mi][ni], rA_all[s][r][mi], rB_all[s][ni], C_accum[r][mi][ni],
                            &sav, &sbv);
                        __builtin_amdgcn_sched_barrier(0);
                    }
#else
        // BM16 decode: per-K-step S2R + WMMA, one reused rA (lowest VGPR pressure, HBM-latency-bound).
        #pragma unroll
        for (int s = 0; s < KSUB; s++) {
            b_rt rB[NSUB_N];
            #pragma unroll
            for (int ni = 0; ni < NSUB_N; ni++)
                kittens::load_b_preshuffled(rB[ni], ts_B[buf][wtid][ni], s * 1024);
            #pragma unroll
            for (int r = 0; r < MTILES; r++)
                #pragma unroll
                for (int mi = 0; mi < NSUB_M; mi++) {
                    a_rt rA;
                    kittens::load(rA, kittens::subtile_view(ts_A[buf][r], warp_a_row0 + mi * 16, s * K_STEP));
                    #pragma unroll
                    for (int ni = 0; ni < NSUB_N; ni++) {
                        const kittens::fp8e8m0_4 sav{ sA[r][mi][s] }, sbv{ sB[ni][s] };
                        kittens::mma_ABt_scaled(
                            C_accum[r][mi][ni], rA, rB[ni], C_accum[r][mi][ni], &sav, &sbv);
                        __builtin_amdgcn_sched_barrier(0);   // pin WMMA vs ds_load (see hoist path)
                    }
                }
        }
#endif
    };
    // Single-buffer (HBM-latency-bound decode): load tile in place each iter, WAR barrier.
    for (int kt = 0; kt < k_tiles; kt++) {
        uint32_t sA[MTILES][NSUB_M][KSUB], sB[NSUB_N][KSUB];
        load_scales_typed(kt, sA, sB);
        kittens::sync::wait_load<0>();
        load_ktile_typed(kt, 0);
        kittens::sync::wait_tdm<0>();
        kittens::sync::sync();
        compute_ktile(0, sA, sB);
        kittens::sync::sync();                          // WAR before next iter overwrites the buffer
    }
#endif
    return true;
}

// ======================== XCD/AID-swizzle (L2 weight locality) ==================================
// gfx1250 has 2 L2 domains (AIDs), not 8 XCDs -- set MOE_XCD=2 for a domain-correct remap. The
// function is generic over nxcd, but note: measured neutral at the target shapes (see MOE_XCD note),
// since the kernel is latency-bound, not L2-locality-bound. Assumes HW assigns a workgroup's physical
// domain round-robin as `raw % nxcd` (verify on gfx1250 before relying on it). To improve GL2/L2 weight
// reuse we invert that mapping so a contiguous run of logical tiles lands on ONE XCD/AID (their
// shared weight slices then hit GL2 rather than HBM). This is a bijection over [0, total); the
// tail handling (when total % nxcd != 0) follows the AMD Triton convention where the first
// `tall_xcds` XCDs carry one extra tile. No-op when nxcd <= 1.
__device__ __host__ __forceinline__ int xcd_swizzle(int raw, int total, int nxcd) {
    if (nxcd <= 1) return raw;
    const int pids_per_xcd = (total + nxcd - 1) / nxcd;
    int tall_xcds = total % nxcd;
    tall_xcds = (tall_xcds == 0) ? nxcd : tall_xcds;
    const int xcd   = raw % nxcd;
    const int local = raw / nxcd;
    if (xcd < tall_xcds)
        return xcd * pids_per_xcd + local;
    return tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local;
}

// Map (blockIdx.x, blockIdx.y) -> (pid_m, tile_n). grid is dim3(grid_m, grid_n) so the linear id
// is raw = blockIdx.x + blockIdx.y * gridDim.x; pid_m = raw % gridDim.x, tile_n = raw / gridDim.x.
// With MOE_XCD == 1 this reduces exactly to pid_m = blockIdx.x, tile_n = blockIdx.y (no-op).
__device__ __forceinline__ void moe_tile_ids(int &pid_m, int &tile_n) {
    int raw = blockIdx.x + blockIdx.y * gridDim.x;
#if MOE_XCD > 1
    raw = xcd_swizzle(raw, gridDim.x * gridDim.y, MOE_XCD);
#endif
    pid_m  = raw % gridDim.x;
    tile_n = raw / gridDim.x;
}

// ======================== Dispatch ========================

// Always-on dispatch validation. Unlike assert(), these survive -DNDEBUG (release / pybind) builds,
// where a bad shape otherwise silently truncates in grid() via integer division -> dropped tiles or
// out-of-bounds reads. Throws std::invalid_argument (pybind surfaces it as a Python ValueError).
static inline void moe_check(bool ok, const char *msg) {
    if (!ok) throw std::invalid_argument(msg);
}
static void moe_validate_dims(int M, int N, int K, int E) {
    moe_check(M > 0 && N > 0 && K > 0 && E > 0, "moe_gemm: M,N,K,E must all be > 0");
    moe_check(M % BLOCK_M == 0, "moe_gemm: M must be a multiple of BLOCK_M");
    moe_check(N % BLOCK_N == 0, "moe_gemm: N must be a multiple of BLOCK_N");
    // K >= KTILE guards k_tiles == 0 (K < KTILE would run zero K-iterations -> silent all-zero output).
    moe_check(K % KTILE == 0 && K >= KTILE, "moe_gemm: K must be a positive multiple of KTILE (K >= KTILE)");
    moe_check((M / BLOCK_M) % MTILES == 0,
              "moe_gemm: grid_m (M/BLOCK_M) must be a multiple of MOE_MTILES; pad expt_data and the "
              "A/scale_a row count to a multiple of MTILES with -1 tiles");
}

// Validation of the data-dependent MOE_MTILES straddle contract. The device-side assert in
// moe_compute_tile is compiled out under -DNDEBUG (release / pybind), so a straddling expt_data would
// SILENTLY multiply an m-tile by the wrong expert's weights AND scales. When MTILES>1 this reads back
// expt_data once (one D->H copy) and verifies each group of MTILES consecutive m-tiles shares a single
// expert (padding -1s are the tail). No-op / zero-cost at the default MTILES==1. The D->H copy is a
// per-dispatch host/device sync, so define MOE_DISABLE_MTILES_CONTRACT_CHECK to skip it once upstream
// routing is trusted (a straddle then corrupts silently -- only disable when the routing is known-good).
static void moe_check_mtiles_contract(const int32_t *d_expt_data, int grid_m) {
#if defined(MOE_DISABLE_MTILES_CONTRACT_CHECK)
    (void)d_expt_data; (void)grid_m; return;   // trusted routing: skip the per-dispatch D->H sync
#endif
    if (MTILES <= 1 || grid_m <= 0 || d_expt_data == nullptr) return;
    std::vector<int32_t> h(grid_m);
    hipError_t e = hipMemcpy(h.data(), d_expt_data, (size_t)grid_m * sizeof(int32_t),
                             hipMemcpyDeviceToHost);
    moe_check(e == hipSuccess,
              "moe_gemm: expt_data device->host readback failed (MOE_MTILES contract check)");
    for (int grp = 0; grp * MTILES < grid_m; grp++) {
        int expert = -1;
        for (int r = 0; r < MTILES && grp * MTILES + r < grid_m; r++) {
            const int e_r = h[grp * MTILES + r];
            if (e_r < 0) continue;                 // padding tile
            if (expert < 0) expert = e_r;          // first valid expert in the group
            else moe_check(e_r == expert,
                "moe_gemm: MOE_MTILES contract violated -- an m-tile group straddles two experts; pad "
                "each expert's m-tile run to a multiple of MOE_MTILES with -1 tiles");
        }
    }
}

// ======================== Host router: batch-adaptive BLOCK_M ========================
// Pick the compile-time BLOCK_M tile for a routing density (the kernel is instantiated per tile, so
// this returns the tier the host dispatcher should launch). Over the usable ladder {16,32,48,64},
// maximize effective useful throughput = rate(bm) * (1 - per-expert padding), where d = avg
// tokens/expert. This minimizes actual padding (HK cannot mask padding for free) while preferring the
// higher-rate tile on a tie. BM128 is excluded (LDS/VGPR overflow). BM48's 1x4 warp split is the
// compute-bound peak; BM64's 2x2 split runs slower, so it stays below BM48 and is never chosen.
static inline int moe_recommend_block_m(long long num_tokens, int top_k, int E) {
    if (E < 1) E = 1;
    const long long d = (num_tokens * (top_k < 1 ? 1 : top_k) + E - 1) / E;   // ceil avg tokens/expert
    const int cands[4] = {16, 32, 48, 64};
    const double rate[4] = {0.37, 0.62, 0.82, 0.68};   // measured full-tile rate; BM64 (2x2) < BM48 (1x4)
    int best = 64; double best_eff = -1.0;
    for (int i = 0; i < 4; ++i) {
        const int bm = cands[i];
        const long long tiles = (d + bm - 1) / bm;
        const double useful = (double)d / (double)(tiles * bm);   // 1 - padding fraction
        const double eff = rate[i] * useful;                      // effective useful throughput
        if (eff > best_eff) { best_eff = eff; best = bm; }
    }
    return best;
}
