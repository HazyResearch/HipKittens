// gemm_i8_kernel.h -- the device side of the int8 GEMM: a configuration struct and a
// kernel templated on it. gemm_i8.cpp instantiates it once per tile shape and picks
// one per problem at launch.
//
// C[i][j] = sum_k A[i][k] * B[j][k]   -- A row-major MxK, B row-major NxK, int8 in,
// int32 out, through `kittens::wmma161616_acc`. The memory path is hand-written: the
// rdna4 tree has no int8 shared-tile type.
#pragma once
#include "kittens.cuh"

namespace gemm_i8_detail {

/**
 * The kernel's tuning parameters. The defaults are the shipped values.
 *
 *  BM, BN, BK      Block tile. 192x256 (BK 64) is ~3% faster than 256x256 (BK 32) where
 *                  it tiles, but needs a factor of 3 in M; powers of two use 256x256.
 *  MULTIBUF        N-deep LDS ring: the producer writes slot (t+LEAD)%NBUF while consumers
 *  NBUF, LEAD      read slot t%NBUF, with a block barrier every BAR_EVERY tiles. Requires
 *  BAR_EVERY       LEAD >= BAR_EVERY and NBUF >= LEAD + BAR_EVERY. Off = double buffer.
 *  FRAG_PIPELINE   Stops LLVM sinking the A-fragment LDS reads into the MMA nest:
 *  FRAG_MMA          0 = compiler's choice; 1 = hoist all reads above the nest
 *                    (sched_barrier); 2 = 1 ds_read then FRAG_MMA WMMAs, repeated.
 *  WARPS_M, WARPS_N  Consumer wave grid. A square per-wave tile minimises LDS reads per WMMA.
 *  PROD_WAVES      Producer waves (global -> LDS only, no WMMA). 0 = every wave does both.
 *  GROUP_M         L2 tile ordering: M-tiles walked before stepping in N.
 *  SPLIT_BARRIER   Consumers signal the end-of-tile barrier before their last substep's WMMAs
 *                  and wait at the end, so the barrier overlaps compute. -1 derives it as
 *                  !MULTIBUF; the ring barriers with __syncthreads() and cannot combine with it.
 */
template <int BM_, int BN_, int BK_, bool MULTIBUF_, int FRAG_PIPELINE_,
          int WARPS_M_ = 4, int WARPS_N_ = 4, int PROD_WAVES_ = 8, int GROUP_M_ = 6,
          int NBUF_ = 4, int LEAD_ = 2, int BAR_EVERY_ = 2, int FRAG_MMA_ = 4,
          int SPLIT_BARRIER_ = -1>
struct gemm_i8_config {
    static constexpr int  BM = BM_, BN = BN_, BK = BK_;
    static constexpr bool MULTIBUF = MULTIBUF_;
    static constexpr int  FRAG_PIPELINE = FRAG_PIPELINE_, FRAG_MMA = FRAG_MMA_;
    static constexpr int  WARPS_M = WARPS_M_, WARPS_N = WARPS_N_;
    static constexpr int  PROD_W = PROD_WAVES_, GROUP_M = GROUP_M_;
    static constexpr int  NBUF = NBUF_, LEAD = LEAD_, BAR_EVERY = BAR_EVERY_;
    static constexpr bool SPLIT_BARRIER = SPLIT_BARRIER_ < 0 ? !MULTIBUF_ : SPLIT_BARRIER_ != 0;

    static constexpr int PROD_THREADS = PROD_W * 32;
    static constexpr int NWARP   = WARPS_M * WARPS_N;
    static constexpr int NTHREAD = NWARP * 32;                 // CONSUMER threads
    static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    static constexpr int TM = WM / 16, TN = WN / 16;

    // FRAGMENT-UNIT LDS LAYOUT. A unit is 512 B = 32 lanes x 16 B holding one 16-row group
    // for TWO k-steps interleaved at 8-byte granularity: lane l = p + 16q reads bytes
    // [16l, 16l+16) and gets k-step 2u in the low 8 bytes and 2u+1 in the high 8. A fragment
    // pair is one contiguous 16-byte read, and 32 lanes covering 512 contiguous bytes is
    // bank-conflict-free by construction. No row padding: padding pushed LDS past the size
    // that fits two workgroups per CU, and 1 vs 2 workgroups/CU is worth ~14%.
    static constexpr int UNIT   = 512;
    static constexpr int NKPAIR = BK / 32;
    // A and B are sized separately: sizing both by BM silently broke BN != BM three ways.
    static constexpr int NGRP_A    = BM / 16;
    static constexpr int NGRP_B    = BN / 16;
    static constexpr int TILE_A_SZ = NGRP_A * NKPAIR * UNIT;
    static constexpr int TILE_B_SZ = NGRP_B * NKPAIR * UNIT;
    static constexpr int STAGE_SZ  = TILE_A_SZ + TILE_B_SZ;   // one buffer: A then B
    static constexpr int LDS_TOTAL = (MULTIBUF ? NBUF : 2) * STAGE_SZ;

    // Global -> LDS map: each thread moves 16 bytes of a BK-byte row. Sized by the staging
    // pool (the producers), not the workgroup.
    static constexpr int THREADS_PER_ROW = BK / 16;
    static constexpr int STAGE_THREADS   = (PROD_W > 0) ? PROD_THREADS : NTHREAD;
    static constexpr int ROWS_PER_PASS   = STAGE_THREADS / THREADS_PER_ROW;
    static constexpr int PASSES_A = BM / ROWS_PER_PASS;
    static constexpr int PASSES_B = BN / ROWS_PER_PASS;

    static_assert(LDS_TOTAL <= 65536, "exceeds the 64 KB per-workgroup LDS cap");
    static_assert(BM % ROWS_PER_PASS == 0, "BM must be a multiple of ROWS_PER_PASS");
    static_assert(BN % ROWS_PER_PASS == 0, "BN must be a multiple of ROWS_PER_PASS");
    // With PROD_W == 0 every wave is BOTH producer and consumer, so under SPLIT_BARRIER it
    // would signal twice per wait and silently corrupt the result.
    static_assert(!(SPLIT_BARRIER && PROD_W == 0),
                  "SPLIT_BARRIER signals once per role; PROD_WAVES=0 makes every wave "
                  "both roles, so it signals twice per wait. Use PROD_WAVES>0 or "
                  "SPLIT_BARRIER=0.");
    static_assert(!MULTIBUF || LEAD >= BAR_EVERY,
                  "producer must lead by at least one barrier period");
    static_assert(!MULTIBUF || NBUF >= LEAD + BAR_EVERY,
                  "ring too small: reader and writer ranges alias");
    // MULTIBUF barriers with __syncthreads() while a split consumer signal still fires.
    static_assert(!(MULTIBUF && SPLIT_BARRIER),
                  "MULTIBUF is incompatible with SPLIT_BARRIER at any BAR_EVERY: the ring "
                  "barriers with __syncthreads() while the consumer still emits a split "
                  "signal, so waves signal twice per wait. Use SPLIT_BARRIER=0.");

    __device__ __forceinline__ static int unit_off(int g, int u) { return (g * NKPAIR + u) * UNIT; }
    __device__ __forceinline__ static int8_t* tileA(int8_t* lds, int buf) { return lds + buf * STAGE_SZ; }
    __device__ __forceinline__ static int8_t* tileB(int8_t* lds, int buf) { return lds + buf * STAGE_SZ + TILE_A_SZ; }
};

}  // namespace gemm_i8_detail

using gemm_i8_detail::gemm_i8_config;

template <typename C>
__global__ __launch_bounds__(C::NTHREAD + C::PROD_THREADS)
void gemm_i8(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
             int* __restrict__ Cout, int M, int N, int K)
{
    using namespace kittens;
    extern __shared__ int8_t lds[];
    constexpr int BM = C::BM, BN = C::BN, BK = C::BK;
    constexpr int TM = C::TM, TN = C::TN, WM = C::WM, WN = C::WN;
    constexpr int PROD_W = C::PROD_W, NKPAIR = C::NKPAIR;
    constexpr int ROWS_PER_PASS = C::ROWS_PER_PASS, THREADS_PER_ROW = C::THREADS_PER_ROW;
    constexpr int PASSES_A = C::PASSES_A, PASSES_B = C::PASSES_B;

    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    // Producers are the LOW waves, so a producer's tid is already its index into the
    // staging pool. Consumers renumber from zero for the wave grid.
    const bool is_prod = (PROD_W == 0) || (wid < PROD_W);
    const bool is_cons = (PROD_W == 0) || (wid >= PROD_W);
    // Producers must NOT fall through into the MMA loop or the epilogue: their cwid would
    // be negative and the C store would write out of bounds. They do still execute every
    // barrier, which is what keeps a whole-block barrier legal under this divergence.
    const int  cwid    = (PROD_W == 0) ? wid : (wid - PROD_W);

    // Row-major wave map: a SIMD's 4 consumers get constant wc, so they hold 4 distinct A
    // sets and 1 B set.
    const int wr = cwid / C::WARPS_N, wc = cwid % C::WARPS_N;
    // L2 tile ordering (Triton's grouped order): walk GROUP_M tiles down M before
    // stepping in N, so both operands' working sets stay live in L2.
    const int num_pid_m = gridDim.y;
    const int num_pid_n = gridDim.x;
    const int pid       = blockIdx.x + blockIdx.y * gridDim.x;
    const int num_pid_in_group = C::GROUP_M * num_pid_n;
    const int group_id         = pid / num_pid_in_group;
    const int first_pid_m      = group_id * C::GROUP_M;
    const int group_size_m     = min(num_pid_m - first_pid_m, C::GROUP_M);
    const int m0 = (first_pid_m + (pid % group_size_m)) * BM;
    const int n0 = ((pid % num_pid_in_group) / group_size_m) * BN;

    const int lr = tid / THREADS_PER_ROW, lc = (tid % THREADS_PER_ROW) * 16;

    // Producer staging registers: fetch() issues a tile's global loads into these,
    // commit() writes them to LDS.
    longlong2 rega[PASSES_A], regb[PASSES_B];

    auto fetch = [&](int k0) {
#pragma unroll
        for (int q2 = 0; q2 < PASSES_A; ++q2) {
            const int r = lr + q2 * ROWS_PER_PASS;
            rega[q2] = *reinterpret_cast<const longlong2*>(A + (long)(m0 + r) * K + k0 + lc);
        }
#pragma unroll
        for (int q2 = 0; q2 < PASSES_B; ++q2) {
            const int r = lr + q2 * ROWS_PER_PASS;
            regb[q2] = *reinterpret_cast<const longlong2*>(B + (long)(n0 + r) * K + k0 + lc);
        }
    };
    // A thread's 16 global bytes are one whole k-step of one row; in the unit layout they
    // split across lanes p and p+16, 256 B apart, so staging is two 8-byte stores.
    auto commit = [&](int buf) {
        int8_t* ta = C::tileA(lds, buf);
        int8_t* tb = C::tileB(lds, buf);
        const int kstep = lc >> 4, u = kstep >> 1, h = kstep & 1;
#pragma unroll
        for (int q2 = 0; q2 < PASSES_A; ++q2) {
            const int r = lr + q2 * ROWS_PER_PASS;
            const int g = r >> 4, pp = r & 15;
            const int lo = C::unit_off(g, u) + 16 * pp        + 8 * h;
            const int hi = C::unit_off(g, u) + 16 * (pp + 16) + 8 * h;
            *reinterpret_cast<long*>(ta + lo) = rega[q2].x;
            *reinterpret_cast<long*>(ta + hi) = rega[q2].y;
        }
#pragma unroll
        for (int q2 = 0; q2 < PASSES_B; ++q2) {
            const int r = lr + q2 * ROWS_PER_PASS;
            const int g = r >> 4, pp = r & 15;
            const int lo = C::unit_off(g, u) + 16 * pp        + 8 * h;
            const int hi = C::unit_off(g, u) + 16 * (pp + 16) + 8 * h;
            *reinterpret_cast<long*>(tb + lo) = regb[q2].x;
            *reinterpret_cast<long*>(tb + hi) = regb[q2].y;
        }
    };

    int2 acc[TM][TN][4];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = base_types::constants<int2>::zero();

    if constexpr (C::MULTIBUF) {
        if (is_prod) {
#pragma unroll
            for (int i = 0; i < C::LEAD; ++i) { fetch(i * BK); commit(i % C::NBUF); }
        }
    } else {
        if (is_prod) { fetch(0); commit(0); }
    }
    __syncthreads();

    const int ntile = K / BK;
    for (int t = 0; t < ntile; ++t) {
        const int buf = C::MULTIBUF ? (t % C::NBUF) : (t & 1);
        if constexpr (C::MULTIBUF) {
            if (is_prod && t + C::LEAD < ntile) fetch((t + C::LEAD) * BK);
        } else {
            if (is_prod && t + 1 < ntile) fetch((t + 1) * BK);
        }

        const int8_t* ta = C::tileA(lds, buf);
        const int8_t* tb = C::tileB(lds, buf);
        if (is_cons) {

        // Fragment map, the same one the bf16 path uses: for a 16x16 tile, lane l holds
        // row (l%16), k-elements [8*(l/16) .. +8).
        const int p = lane & 15, q = lane >> 4;
        // One 16-byte read yields BOTH k-steps of a pair: .x/.y are k-step 2u, .z/.w 2u+1.
#pragma unroll
        for (int us = 0; us < NKPAIR; ++us) {
            const int u = us;
            longlong2 av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = wr * WM + i * 16 + p;
                av[i] = *reinterpret_cast<const longlong2*>(
                            ta + C::unit_off(row >> 4, u) + 16 * ((row & 15) + 16 * q));
            }
            // B stays in LDS: it is K-major, so a fragment's two k-steps sit 8 bytes apart in
            // global and would need two 8-byte loads per fragment.
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int row = wc * WN + j * 16 + p;
                bv[j] = *reinterpret_cast<const longlong2*>(
                            tb + C::unit_off(row >> 4, u) + 16 * ((row & 15) + 16 * q));
            }
            if constexpr (C::SPLIT_BARRIER) {
                // All of this tile's LDS reads have landed by here (the MMA nest below
                // forces dscnt=0 anyway), so buf may be released now. The WMMAs that
                // follow then run underneath the barrier.
                if (us == NKPAIR - 1) {
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
                    __builtin_amdgcn_s_barrier_signal(-1);
                }
            }
            // FRAG_PIPELINE: stop LLVM streaming every A fragment through one recycled
            // register quad (a WAR serialisation that forces a full s_wait_dscnt drain per
            // fragment). 1 = hoist all loads above the MMA nest; 2 = 1 ds_read then
            // FRAG_MMA WMMAs, repeated (sched_group_barrier masks 0x100 DS read, 0x008 WMMA).
            if constexpr (C::FRAG_PIPELINE == 1) {
                __builtin_amdgcn_sched_barrier(0);
            } else if constexpr (C::FRAG_PIPELINE == 2) {
#pragma unroll
                for (int g = 0; g < (2 * TM * TN) / C::FRAG_MMA; ++g) {
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);           // 1 ds_read
                    __builtin_amdgcn_sched_group_barrier(0x008, C::FRAG_MMA, 0); // then N WMMAs
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {          // the two k-steps in the pair
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const int8_4* aa = reinterpret_cast<const int8_4*>(&av[i]) + 2 * h;
                        const int8_4* bb = reinterpret_cast<const int8_4*>(&bv[j]) + 2 * h;
                        kittens::wmma161616_acc(acc[i][j],
                            *reinterpret_cast<const int8_4(*)[2]>(aa),
                            *reinterpret_cast<const int8_4(*)[2]>(bb));
                    }
            }
        }
        }  // is_cons
        // buf^1 was last READ in iteration t-1, behind that iteration's barrier, so
        // committing to it here needs no extra barrier of its own.
        if constexpr (C::MULTIBUF) {
            if (is_prod && t + C::LEAD < ntile) commit((t + C::LEAD) % C::NBUF);
        } else {
            if (is_prod && t + 1 < ntile) commit(buf ^ 1);
        }
        if constexpr (C::MULTIBUF) {
            if ((t % C::BAR_EVERY) == C::BAR_EVERY - 1) __syncthreads();
        } else if constexpr (C::SPLIT_BARRIER) {
            if (is_prod) {                       // consumers already signalled above
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
                __builtin_amdgcn_s_barrier_signal(-1);
            }
            __builtin_amdgcn_s_barrier_wait(-1);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
        } else {
            __syncthreads();
        }
    }

    // Accumulator -> C. Element el of the 8 is row (8*(lane/16) + el), column (lane%16),
    // within the 16x16 tile. Producers own no output tile.
    if (!is_cons) return;
    const int p = lane & 15, q = lane >> 4;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int* a8 = reinterpret_cast<const int*>(acc[i][j]);
#pragma unroll
            for (int el = 0; el < 8; ++el) {
                const int r = m0 + wr * WM + i * 16 + 8 * q + el;
                const int c = n0 + wc * WN + j * 16 + p;
                Cout[(long)r * N + c] = a8[el];
            }
        }
}
