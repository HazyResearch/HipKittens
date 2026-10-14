// Single-GPU SDPA / flash-attention forward for gfx1100 (RDNA3), bf16 in,
// fp32 accumulate. Head dim 64 or 128, MHA or GQA, causal or not, forward only.
//
// The shape of this kernel is not a port of the CDNA4 one; it is what the
// gfx11 register layouts leave you with.  Three facts drive every decision:
//
//   * The WMMA accumulator is `col` layout, and a *reduction along the element
//     axis* is register-local while one along the lane axis is a four-step
//     butterfly (see the table at the top of reductions.cuh).  For a col-layout
//     accumulator the element axis is the *row* axis.  So if the score matrix is
//     stored transposed -- S^T, indexed [kv, q] -- then the online softmax's max
//     and sum over kv land on the cheap axis, and the running statistics are one
//     VGPR each.
//
//   * Only `row`-layout bf16 operand tiles take the vectorized ds_read_b128 path
//     out of LDS (shared_to_register.cuh).  A `col`-layout one degrades to 16
//     scalar ds_read_u16 per base tile.  So every operand that comes out of LDS
//     must be row layout, which fixes which mma_* primitive each matmul uses.
//
//   * A row/col operand pair is bit-identical on gfx11 (the note above
//     mma_ABt_base), so choosing between mma_AB / mma_ABt / mma_AtBt is free --
//     they all lower to the same v_wmma_f32_16x16x16_bf16.  The choice is purely
//     about which layout tag the tiles get to carry.
//
// Which gives:
//
//     S^T[kv,q]  = K . Q^T      mma_ABt(s_t, k, q)     k row (LDS) ✓  q row ✓
//     O^T[d,q]  += V^T . P^T    mma_AB (o_t, vt, p_t)  vt row (LDS) ✓  p_t col
//
// O is accumulated transposed for the same reason S is: the rescale factor and
// the final 1/l are per-q, and on O^T[d,q] "per-q" is per-column, which takes
// the same ortho row-vector the softmax already produced.  Keeping O as [q,d]
// would need the same statistic in the *other* vector layout and a conversion
// between them every KV block.  The one transpose this costs is paid once per
// Q block in the epilogue, not once per KV block.
//
// V^T is what the second matmul needs and V arrives [n, d], so V is transposed
// on the way into LDS -- see stage_vt() below.  There is no way around a
// transpose somewhere: the PV matmul's B operand must have kv on the element
// axis, and a vectorized LDS read always puts the tile's row index on the lane
// axis.  Doing it here means it is paid once per KV block per workgroup rather
// than once per KV block per warp.

#include <utility>

#include "kittens.cuh"
#ifndef HK_ATTN_NO_PYBIND
#include "pyutils/pyutils.cuh"
#endif
using namespace kittens;

// Tiling knobs. sweep.sh overrides these on the command line.
#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
// Query rows per warp. Each warp owns its own Q block and its own accumulator;
// K and V^T are shared by the whole workgroup.
#ifndef Q_BLOCK
#define Q_BLOCK 16
#endif
#ifndef KV_BLOCK
#define KV_BLOCK 32
#endif
// Warps per workgroup, which is also the Q tile: Q_TILE = Q_BLOCK * NUM_WARPS,
// and every warp in the workgroup reads the same staged K and V^T. So this is
// the knob that amortizes staging, which the ablation table puts at 18% of the
// runtime -- the global K/V traffic is (N / Q_TILE) passes over the whole of K
// and V, so a wider workgroup divides it down.
//
// The measured shape of this knob is not "wider is better", it is a comb:
//
//     warps    8    10    12    14    16    24
//     TF    56.8  51.2  59.7  43.5  49.5  56.2   (N=16384, cross-process)
//
// At 221 VGPRs the occupancy limit is 6 waves/SIMD, i.e. 24 waves per CU, and
// what matters is whether NUM_WARPS divides that. 8 and 12 do (3 and 2
// workgroups per CU); 10, 14 and 16 leave 4, 10 and 8 wave slots stranded, and
// the loss swamps the staging they save. Among the divisors, 12 wins because it
// stages half as often as 8. 24 divides it too but leaves one workgroup per CU,
// with nothing to cover its barriers.
//
// That comb was measured one build per process, which ab.py exists to say is
// not comparable; the ranking survived a single-process rerun but the spreads
// did not. Interleaved (ab.py, N=4096/16384 non-causal):
//
//     warps 12       59.93  62.52      <- shipped
//     warps 12 +DBUF 59.90  62.41
//     warps 24       59.69  61.82
//     warps 24 +DBUF 60.99  63.01      <- +1.8% / +0.8%
//
// So the whole Q_TILE lever is worth about 1%, not the 9% that halving the
// staged bytes should buy if staging's 18% ablation share were recoverable
// time. It is not: at 12 warps there are two workgroups per WGP and the
// co-resident one already covers staging. 24 warps buys the amortization but
// gives up the cover (one workgroup per WGP), and DBUF only claws back the
// barrier half of what that costs. Not shipped: it also doubles Q_TILE to 384,
// which the drop-in would have to fall back below.
//
// The cost is coverage at the bottom end: Q_TILE is 192 rather than 128, so the
// kernel needs N >= 192 and the drop-in falls back below that. For H3, where N
// is tens of thousands, that is free.
#ifndef NUM_WARPS
#define NUM_WARPS 12
#endif
// Rows of V^T (i.e. columns of V, head-dim elements) one warp transposes at a
// time while staging. The global reads and the LDS writes total the same
// whatever this is; what changes is the transient (2 * VT_D_CHUNK VGPRs for the
// untransposed and transposed copies) and how many warps have work. At
// HEAD_DIM=128 this gives 8 bands, so with 12 warps the last 4 idle through the
// V^T staging; VT_D_CHUNK=32 spreads it differently and measures slower (58.7
// against 59.7), because the wider transpose costs more than the idle warps do.
#ifndef VT_D_CHUNK
#define VT_D_CHUNK 16
#endif
// How many 16x16 operand fragments of K (resp. V^T) one LDS read brings in.
//
// These are pure register-pressure knobs and they are the ones that decide
// whether the kernel spills. The whole K slice a warp needs is
// KV_BLOCK/16 * HEAD_DIM/16 fragments = 32 at the default shape = 256 VGPRs, so
// it was never going to be resident; the only question is how much of it is.
// Larger batches more ds_read_b128 before the first s_waitcnt, which is what
// hides LDS latency under the WMMAs; smaller leaves room for Q and the
// accumulator, which are 128 VGPRs between them and cannot move.
//
// PV_TILES was 4 and is now 2, which is worth 5% on causal and 2% non-causal.
// (The first measurement of this said 13-16% on causal. That was one build per
// process -- see ab.py. Interleaved against git HEAD in a single process it is
// 5%.) The VGPR allocation granule here is 24, so occupancy 6 needs
// <= 240 VGPRs; PV_TILES=4 put the *causal* instantiation at 252 and dropped it
// to 5 waves/SIMD, while the non-causal one squeaked in at 238. That, and not
// the staging overrun this file used to blame, is most of why causal was
// slower than the Triton baseline. PV_TILES=1 measures the same as 2 (both 227
// VGPRs), so 2 is the knee: it is the deepest LDS read batch that still fits.
#ifndef QK_TILES
#define QK_TILES 2
#endif
#ifndef PV_TILES
#define PV_TILES 2
#endif
#ifndef MIN_BLOCKS_PER_CU
#define MIN_BLOCKS_PER_CU 1
#endif

// Prefetch the next KV block's global reads into registers, one KV block ahead.
//
// gfx11 has no global->LDS DMA, so every staged byte passes through a VGPR
// anyway; the only question is *when* the LDS write happens relative to the
// math. Without this the sequence is global -> (stall on vmcnt) -> LDS -> wait
// -> barrier -> math, and the global latency is exposed: staging measures 27.7%
// of the runtime and none of it overlaps anything. With it, block kb+1's
// buffer_loads are issued immediately after kb's LDS commit and land while kb's
// WMMAs run; the vmcnt wait is paid at the top of kb+1, by which time the data
// is there.
//
// This costs registers that stay live across the whole math body -- stage_calls
// float4s for K plus one bf16 fragment per V band -- and the kernel is at 217 of
// 256 VGPRs, so it is a knob and not a given. Build with GPREFETCH=0 to compare.
// Same pattern as the GEMM's GPREFETCH; see the note above gload() there.
#ifndef GPREFETCH
#define GPREFETCH 1
#endif

// Double-buffer the K and V^T tiles in LDS, so that block kb+1 is written into
// one buffer while block kb's math reads the other.
//
// GPREFETCH alone only hides the *global* latency; it measured +1.5%, because
// the rest of staging -- the transposes and the ds_writes -- still sat alone
// between two barriers with nothing to overlap. The barriers are the real cost:
// with a single buffer the write has to wait for every wave to finish reading,
// so all NUM_WARPS waves are lockstepped twice per KV block and a SIMD has
// nothing to switch to while one of its waves stages.
//
// With two buffers the commit targets a buffer nobody is reading and the
// mid-block barrier disappears -- one barrier per KV block instead of two.
// Nothing is reordered within a wave (every LDS access here is volatile asm);
// the win is that waves are free to drift apart, so one wave's ds_writes issue
// under another wave's WMMAs. It also gives the prefetched global reads a full
// iteration to land instead of a partial one.
//
// Costs 16 KB more LDS and ~16 v_xor per KV block. Build with DBUF=0 to compare.
//
// DBUF and GPREFETCH are independent, and at HEAD_DIM=128 they do not fit
// together: holding buf_k and v_rows live across the math is ~21 VGPRs on a
// kernel already at 238 of 256, and turning both on spills 383 of them (which on
// a kernel that hand-manages s_waitcnt is wrong, not slow). They also overlap in
// what they buy. GPREFETCH exists to cover the global latency *within* a wave,
// because with a single buffer the wave in front of a barrier has nothing else
// to do; under DBUF there is no barrier between the math and the staging, so the
// SIMD covers that latency with another wave's WMMAs and the registers are
// better spent elsewhere. DBUF=1 GPREFETCH=0 is the shipped combination.
//
// Measured, and off by default: it is worth +1.5-2% causal and -0.5%
// non-causal, which does not pay for 16 KB of LDS and the buffer-parity
// bookkeeping. The reason it disappoints is the finding itself -- staging is not
// latency-bound or barrier-bound, it is issue-bound. The transposes and the
// ds_writes are real work competing with the WMMAs for the same SIMD, and
// overlapping them better cannot remove them. Kept because the measurement is
// the useful part, and because it is the right structure if staging ever gets
// cheaper (a wider KV_BLOCK, or an architecture with global->LDS DMA).
#ifndef DBUF
#define DBUF 0
#endif

// Skip the accumulator rescale on KV blocks that did not raise any row's max.
//
// col_max writes m_new = max(m_old, block_max), so a block that raises nothing
// leaves m_new *bitwise* identical to m_old, alpha = exp2(0) = 1 exactly, and
// both mul(l_run, alpha) and mul_col(o_t, alpha) are exact no-ops. The test is
// an equality, not a tolerance, so this changes no bit of the output -- unlike
// the CDNA4 kernel's RESCALE_THRESHOLD, which skips when alpha is merely close
// to 1 and is therefore an approximation.
//
// Worth doing because mul_col(o_t, ...) is the single largest VALU item in the
// inner loop: o_t is HEAD_DIM/16 = 8 fp32 col base tiles of 8 registers, so 64
// v_mul_f32 against the block's 32 WMMAs. After the first few KV blocks the
// running max is usually already the global max and every one of those 64 is
// wasted.
#ifndef RESCALE_SKIP
#define RESCALE_SKIP 1
#endif

// Longest-processing-time-first dispatch order for the causal grid. See the
// note at q_block in micro_tk.
#ifndef CAUSAL_LPT
#define CAUSAL_LPT 1
#endif

// Ablation switches. Each makes the result wrong on purpose; they exist to time
// one stage at a time. See sweep.sh.
#ifndef ABLATE_STAGE
#define ABLATE_STAGE 0
#endif
#ifndef ABLATE_QK
#define ABLATE_QK 0
#endif
#ifndef ABLATE_SOFTMAX
#define ABLATE_SOFTMAX 0
#endif
#ifndef ABLATE_PV
#define ABLATE_PV 0
#endif
// Keep both matmuls and drop the LDS reads that feed them: the operand tiles are
// zeroed once and then reused for every fragment. This is the one ablation that
// separates "LDS-bound" from "WMMA-bound", which no combination of the stage
// switches above can do -- each of those removes a stage's reads and its math
// together.
#ifndef ABLATE_LDS_READ
#define ABLATE_LDS_READ 0
#endif

// Debug ladder. Each level writes one intermediate into the output tensor and
// returns, so dump.py can compare it against a numpy model of that one stage.
// 0 = off (the real kernel).
//   1  K as it reads back out of LDS   -> o[b,h,  0:KV_BLOCK, 0:HEAD_DIM]
//   2  V^T as it reads back out of LDS -> o[b,h, 0:HEAD_DIM, 0:KV_BLOCK]
//   3  S^T after the QK matmul, transposed back to [q,kv]
//                                      -> o[b,h,   0:Q_TILE, 0:KV_BLOCK]
//   4  P^T after the online softmax, transposed back to [q,kv]  (same region)
//   9  S^T after the log2(e)*scale fold                         (same region)
//
// Levels 10 and 11 are a different shape: warp 0 writes its 16 raw accumulator
// VGPRs to o[lane, slot], plus skip/kv_start/N, with no transpose and no store
// path in between -- 10 before the scale fold, 11 after.  Use these when a level
// that goes through store() looks right but something downstream does not: the
// store path reads the same registers, so it hides a scheduling race in a way a
// per-lane dump does not.  Finding the lds_wait/WMMA reordering needed exactly
// this.
#ifndef HK_DUMP
#define HK_DUMP 0
#endif

// A constexpr for loop, so the loop index can be a template argument.
template<int N, typename F> __device__ inline void static_for(F &&f) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (f(std::integral_constant<int, I>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

// The LDS addressing for the two hot matmuls (lds_granules / lds_read_frag /
// lds_write_frag) now lives in include/rdna3/ops/warp/memory/util/granule.cuh.
// It was found here; it is in the library because the kernels hk-dsl
// generates need it for exactly the same reason.
using kittens::lds_granules;
using kittens::lds_read_frag;
using kittens::lds_write_frag;

using _gl_q = gl<bf16, -1, -1, -1, -1>;   // (b, h, n, d)
using _gl_k = gl<bf16, -1, -1, -1, -1>;
using _gl_v = gl<bf16, -1, -1, -1, -1>;
using _gl_o = gl<bf16, -1, -1, -1, -1>;

struct micro_globals {
    _gl_q q;
    _gl_k k;
    _gl_v v;
    _gl_o o;
    // Softmax scale, *before* the log2(e) fold. 0 means "use 1/sqrt(d)", which
    // is what every caller in this tree wants and what torch defaults to.
    float scale = 0.f;
    // Causal masking. Only meaningful when k and q have the same sequence
    // length, which is what the dispatcher enforces -- with n_kv != n_q there
    // are two incompatible conventions for where the diagonal sits, and this
    // kernel implements neither.
    bool causal = false;
    hipStream_t stream = nullptr;
};

namespace cli {
constexpr int head_dim = HEAD_DIM, q_block = Q_BLOCK, kv_block = KV_BLOCK,
              num_warps = NUM_WARPS, vt_d_chunk = VT_D_CHUNK,
              qk_tiles = QK_TILES, pv_tiles = PV_TILES;
}
#undef HEAD_DIM
#undef Q_BLOCK
#undef KV_BLOCK
#undef NUM_WARPS
#undef VT_D_CHUNK
#undef QK_TILES
#undef PV_TILES

template<int _HEAD_DIM, int _Q_BLOCK, int _KV_BLOCK, int _NUM_WARPS,
         int _VT_D_CHUNK = 32, int _QK_TILES = 2, int _PV_TILES = 2>
struct config {
    static constexpr int HEAD_DIM = _HEAD_DIM, Q_BLOCK = _Q_BLOCK;
    static constexpr int KV_BLOCK = _KV_BLOCK, NUM_WARPS = _NUM_WARPS;
    static constexpr int VT_D_CHUNK = _VT_D_CHUNK;
    static constexpr int QK_TILES = _QK_TILES, PV_TILES = _PV_TILES;
    static_assert(KV_BLOCK % (16 * QK_TILES) == 0,
                  "the K operand window must tile the KV block");
    static_assert(HEAD_DIM % (16 * PV_TILES) == 0,
                  "the V^T operand window must tile the head dimension");

    static constexpr int NUM_THREADS = kittens::WARP_THREADS * NUM_WARPS;
    // Query rows one workgroup covers.
    static constexpr int Q_TILE = Q_BLOCK * NUM_WARPS;

    static_assert(HEAD_DIM % 16 == 0 && Q_BLOCK % 16 == 0 && KV_BLOCK % 16 == 0,
                  "every tile extent is in units of a 16x16 WMMA fragment");
    static_assert(VT_D_CHUNK % 16 == 0 && HEAD_DIM % VT_D_CHUNK == 0,
                  "the V^T staging chunk must tile the head dimension");

    using st_k  = st_bf<KV_BLOCK, HEAD_DIM>;
    using st_vt = st_bf<HEAD_DIM, KV_BLOCK>;
    using G     = kittens::group<NUM_WARPS>;

    // At KV_BLOCK=32, HEAD_DIM=128 each of these is exactly 8 KB, so double
    // buffering is 32 KB and two workgroups per WGP is 64 KB -- the whole
    // budget, and exactly the occupancy the 238-VGPR register footprint allows
    // anyway (6 waves/SIMD = 24 waves = 2 workgroups of 12). It is free. An
    // earlier version of this comment ruled double buffering out as "64 KB";
    // that was written when KV_BLOCK was 64 and both tiles were 16 KB.
    static constexpr int LDS_STAGES = DBUF ? 2 : 1;
    static constexpr size_t SHARED_BYTES = LDS_STAGES * (sizeof(st_k) + sizeof(st_vt));
    static_assert(SHARED_BYTES <= MAX_SHARED_MEMORY, "block does not fit in 64KB of LDS");

    // The two buffers are swapped by XOR-ing one bit into every LDS address the
    // kernel holds, rather than by re-deriving them. That needs the two tiles to
    // be the same power-of-two size and to be allocated in the order k0, k1, v0,
    // v1 from an aligned base, so that buffer 1 of each pair differs from buffer
    // 0 in exactly the sizeof(st_k) bit. Both hold for every config in this file;
    // the static_asserts are here so that a config where they do not hold fails
    // to compile instead of silently reading the wrong half.
    static constexpr uint32_t LDS_XOR = DBUF ? (uint32_t)sizeof(st_k) : 0u;
    static_assert(!DBUF || sizeof(st_k) == sizeof(st_vt),
                  "the XOR buffer swap needs K and V^T to be the same size");
    static_assert(!DBUF || (sizeof(st_k) & (sizeof(st_k) - 1)) == 0,
                  "the XOR buffer swap needs a power-of-two tile");
};

// The tiling knobs are shared across head dimensions; only HEAD_DIM varies, and
// it is the one parameter that has to be a compile-time constant in every tile
// type in the kernel. D=64 is not a separate code path, it is this template at a
// different width -- and it is *cheaper* per byte, because q and o_t are half
// the registers, which is the whole reason the D=128 shape is the tight one.
template<int D> using cfg_for = config<D, cli::q_block, cli::kv_block,
                                       cli::num_warps, cli::vt_d_chunk,
                                       cli::qk_tiles, cli::pv_tiles>;
using macro_config = cfg_for<cli::head_dim>;

// CAUSAL is a template parameter rather than a branch on g.causal so that the
// non-causal path -- the one H3 runs -- keeps exactly the codegen it had before
// causal existed. The masking below is a handful of VALU per KV block, but the
// block-count bound it enables changes the loop trip count, and a runtime
// causal flag would put that behind a register the scheduler has to respect.
template<typename C, bool CAUSAL, typename GL = micro_globals>
__global__ __launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_CU)
void micro_tk(const GL g) {
    constexpr int HEAD_DIM = C::HEAD_DIM, Q_BLOCK = C::Q_BLOCK;
    constexpr int KV_BLOCK = C::KV_BLOCK, Q_TILE = C::Q_TILE;
    constexpr int VT_D_CHUNK = C::VT_D_CHUNK;
    constexpr int NUM_WARPS = C::NUM_WARPS;
    constexpr int QK_TILES = C::QK_TILES, PV_TILES = C::PV_TILES;
    using st_k  = typename C::st_k;
    using st_vt = typename C::st_vt;
    using G     = typename C::G;

    constexpr int LDS_STAGES = C::LDS_STAGES;
    constexpr uint32_t LDS_XOR = C::LDS_XOR;

    // Allocated k0, k1, v0, v1 so that buffer 1 of each pair is sizeof(st_k)
    // above buffer 0 -- which is what makes the XOR swap in lds_granules::swap
    // legal. With LDS_STAGES == 1 the two entries of each array are the same
    // tile and every swap below is a no-op.
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // Named references, not an array of pointers. A two-element pointer array
    // here -- captured by reference into the staging lambdas -- defeats SROA, and
    // the array lands in scratch: 508 bytes/lane and 383 spilled VGPRs, identical
    // for every QK_TILES/PV_TILES setting, which is how it was identified as
    // structural rather than pressure. On a kernel that hand-manages s_waitcnt a
    // spill is wrong, not slow.
    st_k  &k_smem0  = al.template allocate<st_k>();
#if DBUF
    st_k  &k_smem1  = al.template allocate<st_k>();
#else
    st_k  &k_smem1  = k_smem0;
#endif
    st_vt &vt_smem0 = al.template allocate<st_vt>();
#if DBUF
    st_vt &vt_smem1 = al.template allocate<st_vt>();
#else
    st_vt &vt_smem1 = vt_smem0;
#endif
    (void)&vt_smem1;

    const int N       = g.q.rows();
    const int warp_id = kittens::warpid();
    const int lane    = kittens::laneid();

    const int batch = blockIdx.z;
    const int head  = blockIdx.y;
    // GQA is an addressing change and nothing else: the grid is over *query*
    // heads, and several of them read the same K/V head. One scalar divide per
    // workgroup, hoisted out of the KV loop, against thousands of trips through
    // it -- so there is no MHA/GQA specialization, just this line. When the head
    // counts match it is a divide by 1.
    const int head_kv = head / (g.q.depth() / g.k.depth());

    // Q remainder, by backing the last block up rather than predicating it. Each
    // query row's output depends on nothing but that row, so the rows in the
    // overlap are computed twice from identical inputs and written twice with
    // identical values -- idempotent, not a race. The same trick is in the GEMM;
    // see the long note there.
    // Under CAUSAL the work per workgroup is not uniform: workgroup i does i+1
    // KV blocks, so the dispatch is a ramp. Workgroups are handed to WGPs
    // roughly in index order, which means the *last* ones dispatched are the
    // longest and the machine drains with a tail of them. Reversing the q index
    // is longest-processing-time-first: the long workgroups start while there is
    // still short work left to fill the slots they free. It is an index
    // permutation within one head, so it does not disturb the grid's K/V
    // locality (x is still the fastest-varying axis; see the launch).
#if CAUSAL_LPT
    const int q_block      = CAUSAL ? (gridDim.x - 1 - (int)blockIdx.x) : (int)blockIdx.x;
#else
    const int q_block      = (int)blockIdx.x;
#endif
    const int q_tile_start = min(q_block * Q_TILE, N - Q_TILE);
    const int q_row        = q_tile_start + warp_id * Q_BLOCK;

    // exp2 rather than exp: one hardware instruction. log2(e) is folded into the
    // softmax scale, and the scale is applied to S rather than to Q, because Q is
    // bf16 and multiplying it by a non-power-of-two would round the inputs a
    // second time. Applying it to the fp32 scores costs KV_BLOCK/16 * 4 packed
    // multiplies per KV block against ~64 WMMAs.
    const float scale_l2e =
        (g.scale > 0.f ? g.scale : rsqrtf((float)HEAD_DIM)) * 1.44269504088896340736f;

    rt_bf<Q_BLOCK, HEAD_DIM, row_l> q;
    load(q, g.q, coord<>{batch, head, q_row, 0});

#if HK_DUMP == 7
    // Identity: global -> registers -> global, no LDS at all. If this is wrong
    // nothing downstream can be right, and the fault is in the gl binding or in
    // the register<->global path, not in anything this kernel does.
    store(g.o, q, coord<>{batch, head, q_row, 0});
    return;
#endif
#if HK_DUMP == 8
    // Same, but reading K the way the stager does, one 16-row tile per warp.
    {
        rt_bf<16, HEAD_DIM, row_l> dbg;
        load(dbg, g.k, coord<>{batch, head_kv, 16 * warp_id, 0});
        store(g.o, dbg, coord<>{batch, head, 16 * warp_id, 0});
    }
    return;
#endif

    // The accumulator is O^T: [d, q]. See the header.
    rt_fl<HEAD_DIM, Q_BLOCK, col_l> o_t;
    zero(o_t);

    rt_fl<KV_BLOCK, Q_BLOCK, col_l> s_t;

    // The running max and sum are indexed by q, which is the *column* axis of
    // both s_t and o_t, so both are the same `row_vec` type -- one VGPR each for
    // Q_BLOCK=16, and directly consumable by sub_col / mul_col / div_col with no
    // layout conversion anywhere. This is the whole reason S and O are stored
    // transposed.
    using rv_t = typename decltype(s_t)::row_vec;
    static_assert(std::is_same_v<rv_t, typename decltype(o_t)::row_vec>,
                  "s_t and o_t must share a statistics vector type");
    rv_t m_old, m_new, l_run, alpha;
    neg_infty(m_old);
    zero(l_run);

    // Nine VGPRs of LDS addressing for the whole kernel. See the note above
    // lds_granules.
    // The read granules start on buffer *1*, not 0, so that the preamble below is
    // already the steady state: commit_kv writes V through `vg ^ LDS_XOR`, which
    // is buffer 0, and K through k_wr, which is also buffer 0. Initializing both
    // to buffer 0 is the obvious thing to write and it is wrong -- K lands in
    // buffer 0 and V in buffer 1, so the first block's math reads a V buffer
    // nobody wrote and the output is non-finite. With LDS_STAGES == 1 the two
    // buffers are the same tile and this reads as buffer 0 either way.
    lds_granules<st_k>  kg; kg.init(k_smem1);
    lds_granules<st_vt> vg; vg.init(vt_smem1);
    // The K buffer the *next* block is staged into. V needs no equivalent:
    // commit_kv toggles `vg` across its writes and back, which is eight more
    // v_xor per KV block and four fewer VGPRs than a second granule set. K cannot
    // do the same because its commit goes through the library's
    // store_register_buffer_to_shared, which addresses off the tile itself.
    st_k *k_wr = &k_smem0;

    // One swap at the bottom of each iteration moves everything to the other
    // buffer. With LDS_XOR == 0 (single-buffered) all of this vanishes.
    auto swap_buffers = [&]() {
        kg.template swap<LDS_XOR>();
        vg.template swap<LDS_XOR>();
        if constexpr (LDS_STAGES == 2) k_wr = (k_wr == &k_smem0) ? &k_smem1 : &k_smem0;
    };

    // Stage one KV block into LDS: K straight in, V transposed.
    //
    // V^T[d, kv] is what the PV matmul reads. V arrives [kv, d], so a warp reads
    // a VT_D_CHUNK-wide slice of 16 V rows (a perfectly coalesced [16, VT_D_CHUNK]
    // read), transposes it in registers, and writes the result as a
    // [VT_D_CHUNK, 16] row-major block of vt_smem -- which is a vectorized
    // ds_write_b128 store, because the transposed tile is still row layout.
    //
    // The alternative is to leave V row-major in LDS and let each warp read
    // col-layout operand tiles out of it: 16 ds_read_u16 per base tile, eight
    // times the instructions, paid by every warp on every KV block instead of
    // once here. transpose_base_data costs 20 lane exchanges per base tile
    // (conversions.cuh); at KV_BLOCK=64, HEAD_DIM=128 that is 160 VALU per warp
    // per KV block against roughly 64 WMMAs.
    // The kv tile is the *outer*, compile-time loop and the head-dim chunk the
    // inner, runtime one -- not the other way round, which is what it wants to
    // be written as. The kv tile index is the destination's column in vt_smem,
    // so it picks the granule; the head-dim chunk is the destination's row, and
    // a row is a runtime addend that all eight writes share. Written the other
    // way each write needs its own address, and those addresses are invariant
    // across the KV loop, so LLVM hoists eight of them and spills three.
    // Staging is split in two halves so a prefetch can put the math between
    // them: load_kv() issues the global reads into registers and waits for
    // nothing, commit_kv() does the transpose and the LDS writes. With
    // GPREFETCH=0 they are called back to back and the pair behaves exactly like
    // the single stage_vt this replaced.
    constexpr int D_CHUNKS = HEAD_DIM / VT_D_CHUNK;
    // Bands of V^T one warp owns. At HEAD_DIM=128, VT_D_CHUNK=16 and 12 warps
    // this is 1 and the last four warps idle; it is a ceiling so that a config
    // with more bands than warps still works, and it is compile-time so that
    // v_rows[] is indexed by a constant. A runtime index puts it in scratch,
    // which on a kernel that hand-manages s_waitcnt is silently wrong, not slow.
    constexpr int V_ITERS = (D_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int V_BANDS = KV_BLOCK / 16;
    constexpr int stage_k = G::template stage_calls<st_k>;

    float4 buf_k[stage_k];
    rt_bf<16, VT_D_CHUNK, row_l> v_rows[V_ITERS][V_BANDS];

    auto load_kv = [&](int kv_start) {
        G::load_global_to_register_buffer(buf_k, stage_k, g.k,
                                          coord<>{batch, head_kv, kv_start, 0}, *k_wr);
        static_for<V_ITERS>([&](auto i) {
            const int c = warp_id + decltype(i)::value * NUM_WARPS;
            if (c < D_CHUNKS) {
                static_for<V_BANDS>([&](auto w) {
                    load(v_rows[decltype(i)::value][decltype(w)::value], g.v,
                         coord<>{batch, head_kv,
                                 kv_start + 16 * decltype(w)::value, c * VT_D_CHUNK});
                });
            }
        });
    };

    auto commit_kv = [&]() {
        G::template store_register_buffer_to_shared<false>(*k_wr, buf_k, stage_k);
        // vg addresses the buffer being *read*; the writes go to the other one.
        vg.template swap<LDS_XOR>();
        constexpr int SB = st_vt::swizzle_bytes;
        rt_bf<VT_D_CHUNK, 16, row_l> v_t;
        static_for<V_ITERS>([&](auto i) {
            const int c = warp_id + decltype(i)::value * NUM_WARPS;
            if (c < D_CHUNKS) {
                const uint32_t row_off = (uint32_t)(SB * VT_D_CHUNK) * (uint32_t)c;
                static_for<V_BANDS>([&](auto w) {
                    constexpr int W = decltype(w)::value;
                    transpose_sep(v_t, v_rows[decltype(i)::value][W]);
                    static_for<VT_D_CHUNK / 16>([&](auto t) {
                        lds_write_frag<16 * decltype(t)::value, 16 * W>(
                            vg, row_off, v_t.tiles[decltype(t)::value][0]);
                    });
                });
            }
        });
        vg.template swap<LDS_XOR>();
    };

    // KV blocks, with the last one backed up the same way the Q blocks are --
    // except that here the overlap is *not* idempotent (a score counted twice
    // would be counted twice in the softmax sum), so the overlapping rows are
    // masked to -inf below. Backing up is still worth it: it keeps every global
    // read in bounds without threading a predicate through the staging pipeline.
    //
    // Under causal the loop also stops early. The last query this *workgroup*
    // owns is q_tile_start + Q_TILE - 1, so every KV block starting past it is
    // entirely masked and is not staged at all. This bound has to be uniform
    // across the workgroup -- staging is a group operation with barriers in it
    // -- which is why it uses the tile's last query and not the warp's; the
    // per-warp slack is taken by warp_skip below.
    const int kv_blocks_all = (N + KV_BLOCK - 1) / KV_BLOCK;
    const int kv_blocks = CAUSAL
        ? min(kv_blocks_all, (q_tile_start + Q_TILE + KV_BLOCK - 1) / KV_BLOCK)
        : kv_blocks_all;

    // Where block kb actually reads from, with the tail backed up. Used for the
    // prefetch as well, which is why it is a function of kb rather than a value
    // computed inside the loop: the prefetch needs block kb+1's start, and it has
    // to be clamped to a real block -- the warp-level load() below goes through
    // raw pointers, not a buffer descriptor, so an out-of-range prefetch would be
    // an out-of-bounds read rather than a harmless zero.
    auto block_start = [&](int kb) {
        return min(min(kb, kv_blocks - 1) * KV_BLOCK, N - KV_BLOCK);
    };

#if DBUF && !ABLATE_STAGE
    // Stage block 0 into buffer 0, swap so the loop reads it and writes buffer 1,
    // then issue block 1's globals. From here the invariant at the top of every
    // iteration kb is: buffer `kg`/`vg` holds block kb and is published; buffer
    // `k_wr`/`vgw` is untouched by any wave; block kb+1's globals are in flight.
    load_kv(block_start(0));
    commit_kv();
    swap_buffers();
#if GPREFETCH
    load_kv(block_start(1));
#endif
    lds_wait<0>();
    __builtin_amdgcn_s_barrier();
#elif GPREFETCH && !ABLATE_STAGE
    load_kv(block_start(0));
#endif

    for (int kb = 0; kb < kv_blocks; kb++) {
        const int kv_lo    = kb * KV_BLOCK;                 // first kv this block owns
        const int kv_start = min(kv_lo, N - KV_BLOCK);      // where it actually reads
        const int skip     = kv_lo - kv_start;              // rows already accounted for
        // This warp's share of the block may be empty even when the workgroup's
        // is not: it still stages and still hits both barriers, it just has no
        // math to do. On the diagonal block that is most of the workgroup.
        const bool warp_skip = CAUSAL && (kv_lo > q_row + Q_BLOCK - 1);

#if !DBUF
        // Previous iteration's reads must retire before this one overwrites the
        // buffers. Single-buffered, so this barrier is the whole hazard.
        __builtin_amdgcn_s_barrier();
#if !ABLATE_STAGE
#if GPREFETCH
        // The global reads for this block were issued one iteration ago and have
        // been in flight under the previous block's WMMAs. This is where their
        // vmcnt is paid -- inside store_register_buffer_to_shared, at the first
        // use of each float4 -- and by now there is usually nothing to pay.
        commit_kv();
#else
        load_kv(kv_start);
        commit_kv();
#endif
#endif
        // s_barrier does not order memory; the LDS writes above have to be
        // retired explicitly or the next warp's ds_reads race them.
        lds_wait<0>();
        __builtin_amdgcn_s_barrier();

#if GPREFETCH && !ABLATE_STAGE
        // Issue the next block's global reads *before* this block's math, and
        // after the barrier that published this block. Nothing below touches
        // buf_k or v_rows until the next iteration's commit_kv(), so the loads
        // have the whole math body to land in.
        if (kb + 1 < kv_blocks) load_kv(block_start(kb + 1));
#endif
#endif  // !DBUF

#if HK_DUMP == 5 || HK_DUMP == 6
        // Same readback as 1/2, but through the library's own load(). Splits
        // "the staging is wrong" from "lds_read_frag is wrong".
        if (blockIdx.x == 0 && warp_id == 0) {
#if HK_DUMP == 5
            rt_bf<KV_BLOCK, HEAD_DIM, row_l> dbg;
            load(dbg, k_smem0);
#else
            rt_bf<HEAD_DIM, KV_BLOCK, row_l> dbg;
            load(dbg, vt_smem0);
#endif
            store(g.o, dbg, coord<>{batch, head, 0, 0});
        }
        return;
#endif

#if HK_DUMP == 1 || HK_DUMP == 2
        if (blockIdx.x == 0 && warp_id == 0) {
            rt_bf<16, 16, row_l> dbg;
#if HK_DUMP == 1
            static_for<KV_BLOCK / 16>([&](auto r) {
                static_for<HEAD_DIM / 16>([&](auto c) {
                    lds_read_frag<16 * decltype(r)::value, 16 * decltype(c)::value>(
                        dbg.tiles[0][0], kg);
                    lds_wait<0>();
                    lds_bind(dbg.tiles[0][0]);
                    store(g.o, dbg, coord<>{batch, head,
                                            16 * decltype(r)::value,
                                            16 * decltype(c)::value});
                });
            });
#else
            static_for<HEAD_DIM / 16>([&](auto r) {
                static_for<KV_BLOCK / 16>([&](auto c) {
                    lds_read_frag<16 * decltype(r)::value, 16 * decltype(c)::value>(
                        dbg.tiles[0][0], vg);
                    lds_wait<0>();
                    lds_bind(dbg.tiles[0][0]);
                    store(g.o, dbg, coord<>{batch, head,
                                            16 * decltype(r)::value,
                                            16 * decltype(c)::value});
                });
            });
#endif
        }
        return;
#endif

        // A wave with nothing above the diagonal does no math this block. It
        // still falls through to the staging and the barrier at the bottom --
        // that is a workgroup-wide operation and its share of the next block is
        // nobody else's to do. Skipping the math is exactly a no-op and not an
        // approximation: an all -inf S^T gives m_new == m_old, alpha == 1 and a
        // zero P^T, so l_run and o_t would come out unchanged.
        // The predicate is made opaque so that this is a real branch even when
        // CAUSAL is false and warp_skip folds to a constant.
        //
        // It is not a branch we want taken -- non-causal it never is. It is a
        // control-flow boundary, and the register allocator needs one here. With
        // the math and the staging in a single straight-line region the allocator
        // has to hold both live at once and gives up: 256 VGPRs and 472-508
        // bytes/lane of scratch, identical for every QK_TILES/PV_TILES setting,
        // which is what ruled out simple pressure as the cause. The causal
        // instantiation, which has this branch for real, fits in 249 with no
        // scratch. Adding it back non-causal costs one VGPR and a never-taken
        // s_cbranch, and drops scratch to zero.
        //
        // Scratch is not a slowdown in this kernel, it is a wrong answer: the
        // LDS schedule hand-manages s_waitcnt and a spill breaks what the waits
        // mean. See the note at lds_bind.
        int do_math = warp_skip ? 0 : 1;
        asm volatile("" : "+v"(do_math));
        if (do_math) {

        // ---- S^T = K . Q^T ----------------------------------------------
        // A QK_TILES-tall, 16-wide window of K at a time. Q is already resident,
        // so the loop is: read the window, multiply it against the matching
        // 16-column slice of Q, move on.
        zero(s_t);
#if !ABLATE_QK
        {
            rt_bf<16 * QK_TILES, 16, row_l> k_chunk;
#if ABLATE_LDS_READ
            zero(k_chunk);
#endif
            static_for<KV_BLOCK / (16 * QK_TILES)>([&](auto ng) {
                static_for<HEAD_DIM / 16>([&](auto kk) {
#if !ABLATE_LDS_READ
                    static_for<QK_TILES>([&](auto n) {
                        lds_read_frag<16 * (QK_TILES * decltype(ng)::value + decltype(n)::value),
                                      16 * decltype(kk)::value>(k_chunk.tiles[decltype(n)::value][0], kg);
                    });
                    lds_wait<0>();
                    static_for<QK_TILES>([&](auto n) {
                        lds_bind(k_chunk.tiles[decltype(n)::value][0]);
                    });
#endif
                    static_for<QK_TILES>([&](auto n) {
                        constexpr int r = QK_TILES * decltype(ng)::value + decltype(n)::value;
                        mma_ABt_base(s_t.tiles[r][0], k_chunk.tiles[decltype(n)::value][0],
                                     q.tiles[0][decltype(kk)::value], s_t.tiles[r][0]);
                    });
                });
            });
        }
#endif
#if HK_DUMP
        // Write a [KV_BLOCK, Q_BLOCK] col tile out as [q, kv] and stop.
#define HK_DUMP_ST(TILE)                                                      \
        do {                                                                  \
            if (blockIdx.x == 0) {                                            \
                rt_fl<Q_BLOCK, 16, col_l> dbg;                                \
                _Pragma("unroll")                                             \
                for (int n = 0; n < KV_BLOCK / 16; n++) {                     \
                    transpose(dbg.tiles[0][0], (TILE).tiles[n][0]);           \
                    store(g.o, dbg, coord<>{batch, head, q_row, n * 16});     \
                }                                                             \
            }                                                                 \
            return;                                                           \
        } while (0)
#endif

        // Raw register dump: warp 0 writes its 16 accumulator registers straight
        // to o[lane, slot], plus skip/kv_start/N, with no transpose and no store
        // path in between.  Placed at several points so the corruption can be
        // pinned to one statement.
#define HK_DUMP_REGS()                                                          \
        do {                                                                    \
            if (blockIdx.x == 0 && warp_id == 0) {                              \
                const int l = threadIdx.x & 31;                                 \
                _Pragma("unroll")                                               \
                for (int i = 0; i < KV_BLOCK / 16; i++) {                       \
                    _Pragma("unroll")                                           \
                    for (int k = 0; k < 4; k++) {                               \
                        g.o[coord<>{batch, head, l, 8 * i + 2 * k}] =           \
                            base_types::convertor<bf16, float>::convert(        \
                                s_t.tiles[i][0].data[k].x);                     \
                        g.o[coord<>{batch, head, l, 8 * i + 2 * k + 1}] =       \
                            base_types::convertor<bf16, float>::convert(        \
                                s_t.tiles[i][0].data[k].y);                     \
                    }                                                           \
                }                                                               \
                g.o[coord<>{batch, head, l, 16}] =                              \
                    base_types::convertor<bf16, float>::convert((float)skip);   \
                g.o[coord<>{batch, head, l, 17}] =                              \
                    base_types::convertor<bf16, float>::convert((float)kv_start);\
                g.o[coord<>{batch, head, l, 18}] =                              \
                    base_types::convertor<bf16, float>::convert((float)N);      \
            }                                                                   \
            return;                                                             \
        } while (0)

#if HK_DUMP == 10
        HK_DUMP_REGS();
#endif
#if HK_DUMP == 3
        HK_DUMP_ST(s_t);
#endif
        mul(s_t, s_t, scale_l2e);
#if HK_DUMP == 9
        HK_DUMP_ST(s_t);
#endif
#if HK_DUMP == 11
        HK_DUMP_REGS();
#endif

        // Mask the scores this block does not own: the backed-up tail overlap,
        // and under causal everything above the diagonal.
        //
        // Both axes of S^T are free here, and that is the second dividend of
        // storing it transposed. A *row* of S^T is a kv index, which for a
        // col-layout fp32 accumulator is the element axis -- unrolled, so the kv
        // of every register is a compile-time-shaped expression in the lane id.
        // A *column* is a query index, which is the lane axis, so q is
        // `lane & 15` and does not vary within a register. The causal predicate
        // is therefore one v_cmp per element with no cross-lane traffic and no
        // materialized mask tile; the [q, kv] orientation would need
        // make_causal's shuffles instead.
        if (CAUSAL || skip > 0) {
            constexpr int E = rt_base<float, ducks::rt_layout::col>::elements_per_thread;
            const int q_abs = q_row + (int)(lane & 15);
            #pragma unroll
            for (int n = 0; n < KV_BLOCK / 16; n++) {
                #pragma unroll
                for (int e = 0; e < E; e++) {
                    const int kv = n * 16 + rt_base_coord<float, ducks::rt_layout::col>(e, lane).x;
                    const bool masked = kv < skip || (CAUSAL && kv_start + kv > q_abs);
                    if (masked) {
                        if (e & 1) s_t.tiles[n][0].data[e >> 1].y = -INFINITY;
                        else       s_t.tiles[n][0].data[e >> 1].x = -INFINITY;
                    }
                }
            }
        }

        // ---- online softmax ---------------------------------------------
        // Every reduction here is over kv, which is s_t's element axis: eight
        // register-local steps plus one cross-half exchange, no butterfly.
#if !ABLATE_SOFTMAX
        col_max(m_new, s_t, m_old);
        sub_col(s_t, s_t, m_new);
        exp2(s_t, s_t);                  // s_t is P^T from here

#if RESCALE_SKIP
        // Wave-uniform: every lane holds the whole q range of this warp's tile
        // in m_*, so the branch is taken or not taken for the wave as a whole.
        bool grew = false;
        #pragma unroll
        for (int o = 0; o < rv_t::outer_dim; o++)
            #pragma unroll
            for (int i = 0; i < rv_t::inner_dim; i++)
                grew |= (m_new.data[o][i] != m_old.data[o][i]);
        if (__any(grew)) {
            sub(alpha, m_old, m_new);
            exp2(alpha, alpha);          // 0 on the first block, where m_old = -inf
            mul(l_run, l_run, alpha);
            mul_col(o_t, o_t, alpha);
        }
        col_sum(l_run, s_t, l_run);
        copy(m_old, m_new);
#else
        sub(alpha, m_old, m_new);
        exp2(alpha, alpha);              // 0 on the first block, where m_old = -inf

        mul(l_run, l_run, alpha);
        col_sum(l_run, s_t, l_run);
        mul_col(o_t, o_t, alpha);
        copy(m_old, m_new);
#endif
#endif

#if HK_DUMP == 4
        HK_DUMP_ST(s_t);
#endif

        // ---- O^T += V^T . P^T -------------------------------------------
        // The kv fragment is the outer loop so that only *one* 16x16 block of
        // P^T is ever in bf16 at a time. Converting the whole score tile up
        // front is the obvious way to write this and it costs KV_BLOCK/16 * 8
        // more VGPRs, which at HEAD_DIM=128 is the difference between fitting
        // and spilling -- and a spill here is not slow, it is wrong, because
        // the LDS schedule hand-manages s_waitcnt.
#if !ABLATE_PV
        {
            rt_bf<16, Q_BLOCK, col_l> p_base;
            rt_bf<16 * PV_TILES, 16, row_l> vt_chunk;
#if ABLATE_LDS_READ
            zero(vt_chunk);
#endif
            static_for<KV_BLOCK / 16>([&](auto kk) {
                // fp32 accumulator -> bf16 operand. Across replications this is
                // not pointwise: one permlanex16 per element fetches the parity
                // the mirror lane owns. It is the conversion the replicated
                // operand layout exists for -- the alternative is a transpose.
                copy(p_base.tiles[0][0], s_t.tiles[decltype(kk)::value][0]);
                static_for<HEAD_DIM / (16 * PV_TILES)>([&](auto ng) {
#if !ABLATE_LDS_READ
                    static_for<PV_TILES>([&](auto n) {
                        lds_read_frag<16 * (PV_TILES * decltype(ng)::value + decltype(n)::value),
                                      16 * decltype(kk)::value>(vt_chunk.tiles[decltype(n)::value][0], vg);
                    });
                    lds_wait<0>();
                    static_for<PV_TILES>([&](auto n) {
                        lds_bind(vt_chunk.tiles[decltype(n)::value][0]);
                    });
#endif
                    static_for<PV_TILES>([&](auto n) {
                        constexpr int r = PV_TILES * decltype(ng)::value + decltype(n)::value;
                        mma_AB_base(o_t.tiles[r][0], vt_chunk.tiles[decltype(n)::value][0],
                                    p_base.tiles[0][0], o_t.tiles[r][0]);
                    });
                });
            });
        }
#endif
        }  // if (do_math)

#if DBUF
        // Stage block kb+1 into the buffer nobody is reading, and issue kb+2's
        // global reads. There is deliberately no barrier in front of this: the
        // buffer being written was last read in iteration kb-1, and *that*
        // iteration's closing barrier already guarantees every wave is done with
        // it. That is the whole saving -- one barrier per KV block rather than
        // two -- and it is what lets a wave still in its WMMAs and a wave already
        // in its ds_writes share a SIMD.
        // Keep the scheduler from hoisting the staging up into the math. Under
        // CAUSAL the `if (!warp_skip)` above is a real branch and provides this
        // boundary for free, which is why the causal instantiation fits at 249
        // VGPRs; non-causal it folds away, the whole iteration becomes one
        // straight-line region, and the hoisted staging loads overlap the entire
        // math body -- 256 VGPRs and 480 bytes/lane of scratch. The overlap this
        // gives up was never the point: the staging is hidden *across* waves, by
        // the SIMD having another wave to run, not within one.
        __builtin_amdgcn_sched_barrier(0);
#if !ABLATE_STAGE
        if (kb + 1 < kv_blocks) {
#if GPREFETCH
            // buf_k/v_rows were filled an iteration ago; this pays their vmcnt.
            commit_kv();
            load_kv(block_start(kb + 2));
#else
            // Issued and consumed here. The vmcnt stall is real but it is this
            // wave's alone -- no barrier stands between it and the other waves'
            // math -- and it costs no registers across the math body.
            load_kv(block_start(kb + 1));
            commit_kv();
#endif
        }
#endif
        // s_barrier does not order memory; the LDS writes above have to be
        // retired explicitly or the next iteration's ds_reads race them.
        lds_wait<0>();
        __builtin_amdgcn_s_barrier();
        swap_buffers();
#endif
    }

    div_col(o_t, o_t, l_run);

    // Back to [q, d] for the store. An accumulator transpose is 8 permlanex16
    // plus 12 DPP movs per base tile, paid once per Q block rather than once per
    // KV block -- which is the trade the transposed accumulator bought.
    //
    // One fragment at a time, not transpose_sep on the whole thing. The whole
    // thing needs a second [Q_BLOCK, HEAD_DIM] accumulator live alongside o_t --
    // 128 VGPRs between them -- and that single line was worth 392 bytes/lane of
    // scratch and 99 spilled VGPRs, in a kernel whose hand-written s_waitcnt
    // schedule makes a spill a correctness bug rather than a slowdown. Per
    // fragment the transient is 8 VGPRs and the store count is identical.
    {
        rt_fl<Q_BLOCK, 16, col_l> o_slice;
        #pragma unroll
        for (int n = 0; n < HEAD_DIM / 16; n++) {
            transpose(o_slice.tiles[0][0], o_t.tiles[n][0]);
            store(g.o, o_slice, coord<>{batch, head, q_row, n * 16});
        }
    }
}

template<typename C, bool CAUSAL, typename GL = micro_globals>
static void launch(const GL &g) {
    const unsigned long mem_size = C::SHARED_BYTES;
    auto kern = micro_tk<C, CAUSAL, GL>;
    hipFuncSetAttribute((void*)kern, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    if (getenv("HK_OCC")) {
        static bool once = false;
        if (!once) {
            once = true;
            int blocks = 0;
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)kern,
                                                         C::NUM_THREADS, mem_size);
            fprintf(stderr, "occ: %d blocks/WGP, %d waves/WGP, %.1f waves/SIMD"
                            " (LDS %lu B, %d threads)\n",
                    blocks, blocks * C::NUM_WARPS, blocks * C::NUM_WARPS / 4.0,
                    mem_size, C::NUM_THREADS);
        }
    }
    const int n = g.q.rows();
    const int q_blocks = (n + C::Q_TILE - 1) / C::Q_TILE;
    kern<<<dim3(q_blocks, g.q.depth(), g.q.batch()),
           dim3(C::NUM_THREADS), mem_size, g.stream>>>(g);
}

// The head dimensions that get a kernel. The tiling knobs are the same for
// both; only the template width differs. Anything else falls back.
#define HK_ATTN_HEAD_DIMS(X) X(64) X(128)

// The shapes this kernel can take at all. The caller is expected to ask rather
// than to reimplement the rule; the torch extension's fallback is built on it.
//
// `n` is the sequence length of q, k and v alike: the kv loop is bounded by
// q.rows(), so a cross-attention shape with n_kv != n_q would silently read the
// wrong number of keys rather than fail. The dispatcher checks it.
template<typename C>
static bool supported(int n, int d) {
    bool d_ok = false;
#define HK_ATTN_CHECK_D(D) d_ok = d_ok || (d == (D));
    HK_ATTN_HEAD_DIMS(HK_ATTN_CHECK_D)
#undef HK_ATTN_CHECK_D
    return d_ok && n >= C::Q_TILE && n >= C::KV_BLOCK;
}

void dispatch_micro(micro_globals g) {
    const int d = g.q.cols();
    // Two template axes, both compile-time in the kernel: head dimension, which
    // every tile type depends on, and causal, which changes the trip count.
#define HK_ATTN_DISPATCH(D)                                                   \
    if (d == (D)) {                                                           \
        if (g.causal) launch<cfg_for<D>, true>(g);                            \
        else          launch<cfg_for<D>, false>(g);                           \
        return;                                                               \
    }
    HK_ATTN_HEAD_DIMS(HK_ATTN_DISPATCH)
#undef HK_ATTN_DISPATCH
    // Unreachable through the torch extension, which asks supported() first.
    fprintf(stderr, "hk attn: unsupported head_dim %d\n", d);
    abort();
}

#ifndef HK_ATTN_NO_PYBIND
// Overridable so that two builds with different tiling knobs can be imported
// into one process and timed against each other. They have to be: bench() warns
// that clock and power state drift between runs on this node, and a cross-
// process A/B on this part manufactures double-digit differences that do not
// exist. ab.py is the harness.
#ifndef TK_MODULE_NAME
#define TK_MODULE_NAME tk_kernel
#endif
PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
                                      &micro_globals::q, &micro_globals::k,
                                      &micro_globals::v, &micro_globals::o,
                                      &micro_globals::scale,
                                      &micro_globals::causal);
}
#endif
