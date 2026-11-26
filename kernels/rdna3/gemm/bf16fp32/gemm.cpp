// bf16 GEMM with fp32 accumulate for gfx1100 (RDNA3).
//
// Structurally this follows kernels/cdna4/gemm/bf16fp32, but the schedule is
// redesigned for RDNA3, where three things are different:
//
//   * wave32, and a WMMA operand is mirrored across the two wave halves, so a
//     bf16 rt_base costs the same 8 VGPRs as an fp32 one. Operand registers,
//     not accumulator registers, are what bounds the tile here.
//   * no global->LDS DMA. Every byte of a prefetch passes through VGPRs, so a
//     "async copy" is load_global_to_register_buffer now and
//     store_register_buffer_to_shared at the point the buffer is needed.
//   * one XCD, so there is no chiplet swizzle -- only the L2 group swizzle.
//
// B is passed pre-transposed as (n, k), which keeps both shared loads
// contiguous and makes mma_ABt the right primitive.

#include <utility>

#include "kittens.cuh"
// pyutils pulls in pybind11 and the Python headers. The distributed kernel
// includes this file for the tuned inner loop and has its own host harness, so
// both the include and the module below are opt-out -- otherwise a build with
// no Python development headers could not use the GEMM at all.
#ifndef HK_GEMM_NO_PYBIND
#include "pyutils/pyutils.cuh"
#endif
using namespace kittens;

// Tiling knobs. The defaults are the shape this file is named for; the sweep
// script overrides them on the command line.
#ifndef BLOCK_M
#define BLOCK_M 128
#endif
#ifndef BLOCK_N
#define BLOCK_N 128
#endif
#ifndef K_STEP
#define K_STEP 32
#endif
#ifndef DOT_SLICE
#define DOT_SLICE 16
#endif
#ifndef NUM_WARPS
#define NUM_WARPS 8
#endif
#ifndef WARP_ROWS
#define WARP_ROWS 4
#endif
// How many N-direction chunks the warp's operand tile and accumulator are split
// into. See the wait schedule in dot_tile(); 1 reproduces the unsplit loop.
#ifndef N_SPLIT
#define N_SPLIT 2
#endif
// How many K-tiles of global data are kept in flight in the staging registers.
// 1 is the classic one-tile-deep pipeline: the loads for tile t+2 are issued
// just before tile t's math and waited on at the top of the next iteration, so
// the distance they get to cover is one K-tile of WMMAs. Deeper costs
// GPREFETCH * (stage_a + stage_b) float4s and buys a proportionally longer
// shadow, which is what a short K-tile needs.
#ifndef GPREFETCH
#define GPREFETCH 1
#endif
// Where in the iteration the staged K-tile gets written to LDS.
//   0  before the math, which is where the obvious loop structure puts it
//   1  after the math, so the tile's ds_reads are not queued behind the writes
//   2  inside the math, at the point where the last ds_read has been issued:
//      late enough not to delay a read, early enough that the last K-slice's
//      WMMAs cover the write latency instead of the barrier doing it
// 1 is the default because it measures fastest (79 vs 78 TFLOPs at 8192^2x4096).
// 2 is the more interesting schedule and it is correct, but it buys nothing: the
// writes are only 1/8 of the tile's LDS traffic, so hiding them under the last
// K-slice's WMMAs just moves the same cycles around inside a pipe that is
// already the bottleneck.
#ifndef WRITE_POS
#define WRITE_POS 1
#endif
// Rotate each chunk's next-slice reads into the slot the chunk just freed.
#ifndef ROTATE
#define ROTATE 1
#endif
#ifndef MIN_BLOCKS_PER_CU
#define MIN_BLOCKS_PER_CU 1
#endif
// How many block-rows the L2 swizzle walks before moving on. Only affects which
// workgroups are co-resident, not what they compute.
#ifndef WGM
#define WGM 8
#endif

// Ablation switches for the inner loop. Setting any of these makes the result
// numerically wrong on purpose -- they exist to time one stage of the pipeline
// at a time, see sweep.sh.
#ifndef ABLATE_GLOBAL
#define ABLATE_GLOBAL 0
#endif
#ifndef ABLATE_LDS_READ
#define ABLATE_LDS_READ 0
#endif
#ifndef ABLATE_MMA
#define ABLATE_MMA 0
#endif
#ifndef ABLATE_LDS_WRITE
#define ABLATE_LDS_WRITE 0
#endif
#ifndef ABLATE_BARRIER
#define ABLATE_BARRIER 0
#endif

// A constexpr for loop, so the loop index can be a template argument. lds_wait<N>
// needs one; #pragma unroll does not give you that.
template<int N, typename F> __device__ inline void static_for(F &&f) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (f(std::integral_constant<int, I>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

// How many LDS reads are still in flight once chunk (k, c)'s operands have
// landed, under the rotated issue schedule in dot_tile(). Reads retire in
// order, so this is just "how many were issued after the last one this chunk
// needs". Deriving it here rather than writing lds_wait<0> everywhere is the
// entire point of the schedule: a nonzero value is math running on top of LDS.
constexpr int reads_in_flight(int k, int c, int n_split, int num_slices, int chunk_reads) {
    const bool has_next = (k + 1 < num_slices);
    // First slice: its own B chunks were all issued up front, so the tail of
    // them is still moving, plus whatever has been rotated in behind them.
    if (k == 0) return (n_split - 1 - c + (has_next ? c : 0)) * chunk_reads;
    // Later slices: A was issued after B chunks 0..n-2 and before B chunk n-1,
    // so for every chunk but the last, A is the read that gates.
    if (c < n_split - 1) return ((has_next ? c : 0) + 1) * chunk_reads;
    return (has_next ? (n_split - 1) : 0) * chunk_reads;
}

// Pin a register tile's values as live without emitting an instruction. The
// ablations below delete whichever stage they are measuring, and without this
// the compiler would helpfully delete the stage feeding it too, so e.g.
// ABLATE_MMA would also drop every ds_read and measure nothing.
template<typename RT> __device__ inline void keep_live(RT &t) {
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++)
                asm volatile("" :: "v"(t.tiles[i][j].data[k]));
}

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    hipStream_t stream;   // a: (m, k), b: (n, k), c: (m, n)
    // Split-K scratch, [SPLIT_K][c.rows()][c.cols()] floats. Only read when the
    // selected config has SPLIT_K > 1; null otherwise.
    float *ws = nullptr;

    // The epilogue is selected on these rather than on a template parameter of
    // the kernel, so that a caller with a different output policy supplies a
    // different globals type and nothing else changes. See
    // kernels/rdna3/distributed/gemm_rs.hip, whose globals set FUSED_RS true to
    // turn the store into a reduce-scatter push, and gemm_ar_mp.hip, which sets
    // FUSED_AR for the all-reduce form.
    static constexpr bool FUSED_RS = false;
    static constexpr bool FUSED_AR = false;
};

// The command line sets the tunables as macros, and the config struct below
// wants the same names as members, so capture the values and then get the macros
// out of the way.
namespace cli {
constexpr int block_m = BLOCK_M, block_n = BLOCK_N, k_step = K_STEP,
              dot_slice = DOT_SLICE, num_warps = NUM_WARPS,
              warp_rows = WARP_ROWS, wgm = WGM, n_split = N_SPLIT,
              gprefetch = GPREFETCH, write_pos = WRITE_POS;
}
#undef WRITE_POS
#undef GPREFETCH
#undef N_SPLIT
#undef BLOCK_M
#undef BLOCK_N
#undef K_STEP
#undef DOT_SLICE
#undef NUM_WARPS
#undef WARP_ROWS
#undef WGM

// One tiling. The kernel is templated on this rather than compiled from the
// macros directly, because no single tiling wins across shapes -- see the table
// in README.md and the selection rule in dispatch_micro below.
template<int _BLOCK_M, int _BLOCK_N, int _K_STEP, int _DOT_SLICE,
         int _NUM_WARPS, int _WARP_ROWS, int _WGM, int _N_SPLIT = 2,
         int _GPREFETCH = 1, int _WRITE_POS = 2, int _SPLIT_K = 1>
struct config {
    static constexpr int BLOCK_M = _BLOCK_M, BLOCK_N = _BLOCK_N;
    static constexpr int K_STEP  = _K_STEP,  DOT_SLICE = _DOT_SLICE;
    static constexpr int NUM_WARPS = _NUM_WARPS, WARP_ROWS = _WARP_ROWS;
    static constexpr int WGM = _WGM, N_SPLIT = _N_SPLIT;
    static constexpr int GPREFETCH = _GPREFETCH, WRITE_POS = _WRITE_POS;
    // Split-K: cut the K loop into this many independent slices, each its own
    // workgroup, and reduce afterwards. Only worth it when the M/N grid alone
    // cannot fill the device -- decode, where N/BLOCK_N is a couple of dozen
    // workgroups on 48 CUs and the kernel is latency- rather than
    // bandwidth-bound. With SPLIT_K > 1 the epilogue writes fp32 partials
    // instead of C, and a second kernel reduces them; the reduction is
    // deterministic (a fixed-order sum, not fp32 atomics) because a framework
    // cannot accept results that change run to run.
    static constexpr int SPLIT_K = _SPLIT_K;

    static constexpr int NUM_THREADS = kittens::WARP_THREADS * NUM_WARPS;
    // Warps tile the block WARP_ROWS (M) by WARP_COLS (N).
    static constexpr int WARP_COLS = NUM_WARPS / WARP_ROWS;
    static_assert(WARP_ROWS * WARP_COLS == NUM_WARPS, "warp grid must cover the block");
    static constexpr int REG_BLOCK_M = BLOCK_M / WARP_ROWS;
    static constexpr int REG_BLOCK_N = BLOCK_N / WARP_COLS;
    static constexpr int SPLIT_N = REG_BLOCK_N / N_SPLIT;
    static_assert(SPLIT_N % 16 == 0 && SPLIT_N * N_SPLIT == REG_BLOCK_N,
                  "N_SPLIT must divide the warp tile into whole 16-column chunks");
    // ds_read_b128 count for one chunk: two per 16x16 base tile.
    static constexpr int CHUNK_READS = (SPLIT_N / 16) * (DOT_SLICE / 16) * 2;

    using st_a = st_bf<BLOCK_M, K_STEP>;
    using st_b = st_bf<BLOCK_N, K_STEP>;
    using G    = kittens::group<NUM_WARPS>;

    // 2 buffers each of A and B.
    static constexpr size_t SHARED_BYTES = 2 * (sizeof(st_a) + sizeof(st_b));
    static_assert(SHARED_BYTES <= MAX_SHARED_MEMORY, "block does not fit in 64KB of LDS");
};

// What -DBLOCK_M=... on the command line selects. sweep.sh drives this one.
using macro_config = config<cli::block_m, cli::block_n, cli::k_step, cli::dot_slice,
                            cli::num_warps, cli::warp_rows, cli::wgm, cli::n_split,
                            cli::gprefetch, cli::write_pos>;

template<typename C, typename GL = micro_globals>
__global__ __launch_bounds__(C::NUM_THREADS, MIN_BLOCKS_PER_CU)
void micro_tk(const GL g) {
    constexpr int BLOCK_M = C::BLOCK_M, BLOCK_N = C::BLOCK_N;
    constexpr int K_STEP = C::K_STEP, DOT_SLICE = C::DOT_SLICE;
    constexpr int WARP_ROWS = C::WARP_ROWS, WARP_COLS = C::WARP_COLS;
    constexpr int REG_BLOCK_M = C::REG_BLOCK_M, REG_BLOCK_N = C::REG_BLOCK_N;
    constexpr int WGM = C::WGM;
    constexpr int N_SPLIT = C::N_SPLIT, SPLIT_N = C::SPLIT_N, CHUNK_READS = C::CHUNK_READS;
    using st_a = typename C::st_a;
    using st_b = typename C::st_b;
    using G    = typename C::G;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_a (&As)[2] = al.allocate<st_a, 2>();
    st_b (&Bs)[2] = al.allocate<st_b, 2>();

    constexpr int NUM_SLICES = K_STEP / DOT_SLICE;
    // B and the accumulator are cut into N_SPLIT chunks along N. Same total
    // registers as one wide pair -- this is a re-association, not a buffer.
    rt_bf<REG_BLOCK_M, DOT_SLICE, row_l> A_tile;
    rt_bf<SPLIT_N, DOT_SLICE, row_l> B_tile[N_SPLIT];
    rt_fl<REG_BLOCK_M, SPLIT_N, col_l> C_accum[N_SPLIT];
    #pragma unroll
    for (int s = 0; s < N_SPLIT; s++) zero(C_accum[s]);
#if ABLATE_LDS_READ
    zero(A_tile);                 // the ds_reads that would fill these are gone
    #pragma unroll
    for (int s = 0; s < N_SPLIT; s++) zero(B_tile[s]);
#endif

    const int M = g.a.rows();
    // ceil, not floor: a partial last block still has to run. The rows it
    // covers are fixed up by m_base below rather than by predication.
    const int m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
    const int n_blocks = g.b.rows() / BLOCK_N;

    // L2 swizzle: walk WGM block-rows at a time so that the B panel a group of
    // workgroups reads stays resident. No chiplet term -- NUM_XCDS is 1 here.
    int wgid = blockIdx.x;
    const int wgs_per_group = WGM * n_blocks;
    const int group_id      = wgid / wgs_per_group;
    const int first_pid_m   = group_id * WGM;
    const int group_size_m  = min(m_blocks - first_pid_m, WGM);
    const int row = first_pid_m + ((wgid % wgs_per_group) % group_size_m);
    const int col = (wgid % wgs_per_group) / group_size_m;

    // M remainder, by backing the last block up rather than predicating it.
    //
    // The alternative -- masking loads and stores -- is not available cheaply:
    // global_to_shared's load() is raw pointer arithmetic with no bounds check,
    // so predication would have to be threaded through the whole staging
    // pipeline, and the right vmcnt would stop being a compile-time immediate.
    //
    // Backing up costs nothing instead. The last block starts at M - BLOCK_M,
    // so every row it touches is in range and no access needs a predicate. The
    // rows in the overlap with the previous block are computed twice, but both
    // workgroups compute them from the same A rows and the same full K, so they
    // write identical values -- the redundancy is idempotent, not a race.
    //
    // M < BLOCK_M cannot back up at all, so it clamps to 0 and the block runs
    // wider than the data. That is only reachable through the thin config,
    // whose contract says A and C carry BLOCK_M=16 rows of allocation; the
    // rows past M compute and store garbage that the caller slices off.
    const int m_base = M >= BLOCK_M ? min(row * BLOCK_M, M - BLOCK_M) : 0;

    const int warp_id  = kittens::warpid();
    const int warp_row = warp_id / WARP_COLS;
    const int warp_col = warp_id % WARP_COLS;

    // K slice for this workgroup. With SPLIT_K == 1 this collapses to exactly
    // the old `num_tiles = K / K_STEP` with t_begin folded away at compile
    // time, so the non-decode configs are untouched.
    constexpr int SPLIT_K = C::SPLIT_K;
    const int total_tiles = g.a.cols() / K_STEP;
    int t_begin = 0, num_tiles = total_tiles;
    if constexpr (SPLIT_K > 1) {
        const int per = total_tiles / SPLIT_K;   // dispatch checks divisibility
        t_begin   = blockIdx.y * per;
        num_tiles = per;
    }

    // Prefetch staging. There is no global->LDS DMA on RDNA, so the "async
    // copy" is: buffer_load into these VGPRs on one iteration, ds_write them on
    // a later one. GPREFETCH K-tiles of math sit between the two halves.
    constexpr int GP = C::GPREFETCH;
    constexpr int stage_a = G::template stage_calls<st_a>;
    constexpr int stage_b = G::template stage_calls<st_b>;
    float4 buf_a[GP][stage_a];
    float4 buf_b[GP][stage_b];
    // Loads still in flight once the oldest batch has landed. Buffer loads
    // return in vmcnt order, so this is the vmcnt analogue of the lgkmcnt
    // schedule in dot_tile(): wait for batch p and leave the GP-1 newer
    // batches moving.
    constexpr int VM_KEEP = (GP - 1) * (stage_a + stage_b);
    static_assert(VM_KEEP <= 63, "vmcnt is 6 bits; GPREFETCH is too deep");

    // One K-tile of math: read a DOT_SLICE-wide pair of operand tiles out of
    // LDS, multiply, repeat.
    //
    // Double-buffering the operand tiles so slice k+1's reads overlap slice k's
    // WMMAs was tried and is *slower*: 63/68 TFLOPs against 66/72, because the
    // second set of tiles costs 48 VGPRs and takes occupancy from 9 waves/SIMD
    // to 7. So the overlap here is arranged to cost zero registers instead.
    //
    // The whole slice's reads -- A plus every B chunk -- go in flight at once,
    // and then the waits step down: lds_wait<N> means "at most N still
    // outstanding", and LDS retires in order, so waiting for
    // (N_SPLIT-1-s)*CHUNK_READS retires exactly A and B chunks 0..s while
    // chunks s+1.. are still moving. Chunk s's WMMAs then issue on top of them.
    // No tile is duplicated: the reads that overlap the math are reads this
    // slice had to do anyway, just not fenced in front of it.
    //
    // The waits below are lds_wait_for, not lds_wait, and that is a correctness
    // requirement rather than a style: s_waitcnt carries no register dependence,
    // so a v_wmma -- a pure builtin, with no memory effect for the "memory"
    // clobber to order it against -- can be and is scheduled above the wait for
    // the ds_read that feeds it. lds_wait_for binds the operand tiles after the
    // wait and puts that edge back, at zero instructions. This kernel was
    // shipping without it and was correct only by scheduling luck; the attention
    // kernel next door is where the luck ran out. See lds_bind in
    // include/rdna3/ops/warp/memory/util/util.cuh.
    auto load_A = [&](int buf, int k) {
        load<false>(A_tile, subtile_inplace<REG_BLOCK_M, DOT_SLICE>(As[buf], {warp_row, k}));
    };
    auto load_B = [&](int buf, int k, int c) {
        load<false>(B_tile[c], subtile_inplace<SPLIT_N, DOT_SLICE>(
                                   Bs[buf], {warp_col * N_SPLIT + c, k}));
    };
    auto mma_chunk = [&](int c) {
#if !ABLATE_MMA
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[c], A_tile, B_tile[c], C_accum[c]);
        __builtin_amdgcn_s_setprio(0);
#else
        keep_live(A_tile); keep_live(B_tile[c]);
#endif
    };

    // `has_tail` has to be a compile-time argument rather than derived from
    // WRITE_POS alone: the epilogue calls dot_tile with an empty tail, and an
    // lgkmcnt immediate that is too *large* does not over-wait, it fails to
    // wait at all. Counting writes that were never issued there would let the
    // last tile's WMMAs run on operands still in flight.
    auto dot_tile = [&](int buf, auto has_tail, auto &&tail) {
        // LDS ops the interleaved store adds to the queue. Writes issued after
        // a batch of reads retire after them -- lgkmcnt is in order -- so every
        // wait downstream of the store has to carry them, or it would retire
        // more reads than the schedule intends.
        constexpr int TAIL_OPS =
            (has_tail() && C::WRITE_POS == 2) ? (stage_a + stage_b) : 0;
#if ABLATE_LDS_READ
        tail();
        #pragma unroll
        for (int k = 0; k < NUM_SLICES; k++)
            static_for<N_SPLIT>([&](auto c) { mma_chunk(c); });
#elif !ROTATE
        tail();
        #pragma unroll
        for (int k = 0; k < NUM_SLICES; k++) {
            load_A(buf, k);
            static_for<N_SPLIT>([&](auto c) { load_B(buf, k, c); });
            static_for<N_SPLIT>([&](auto c) {
                lds_wait_for<(N_SPLIT - 1 - c) * CHUNK_READS>(A_tile, B_tile[c]);
                mma_chunk(c);
            });
        }
#else
        // Slice 0 up front; after that every read is issued by the chunk whose
        // register slot it is about to reuse. B_tile[c] is dead the instant
        // chunk c's WMMAs have issued, so slice k+1's chunk c goes in flight
        // there, one full chunk of math early, at no register cost. A_tile is
        // dead only after the slice's last chunk, so A goes out at the boundary
        // -- and *before* that chunk's B, so that the next slice's first chunk
        // is gated on A rather than on a read issued behind it.
        load_A(buf, 0);
        static_for<N_SPLIT>([&](auto c) { load_B(buf, 0, c); });
        // Nothing left to issue for this tile when there is only one slice.
        if constexpr (NUM_SLICES == 1) tail();

        static_for<NUM_SLICES>([&](auto k) {
            static_for<N_SPLIT>([&](auto c) {
                // Slice NUM_SLICES-1's reads were all issued before the store,
                // so its waits are the ones that have to count the writes.
                constexpr int extra = (k == NUM_SLICES - 1) ? TAIL_OPS : 0;
                lds_wait_for<reads_in_flight(k, c, N_SPLIT, NUM_SLICES, CHUNK_READS) + extra>(
                    A_tile, B_tile[c]);
                mma_chunk(c);
                if constexpr (k + 1 < NUM_SLICES) {
                    if constexpr (c == N_SPLIT - 1) load_A(buf, k + 1);
                    load_B(buf, k + 1, c);
                    // Last read of the tile is now in flight: hand the LDS
                    // pipe to the store, and let this slice's math cover it.
                    if constexpr (k + 2 == NUM_SLICES && c == N_SPLIT - 1) tail();
                }
            });
        });
#endif
    };

    int tic = 0, toc = 1;

    // Loop invariant at the top of iteration `tile`:
    //   * LDS buffer `tic` holds K-tile `tile` and every warp has passed a
    //     barrier since it was written;
    //   * buf_a/buf_b hold K-tile `tile+1`, possibly still in flight;
    //   * buffer `toc` was last read during iteration `tile-1`, before that
    //     same barrier, so it is free to overwrite right now.
    //
    // That last clause is what lets this run on one barrier per K-tile instead
    // of the two the CDNA kernel uses: the write-after-read hazard on `toc` is
    // already covered by the previous iteration's barrier, so only the
    // write-before-read hazard needs a fresh one, at the bottom.
    // The K index is clamped rather than predicated. Past the end of the K
    // loop this re-reads the last tile into staging registers that are never
    // stored, which is a handful of L2-hot loads per workgroup -- and it keeps
    // exactly GP batches in flight at all times, which is what makes VM_KEEP a
    // compile-time immediate. Predicating the issue instead would make the
    // right vmcnt a runtime value, and there is no such instruction.
    auto gload = [&](int d, int t) {
        const int tt = t_begin + (t < num_tiles ? t : num_tiles - 1);
        // A is addressed in elements so the M start can be m_base; B still
        // tiles exactly, so it keeps the tile-granular coord.
        G::load_global_to_register_buffer(buf_a[d], stage_a, g.a,
                                          coord<>{0, 0, m_base, tt * K_STEP}, As[0]);
        G::load_global_to_register_buffer(buf_b[d], stage_b, g.b, coord<st_b>{0, 0, col, tt}, Bs[0]);
    };

    G::load(As[tic], g.a, coord<>{0, 0, m_base, t_begin * K_STEP});
    G::load(Bs[tic], g.b, coord<st_b>{0, 0, col, t_begin});
#if !ABLATE_GLOBAL
    static_for<GP>([&](auto d) { gload(d, d + 1); });
#endif
    __builtin_amdgcn_s_barrier();

    // Unrolled by GP so that the staging slot is a compile-time index; a
    // runtime one would put buf_a/buf_b in scratch.
    const int n_main = num_tiles - 1;
    for (int base = 0; base < n_main; base += GP) {
        static_for<GP>([&](auto dd) {
            constexpr int d = dd;
            const int tile = base + d;
            if (tile < n_main) {
                // Land batch d -- issued GP iterations ago -- then immediately
                // refill that slot, so the staging registers are never idle.
                auto store_tile = [&] {
#if !ABLATE_GLOBAL
                    vm_wait<VM_KEEP>();
#if !ABLATE_LDS_WRITE
                    // No drain: dot_tile's own lgkmcnt schedule retires these,
                    // and its last chunk waits to zero, which is what the
                    // barrier below needs.
                    G::template store_register_buffer_to_shared<false>(As[toc], buf_a[d], stage_a);
                    G::template store_register_buffer_to_shared<false>(Bs[toc], buf_b[d], stage_b);
#endif
                    gload(d, tile + 1 + GP);
#endif
                };
                auto nop = [] {};
                if constexpr (C::WRITE_POS == 0) {
                    store_tile();
                    dot_tile(tic, std::false_type{}, nop);
                } else if constexpr (C::WRITE_POS == 1) {
                    dot_tile(tic, std::false_type{}, nop);
                    store_tile();
                    // Same reason as the WRITE_POS == 2 branch below, and here
                    // it is not nearly free: the writes were issued on the line
                    // above, so this really does stall. It is still not
                    // optional. store_register_buffer_to_shared<false> leaves
                    // the ds_writes in flight, nothing after it retires them,
                    // and past the barrier the *other* warps read the half of
                    // As[toc]/Bs[toc] that this one just wrote. Within a wave
                    // LDS is ordered and the bug hides; across the workgroup it
                    // does not. That this path costs a full drain while
                    // WRITE_POS == 2 hides it behind a K-slice of WMMAs is the
                    // entire argument for WRITE_POS == 2 being the default.
                    lds_wait<0>();
                } else {
                    dot_tile(tic, std::true_type{}, store_tile);
                    // dot_tile's last chunk stops at lgkmcnt(TAIL_OPS), so the
                    // writes are still moving. s_barrier does not order memory:
                    // without this the next tile's ds_reads race them. It is
                    // nearly free here -- a whole K-slice of WMMAs has run
                    // since the writes were issued.
                    lds_wait<0>();
                }
#if !ABLATE_BARRIER
                __builtin_amdgcn_s_barrier();
#endif
                tic ^= 1;
                toc ^= 1;
            }
        });
    }

    // Epilogue: last tile is already resident, and there is nothing left to
    // stage -- hence false_type, which zeroes TAIL_OPS.
    dot_tile(tic, std::false_type{}, [] {});

    // Column coords are in units of the tile width, which is now SPLIT_N.
    // Rows are in elements, because m_base need not be a multiple of BLOCK_M.
    const int m_row    = m_base + warp_row * REG_BLOCK_M;   // elements
    const int col_tile = (col * WARP_COLS + warp_col) * N_SPLIT;

    if constexpr (C::SPLIT_K > 1) {
        // fp32 partials, one slice per k-split. Everything downstream -- the
        // reduction, and in the distributed forms the peer push -- happens in
        // the follow-up kernel, because that is the first point at which a
        // column's value is final and therefore the first point at which it is
        // worth sending anywhere.
        const int ldn = g.c.cols();
        float *p = g.ws + ((size_t)blockIdx.y * g.c.rows() + m_row) * ldn;
        #pragma unroll
        for (int s = 0; s < N_SPLIT; s++)
            store_at(p + (col_tile + s) * SPLIT_N, ldn, C_accum[s]);
    } else if constexpr (GL::FUSED_AR) {
        // Fused all-reduce, sharded along N instead of M.
        //
        // The reduce-scatter below shards along M because the framework owns
        // that axis -- it splits by token, and the kernel has to agree. An
        // all-reduce has no such obligation: every rank ends up with the whole
        // M x N, so which axis we shard the *reduction* over is ours to pick,
        // and N is the better pick for two reasons.
        //
        // First, it makes M unconstrained. The M-sharded path needs
        // shard_rows % REG_BLOCK_M, which is what rejects most decode shapes
        // outright -- M=1 cannot be row-sharded across ranks at all. Sharding
        // along N leaves M alone entirely, so the same epilogue serves prefill
        // and decode.
        //
        // Second, for world > 2 it is cheaper. Reducing along N and then
        // gathering moves 2(w-1)/w of C; having every rank push its whole
        // partial to everyone moves (w-1).
        //
        // Each rank keeps the columns it owns and pushes the rest into the
        // owner's inbox, at the slot indexed by *my* rank so that no two
        // senders collide. The owner's slot in its own inbox goes unused; that
        // wastes 1/world of a buffer that is already only the size of C, which
        // is cheaper than the arithmetic to compact it.
        //
        // The ownership test is per *warp tile*, not per N_SPLIT chunk, and
        // that is load-bearing rather than tidiness. Deciding per chunk puts a
        // branch with two store bodies inside the N_SPLIT loop, and on
        // big_config that costs enough registers to matter: 209 VGPRs and no
        // spill becomes 256 VGPRs, 8 VGPR spills and 36 bytes/lane of scratch.
        // The kernel does not survive that. Its LDS pipeline hand-manages
        // s_waitcnt counts -- lds_wait<N>() leaves N operations outstanding on
        // purpose -- and the spill/reload traffic the compiler interleaves is
        // counted by the same hardware counters, so the waits stop meaning what
        // they were written to mean. The symptom was precise and silent: every
        // s == N_SPLIT-1 tile, and only those, came out NaN, on prefill shapes
        // only (the decode configs are far enough from the register ceiling
        // that they never spilled). Keep this branch out of the loop.
        //
        // A warp tile is REG_BLOCK_N contiguous columns starting at a multiple
        // of REG_BLOCK_N, so it lies in one shard as long as the shard width
        // does too. The host checks that; see the constraint block in
        // gemm_ar_mp.hip.
        // The host cannot see which config the dispatcher picked, so it checks
        // the loosest sufficient width. This keeps that honest.
        static_assert(64 % REG_BLOCK_N == 0,
                      "the fused all-reduce host check assumes a warp tile is "
                      "at most 64 columns and divides 64");
        const int n_shard = g.shard_cols;
        const int c_base = col_tile * SPLIT_N;
        const int owner = c_base / n_shard;
        const size_t off = (size_t)m_row * n_shard + (c_base - owner * n_shard);
        if (owner == g.my_rank) {
            bf16 *dst = g.c_own + off;
            #pragma unroll
            for (int s = 0; s < N_SPLIT; s++)
                store_at(dst + s * SPLIT_N, n_shard, C_accum[s]);
        } else {
            // c_sym is already *my slot*; translate() re-points it at the
            // owner without changing which slot it is, exactly as in the RS
            // path. Adding my_rank*inbox_stride here as well would offset it
            // twice -- which is what it did at first, putting rank 1's tiles a
            // slot past where rank 0 looked for them and, at prefill M, a whole
            // output past the end of the inbox.
            bf16 *inbox = g.v.translate(g.c_sym, owner) + off;
            // See the note on the RS push below for why this is _nt.
            #pragma unroll
            for (int s = 0; s < N_SPLIT; s++)
                store_at_nt(inbox + s * SPLIT_N, n_shard, C_accum[s]);
        }
        __threadfence_system();
    } else if constexpr (!GL::FUSED_RS) {
        #pragma unroll
        for (int s = 0; s < N_SPLIT; s++)
            store(g.c, C_accum[s], coord<>{0, 0, m_row, (col_tile + s) * SPLIT_N});
    } else {
        // Fused reduce-scatter. Every rank computes a full M x N partial
        // product over its own slice of K, and the final C is the sum of those
        // partials, row-sharded across ranks. So a tile is either mine to keep
        // or the owner's to receive, and the cheapest moment to decide is here:
        // the accumulator is already in registers, and the alternative is to
        // write the whole M x N locally and move half of it again afterwards.
        //
        // The shard boundary is in units of REG_BLOCK_M and the host has
        // checked it divides, so a warp tile never straddles two owners and no
        // tile has to be split.
        //
        // The destination is chosen per tile, and for world > 2 there are
        // world-1 possible peers -- so it cannot be a gl. kittens::gl has no
        // default constructor and only a __host__ one, so a kernel argument can
        // hold neither an array of them nor one built on the device.
        //
        // sym_view is the way out and costs nothing: it is a trivially copyable
        // table of heap bases, already passed by value to kernels elsewhere, and
        // translate(p, owner) turns my address for a symmetric object into the
        // owner's. Pair it with store_at(), which is store() minus the gl->
        // (pointer, stride) step, and the peer store is still an ordinary
        // vectorized global store that happens to land on another GPU -- no
        // per-element loop, and the same code for every world size.
        //
        // shard_rows is in rows, not tiles, because the tile height is a
        // property of the config the dispatcher picked and the host does not
        // know which one that was. One integer division at the very end of the
        // kernel costs nothing.
        // In elements, so that an M remainder (m_base backed up to M - BLOCK_M)
        // works here too. A warp tile stays inside one shard as long as
        // shard_rows and m_base are both multiples of REG_BLOCK_M; the host
        // checks shard_rows, and m_base inherits it from M % REG_BLOCK_M == 0.
        const int owner     = m_row / g.shard_rows;
        const int local_row = m_row - owner * g.shard_rows;
        if (owner == g.my_rank) {
            #pragma unroll
            for (int s = 0; s < N_SPLIT; s++)
                store(g.c, C_accum[s], coord<>{0, 0, local_row, (col_tile + s) * SPLIT_N});
        } else {
            // Same addressing the gl path does, done by hand: the inbox is
            // (shard_rows, N) row-major, so tile (r, c) starts at
            // r*REG_BLOCK_M*N + c*SPLIT_N and rows are N apart.
            const int ldc = g.c.cols();
            bf16 *inbox = g.v.translate(g.c_sym, owner)
                        + (size_t)local_row * ldc;
            // _nt, not plain store_at: this lands in another GPU's memory,
            // which gfx1100 caches in *my* L2, and no fence a kernel can issue
            // writes that L2 back. Without the bypass bits the tile sits dirty
            // on my side long after the barrier has told the peer to read it.
            #pragma unroll
            for (int s = 0; s < N_SPLIT; s++)
                store_at_nt(inbox + (size_t)(col_tile + s) * C::SPLIT_N, ldc, C_accum[s]);
            // And the stores have to retire before the wave ends, or the flag
            // can still get out first -- the tile goes over PCIe to the peer's
            // HBM while the flag goes to a host page, two paths with nothing
            // ordering them.
            __threadfence_system();
        }
    }
}

// Reduce the split-K partials. Fixed-order sum rather than fp32 atomics: the
// result has to be bit-identical run to run for a framework to accept it.
__global__ void reduce_splitk(bf16 *out, const float *ws, size_t n, int slices) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float acc = 0.f;
        for (int s = 0; s < slices; s++) acc += ws[(size_t)s * n + i];
        out[i] = __float2bfloat16(acc);
    }
}

// The all-reduce form of the above. When the dispatcher picks a split-K config
// the register epilogue cannot push to peers -- it has fp32 partials, not an
// answer -- so the scatter moves here, into the pass that turns the partials
// into one. That is the right place for it anyway: this pass already touches
// every output element exactly once, so the push costs only the difference
// between a local store and a peer store.
//
// Templated on the globals rather than taking the fields one by one because
// sym_view is declared in distributed/ipc_heap.cuh, which includes *this* file
// and so cannot be named from it. A template is only instantiated once GL is
// complete, which is after the include has closed.
template<typename GL>
__global__ void reduce_splitk_scatter_n(GL g, int slices) {
    const int N = g.c.cols(), ns = g.shard_cols;
    const size_t n = (size_t)g.c.rows() * N;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float acc = 0.f;
        for (int s = 0; s < slices; s++) acc += g.ws[(size_t)s * n + i];
        const int row = (int)(i / N), col = (int)(i % N);
        const int owner = col / ns, local_col = col - owner * ns;
        const bf16 v = __float2bfloat16(acc);
        const size_t off = (size_t)row * ns + local_col;
        if (owner == g.my_rank) {
            g.c_own[off] = v;
        } else {
            bf16 *inbox = g.v.translate(g.c_sym, owner) + off;
            // Nontemporal for the same reason the register epilogue's peer
            // store is: this lands in the owner's HBM but gfx1100 caches it in
            // *my* L2, and nothing a kernel can issue writes that L2 back.
            unsigned short w; __builtin_memcpy(&w, &v, sizeof(w));
            __builtin_nontemporal_store(w, reinterpret_cast<unsigned short *>(inbox));
        }
    }
    __threadfence_system();
}

// Scratch for the split-K partials, grown on demand and kept for the process.
// The callers that have their own allocator (the distributed harnesses) fill
// g.ws themselves and never reach this; it exists so the plain python entry
// point, whose signature is just (A, B, C), still works.
static float *splitk_ws(size_t floats) {
    static float *p = nullptr;
    static size_t cap = 0;
    if (floats > cap) {
        if (p) hipFree(p);
        hipMalloc(&p, floats * sizeof(float));
        cap = floats;
    }
    return p;
}

// Scratch for the M-padding path below, same lifetime policy as splitk_ws.
static bf16 *pad_ws(size_t elems) {
    static bf16 *p = nullptr;
    static size_t cap = 0;
    if (elems > cap) {
        if (p) hipFree(p);
        hipMalloc(&p, elems * sizeof(bf16));
        cap = elems;
    }
    return p;
}

template<typename C, typename GL = micro_globals>
static void launch(const GL &g_in) {
    GL g = g_in;
    if constexpr (C::SPLIT_K > 1)
        if (!g.ws)
            g.ws = splitk_ws((size_t)C::SPLIT_K * g.c.rows() * g.c.cols());
    const unsigned long mem_size = C::SHARED_BYTES;
    hipFuncSetAttribute((void*)micro_tk<C, GL>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // The compiler's occupancy remark counts registers only. LDS is the other
    // limit and on this kernel it is usually the binding one, so ask the
    // runtime what actually fits. HK_OCC=1 to see it.
    if (getenv("HK_OCC")) {
        static bool once = false;
        if (!once) {
            once = true;
            int blocks = 0;
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)micro_tk<C, GL>,
                                                         C::NUM_THREADS, mem_size);
            fprintf(stderr, "occ: %d blocks/WGP, %d waves/WGP, %.1f waves/SIMD"
                            " (LDS %lu B, %d threads)\n",
                    blocks, blocks * C::NUM_WARPS, blocks * C::NUM_WARPS / 4.0,
                    mem_size, C::NUM_THREADS);
        }
    }
    // ceil on M to match the kernel's m_blocks; a floor here would leave the
    // tail block unlaunched while the kernel still expects it. The K slices go
    // on grid.y so the L2 swizzle over blockIdx.x is undisturbed.
    const int m_blocks = (g.a.rows() + C::BLOCK_M - 1) / C::BLOCK_M;
    micro_tk<C, GL><<<dim3(m_blocks * (g.b.rows() / C::BLOCK_N), C::SPLIT_K),
                  dim3(C::NUM_THREADS), mem_size, g.stream>>>(g);
    if constexpr (C::SPLIT_K > 1) {
        if constexpr (GL::FUSED_AR) {
            // Same pass, but it scatters instead of storing. See the kernel.
            hipLaunchKernelGGL((reduce_splitk_scatter_n<GL>), dim3(1024),
                               dim3(256), 0, g.stream, g, C::SPLIT_K);
        } else {
            const size_t n = (size_t)g.c.rows() * g.c.cols();
            hipLaunchKernelGGL(reduce_splitk, dim3(1024), dim3(256), 0, g.stream,
                               (bf16*)g.c.raw_ptr, g.ws, n, C::SPLIT_K);
        }
    }
}

// Three tilings. Two are picked on the number of workgroups the M/N grid
// produces; the third only exists to cover a K the other two cannot tile.
//
//   big    128x128x64, 8 warps, 32x64 per warp, N_SPLIT=4. 208 VGPRs, 7
//          waves/SIMD, 48 KB of LDS. The deepest K-tile that fits, which is
//          what the rotated schedule in dot_tile() wants: with four K-slices
//          of four chunks each, fifteen of the sixteen chunks have LDS reads
//          in flight underneath them.
//   small  128x64x64, 8 warps, 32x32 per warp, N_SPLIT=2. 134 VGPRs, 10
//          waves/SIMD, 24 KB. Half the intensity, but twice the workgroups.
//   k32    big with K_STEP=32, for a K that is not a multiple of 64. The
//          kernel has no K remainder handling, so this is correctness, not
//          tuning.
//
// The big/small crossover is a tail-quantization effect, not an inner-loop one.
// `big` uses 48 KB of LDS, so only one of its workgroups fits per WGP and 48
// WGPs hold 48 at once; `small` at 24 KB holds 96. A grid that is a poor fit
// for the coarser tiling idles through a mostly-empty final pass.
//
// Measured, TFLOPs (blocks counted at 128x128):
//
//   M,N,K                blocks | big | small
//   512x512x4096             16 |  19 |  33
//   1024x1024x4096           64 |  49 |  51
//   2048x1024x4096          128 |  63 |  58
//   2048x2048x2048          256 |  64 |  60
//   4096x1024x4096          256 |  65 |  60
//   4096x4096x4096         1024 |  73 |   -
//   8192x8192x4096         4096 |  78 |   -
//
// so the rule is `blocks <= 64`. Note this threshold moved by 4x when the
// rotated schedule landed: overlapping the LDS reads made `big` much better at
// mid-size grids, and it now wins everywhere it has enough workgroups to fill
// the machine once.
using big_config   = config<128, 128, 64, 16, 8, 4, 8, 4>;
using small_config = config<128,  64, 64, 16, 8, 4, 8, 2>;
using k32_config   = config<128, 128, 32, 16, 8, 4, 8, 4>;

// Decode: M is one to a few dozen tokens, so this is not a compute problem at
// all, it is weight streaming. The whole mlp_down WMMA is about 6 us against a
// 114 us floor set by reading the weights, so padding M up to the WMMA's 16
// rows costs nothing measurable -- the padded rows ride in cache lines that had
// to be fetched anyway. What matters is keeping enough of B in flight, hence
// all 8 warps spread along N (WARP_ROWS=1) and the widest BLOCK_N that fits.
//
// K_STEP is 32 rather than 64 because at BLOCK_N=256 a 64-deep B tile is 32 KB,
// and two of those plus A overflow the 64 KB of LDS. REG_BLOCK_N is then 32 and
// N_SPLIT=2 puts SPLIT_N at the 16 the WMMA needs.
//
// Requires A and C to have at least BLOCK_M=16 rows allocated even when M is
// smaller: the epilogue is unpredicated, so it writes whole 16-row tiles. Rows
// past M hold garbage and the caller is expected to slice them off.
// BLOCK_N=128 with 4 warps, measured against 256/8 and 64/4: the wide block
// starves the grid (N/256 is 20 workgroups on 48 CUs) and the narrow one gives
// up too much B reuse.
//
// SPLIT_K then has to differ by layer, which is why there are three of these.
// At N=5120 the M/N grid is 40 workgroups however K is cut, so the split is the
// only source of parallelism -- but each slice also writes a full fp32 partial,
// and that traffic is pure overhead. Short K has too little work per block to
// cover latency and wants the deeper split; long K already covers it and just
// pays. Measured at M=8, N=5120 (GB/s of weight traffic):
//
//            SPLIT_K:   1     2     4     8
//   attn_out K=3072    291   314   342   372
//   mlp_down K=8704    588   631   633   577
//
// Hence: 8 slices below ~128 K-tiles, 4 above.
using thin_sk1_config = config<16, 128, 32, 16, 4, 1, 8, 2, 1, 2, 1>;
using thin_config     = config<16, 128, 32, 16, 4, 1, 8, 2, 1, 2, 4>;
using thin_sk8_config = config<16, 128, 32, 16, 4, 1, 8, 2, 1, 2, 8>;

// BLOCK_M=32, for 16 < M <= 32. BLOCK_M has to cover M: every extra M block
// re-reads the whole of B, and B is the entire cost here. Running M=32 through
// the 16-row config reads the weights twice and measured 200 GB/s.
//
// This one does NOT split K, because the split-K workspace is [SPLIT_K][M][N]
// fp32 and is both written and read, so its traffic scales with M while the
// weight traffic does not. At M=32, K=3072, SPLIT_K=8 that is 10.5 MB of
// workspace against 31 MB of weights -- a 33% surcharge on a kernel that is
// purely bandwidth-bound. Measured at M=32, N=5120 (GB/s of weight traffic):
//
//            SPLIT_K:   1     2     4     8
//   attn_out K=3072    329   300   316   258
//   mlp_down K=8704    628   605   595   458
//
// Monotone the other way from the M=8 table above, and SPLIT_K=1 wins outright.
// WARP_ROWS 1 vs 2 (REG_BLOCK_M 32 vs 16) is a wash once K is not split:
// 324/640 against 329/628, so the simpler one stays.
using thin32_config = config<32, 128, 32, 16, 4, 1, 8, 2, 1, 2, 1>;

#ifndef HK_MULTI_CONFIG
#define HK_MULTI_CONFIG 1
#endif

// The selection rule, templated on the globals type so that a caller with a
// different output policy gets the same tiling decisions. The fused kernel used
// to pin itself to `big`, which measures the wrong thing on the small-M shapes
// a decode or short-prefill batch produces -- the point of comparing against an
// unfused baseline is that the two differ only in the epilogue.
template<typename GL>
static void dispatch_any(const GL &g) {
    // Below one thin block there is nothing to tile with. The kernel always
    // writes whole BLOCK_M-row tiles and loads them without a predicate, so an
    // A and C allocated at exactly M rows would be read and *written* past
    // their ends -- which is how M=1 silently corrupted memory before this. So
    // stage through a padded copy. The copies are M*K and M*N elements, under
    // 150 KB at the decode shapes against ~30 MB of weight traffic, so they do
    // not show up in the measurement; and a caller that already hands us
    // BLOCK_M rows (the torch wrapper does, since it has to pad anyway) never
    // takes this path. Only for the plain output policy: the fused epilogues
    // push to peer inboxes whose geometry the caller owns.
    constexpr int PAD_M = thin_config::BLOCK_M;
    if constexpr (!GL::FUSED_RS && !GL::FUSED_AR) {
        if (g.a.rows() < PAD_M) {
            const int M = g.a.rows(), K = g.a.cols(), N = g.c.cols();
            bf16 *pa = pad_ws((size_t)PAD_M * K + (size_t)PAD_M * N);
            bf16 *pc = pa + (size_t)PAD_M * K;
            hipMemsetAsync(pa + (size_t)M * K, 0,
                           (size_t)(PAD_M - M) * K * sizeof(bf16), g.stream);
            hipMemcpyAsync(pa, g.a.raw_ptr, (size_t)M * K * sizeof(bf16),
                           hipMemcpyDeviceToDevice, g.stream);
            GL gp = g;
            gp.a = kittens::make_gl<_gl_A>((uint64_t)pa, 1, 1, PAD_M, K);
            gp.c = kittens::make_gl<_gl_C>((uint64_t)pc, 1, 1, PAD_M, N);
            dispatch_any(gp);   // gp.a.rows() == PAD_M, so this cannot recurse
            hipMemcpyAsync(g.c.raw_ptr, pc, (size_t)M * N * sizeof(bf16),
                           hipMemcpyDeviceToDevice, g.stream);
            return;
        }
    }
    // Decode: below one full M block the 128-row configs would be mostly
    // padding, and the thin configs exist precisely for this range.
    //
    // Not for the reduce-scatter policy. Its epilogue decides ownership from
    // the row, so it needs M >= world * REG_BLOCK_M to shard at all -- M=1
    // cannot be row-sharded -- and a split-K config would hand it fp32 partials
    // with nothing downstream to reduce them. All-reduce shards along N and has
    // neither problem, which is why decode goes through that one.
    if constexpr (!GL::FUSED_RS)
    if (g.a.rows() < big_config::BLOCK_M &&
        g.b.rows() % thin_config::BLOCK_N == 0 &&
        g.a.cols() % thin_config::K_STEP == 0) {
        const int tiles = g.a.cols() / thin_config::K_STEP;
        const bool deep = tiles <= 128 && tiles % 8 == 0;   // short K wants 8 slices
        const bool quad = tiles % 4 == 0;
        if (g.a.rows() <= thin_config::BLOCK_M) {
            if (deep)      launch<thin_sk8_config>(g);
            else if (quad) launch<thin_config>(g);
            else           launch<thin_sk1_config>(g);
        } else if (g.a.rows() < thin32_config::BLOCK_M) {
            // 17..31. The 32-row config would be the faster one, but the
            // last-block backup that makes M remainders work needs
            // M >= BLOCK_M: below that m_base pins to 0 and the kernel writes
            // a whole 32-row tile into an M-row buffer, past its end. The
            // 16-row config has two blocks here and keeps m_base in
            // {0, M-16}, both in range. (M < 16 never reaches this branch;
            // the padded staging above catches it.)
            launch<thin_sk1_config>(g);
        } else {
            launch<thin32_config>(g);   // split-K does not pay at M > 16
        }
        return;
    }
    // num_tiles is a plain division, so a K-tile that does not divide K would
    // silently drop the tail.
    if (g.a.cols() % big_config::K_STEP != 0) {
        launch<k32_config>(g);
        return;
    }
    const int blocks = ((g.a.rows() + big_config::BLOCK_M - 1) / big_config::BLOCK_M)
                     * (g.b.rows() / big_config::BLOCK_N);
    // Fall back to `big` on shapes `small` cannot tile; its BLOCK_N is the
    // smaller of the two, so this only triggers on N not divisible by 128.
    if (blocks <= 64 && g.b.rows() % small_config::BLOCK_N == 0) {
        launch<small_config>(g);
        return;
    }
    launch<big_config>(g);
}

void dispatch_micro(micro_globals g) {
#if HK_MULTI_CONFIG
    dispatch_any(g);
#else
    launch<macro_config>(g);   // -DBLOCK_M=... from sweep.sh
#endif
}

// The distributed kernel includes this file for the tuned inner loop and
// supplies its own host harness, so the python module is opt-out.
#ifndef HK_GEMM_NO_PYBIND
PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}
#endif
