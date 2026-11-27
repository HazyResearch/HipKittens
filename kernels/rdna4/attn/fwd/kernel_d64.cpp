#include "kittens.cuh"


#ifndef ATTN_B
constexpr int ATTN_B = 16; // batch size
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 64; // number of heads
#endif

#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 8; // number of heads for key and value
#endif

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV; // queries per KV head group

#ifndef ATTN_N
constexpr int ATTN_N = 1024; // sequence length
#endif

constexpr int ATTN_D = 64; // head dimension: this file is the d=64 kernel

// Block geometry. Every register tile scales with these (Q = Q_BLOCK_SIZE,
// KV = KV_BLOCK_SIZE), and gfx1201 has 256 VGPRs and no AGPRs:
//   att_block[2]      2 * KV*Q   fp32
//   o_reg                  D*Q   fp32
//   v_reg, k_reg_t        KV*D   bf16   (k_reg_t is D*KV)
//   att_block_bf16        KV*Q   bf16
// Q = 32 no longer fits and spills. The PV stage below is written for four
// 16-row V subtiles, so KV is fixed at 64.
constexpr int Q_BLOCK_SIZE = 16;
constexpr int KV_BLOCK_SIZE = 64;

// Row stride of K/V in global, in elements. The tensors are (B, N, H_KV, D), so
// stepping one position along N skips every KV head: H_KV*D, not D.
constexpr int KV_ROW_STRIDE = ATTN_H_KV * ATTN_D;

constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = kittens::WARP_THREADS * NUM_WARPS;

#define MFMA_MASK 0x08
#define VALU_MASK 0x02
#define EXP_MASK  0x400

using namespace kittens;

// RDNA4 has only v_wmma_f32_16x16x16, and mma_{AB,ABt,AtB}_base require all
// three operand shapes to be rt_16x16. Every register tile here is therefore
// built from 16x16 base tiles; the K-loop trip counts follow from rt::width /
// rt::height.
using S16 = ducks::rt_shape::rt_16x16;

// Fast fp32 -> bf16 pair, round-to-nearest-even, WITHOUT NaN handling.
//
// gfx1201 has no f32->bf16 instruction (v_cvt_pk_bf16_f32 is gfx950 / gfx1250
// only), so HIP synthesises one: 13 instructions per pair, 8 of them NaN
// propagation. This keeps the RNE rounding in 5 (two v_bfe_u32, two v_add3_u32,
// one v_perm_b32). It is safe here and only here: the only thing narrowed with it
// is P = exp2(s - max), which is finite and in (0, 1] by construction, so it is
// deliberately not installed in base_types.cuh's convertor, which the GEMMs share.
__device__ __forceinline__ bf16_2 f32x2_to_bf16x2_fast(const float2 &u) {
    uint32_t x, y;
    __builtin_memcpy(&x, &u.x, 4);
    __builtin_memcpy(&y, &u.y, 4);
    x += 0x7fffu + ((x >> 16) & 1u);      // RNE
    y += 0x7fffu + ((y >> 16) & 1u);
    uint32_t r = (y & 0xffff0000u) | (x >> 16);
    bf16_2 o; __builtin_memcpy(&o, &r, 4); return o;
}

// Shims over the tile ops: constraint checking is skipped for dependent calls
// inside templates, so zero/copy/... report several viable candidates. Dispatching
// with `if constexpr` on the concept never enters overload resolution.
namespace hk {
template<class T> __device__ inline void zero(T& d) {
  if constexpr (kittens::ducks::rv::all<T>) kittens::unary_op<kittens::base_ops::zero, T>(d, d);
  else                                      kittens::unary_map<kittens::base_ops::zero, T>(d, d);
}
template<class T> __device__ inline void ones(T& d) {
  if constexpr (kittens::ducks::rv::all<T>) kittens::unary_op<kittens::base_ops::ones, T>(d, d);
  else                                      kittens::unary_map<kittens::base_ops::ones, T>(d, d);
}
template<class T, class U> __device__ inline void copy(T& d, const U& s) {
  if constexpr (kittens::ducks::rv::all<T>) {
    kittens::bin_op<kittens::base_ops::copy2, T>(d, d, s);
  } else if constexpr (std::is_same_v<T, U>) {
    kittens::bin_map<kittens::base_ops::copy2, T>(d, d, s);
  } else {
    // Cross-type tile copy (the fp32 softmax result narrowed to bf16). bin_map
    // requires both operands to be the same T, so this converts element-wise.
    static_assert(T::height == U::height && T::width == U::width &&
                  T::packed_per_base_tile == U::packed_per_base_tile,
                  "hk::copy: cross-type tiles must share a base-tile decomposition");
    using DT = typename T::dtype; using ST = typename U::dtype;
    #pragma unroll
    for (int i = 0; i < T::height; i++)
      #pragma unroll
      for (int j = 0; j < T::width; j++)
        #pragma unroll
        for (int k = 0; k < T::packed_per_base_tile; k++)
          if constexpr (std::is_same_v<DT, kittens::bf16_2> && std::is_same_v<ST, float2>)
              d.tiles[i][j].data[k] = f32x2_to_bf16x2_fast(s.tiles[i][j].data[k]);
          else
              d.tiles[i][j].data[k] =
                  kittens::base_types::convertor<DT, ST>::convert(s.tiles[i][j].data[k]);
  }
}
template<class T> __device__ inline void exp2(T& d, const T& s) {
  if constexpr (kittens::ducks::rv::all<T>) {
    kittens::unary_op<kittens::base_ops::exp2, T>(d, s);
  } else if constexpr (kittens::ducks::rt_base::all<T>) {
    // A single base tile (unary_map takes only whole rts): exp2_half below
    // exponentiates base tiles one at a time so they interleave with WMMA issue.
    #pragma unroll
    for (int k = 0; k < T::packed_per_thread; k++)
      d.data[k] = kittens::base_ops::exp2::op(s.data[k]);
  } else {
    kittens::unary_map<kittens::base_ops::exp2, T>(d, s);
  }
}
template<class T, class U> __device__ inline void sub(T& d, const T& l, const U& r) {
  if constexpr (kittens::ducks::rv::all<T>) kittens::bin_op<kittens::base_ops::sub, T>(d, l, r);
  else                                      kittens::bin_map<kittens::base_ops::sub, T>(d, l, r);
}
template<class T, class U> __device__ inline void mul(T& d, const T& l, const U& r) {
  if constexpr (kittens::ducks::rv::all<T>) kittens::bin_op<kittens::base_ops::mul, T>(d, l, r);
  else                                      kittens::bin_map<kittens::base_ops::mul, T>(d, l, r);
}
template<class T> __device__ inline void log(T& d, const T& s) {
  if constexpr (kittens::ducks::rv::all<T>) kittens::unary_op<kittens::base_ops::log, T>(d, s);
  else                                      kittens::unary_map<kittens::base_ops::log, T>(d, s);
}
template<class T, class U> __device__ inline void add(T& d, const T& l, const U& r) {
  if constexpr (kittens::ducks::rv::all<T>) kittens::bin_op<kittens::base_ops::sum, T>(d, l, r);
  else                                      kittens::bin_map<kittens::base_ops::sum, T>(d, l, r);
}
} // namespace hk

// exp2 over half an attention tile, by base-tile row band: HALF=0 covers
// base-tile rows [0, H/2), HALF=1 covers [H/2, H). The two halves are issued in
// different pipeline phases so the exponentials interleave with WMMA issue.
template<int HALF, typename T>
__device__ inline void exp2_half(T &a) {
    #pragma unroll
    for (int i = HALF * (T::height / 2); i < (HALF + 1) * (T::height / 2); i++)
        #pragma unroll
        for (int j = 0; j < T::width; j++)
            hk::exp2(a.tiles[i][j], a.tiles[i][j]);
}

// Direct global -> col-layout-fragment load for the QK operands.
//
// A WMMA operand fragment wants 8 K-consecutive elements per lane. For BOTH QK operands the reduction dim is D,
// and Q/K store D fastest, so the col-layout fragment is already 16 contiguous
// bytes -- no LDS, no transpose, no swizzle. Only V has its reduction dim
// strided, which is why V alone needs global_load_tr_b128.
//
// For dst = rt<bf16, D, S, col> (rows = D = reduction, cols = S = seq):
//     lane L (r = L%16, g2 = L/16), tile (k, n), element e:
//         value = M[16n + r][16k + 8*g2 + e]
// every address 16 B aligned. `scale` is applied in fp32 before the single bf16
// rounding.
template<typename RT>
__device__ inline void load_t_direct(RT &dst, const bf16 *__restrict__ m, int ld,
                                     float scale = 1.0f) {
    const int lane = threadIdx.x & 31;
    const int r = lane % 16, g2 = lane / 16;
    #pragma unroll
    for (int k = 0; k < RT::height; k++) {
        #pragma unroll
        for (int n = 0; n < RT::width; n++) {
            const bf16 *pp = m + (size_t)(16 * n + r) * ld + 16 * k + 8 * g2;
            bf16 v[8];
            __builtin_memcpy(v, pp, sizeof(v));
            if (scale != 1.0f) {
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    v[e] = base_types::convertor<bf16, float>::convert(
                               base_types::convertor<float, bf16>::convert(v[e]) * scale);
            }
            __builtin_memcpy(&dst.tiles[k][n].data[0], v, sizeof(v));
        }
    }
}

#define LOAD_KT(blk) load_t_direct(k_reg_transposed, \
    k_base + (size_t)(blk) * KV_BLOCK_SIZE * KV_ROW_STRIDE, KV_ROW_STRIDE)

typedef __bf16 bf16x8_tr __attribute__((ext_vector_type(8)));

// V^T fragment load. gfx1201 has no ds_read_*_tr_* instructions, so V cannot
// be transposed on the way out of LDS; global_load_tr_b128 does the transpose in
// the memory pipe instead.
//
// Each lane supplies its own address, and lane 8g+c receives element c of each
// of the 8 lanes' 16-byte fetches. Landing the A-operand form for V^T (lane
// carries d, element carries kv) needs lane L (g=L/8, c=L%8) to fetch
//
//     V[16*kk + 8*(g/2) + c][16*n + 8*(g%2)]
//
// every address 16 B aligned. EXEC must be all ones: GLOBAL_LOAD_TR_B128 is
// undefined otherwise.
template<typename RT>
__device__ inline void load_v_tr(RT &dst, const bf16 *__restrict__ v, int ld) {
    const int lane = threadIdx.x & 31;
    const int g = lane >> 3, c = lane & 7;
    #pragma unroll
    for (int kk = 0; kk < RT::height; kk++) {
        #pragma unroll
        for (int n = 0; n < RT::width; n++) {
            const bf16 *p = v + (16 * kk + 8 * (g >> 1) + c) * (size_t)ld
                              + (16 * n  + 8 * (g &  1));
            bf16x8_tr tv = __builtin_amdgcn_global_load_tr_b128_v8bf16(
                (__attribute__((address_space(1))) bf16x8_tr *)p);
            __builtin_memcpy(&dst.tiles[kk][n].data[0], &tv, sizeof(tv));
        }
    }
}

// O store, straight from the accumulator tile: no transpose, no LDS.
//
// The WMMA accumulator layout is
//
//     D[row][col] -> lane 16*(row/8) + col, element row%8
//
// For o_reg (rt<float, D, Q_BLOCK_SIZE, col>) row = d and col = q, so
//     lane L, base tile (i,j), element e:  d = 16i + 8*(L/16) + e
//                                          q = 16j + L%16
// d is contiguous in e and O is (B,N,H,D) with d fastest, so each lane's 8
// elements are one 16-byte store, with no transpose in between.
template<typename RT>
__device__ inline void store_o_acc(bf16 *__restrict__ o, const RT &src, int ld) {
    const int lane = threadIdx.x & 31;
    const int dhi = 8 * (lane / 16), q = lane % 16;
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            const float *f = reinterpret_cast<const float *>(&src.tiles[i][j].data[0]);
            bf16 out[8];
            #pragma unroll
            for (int e = 0; e < 8; e++)
                out[e] = base_types::convertor<bf16, float>::convert(f[e]);
            __builtin_memcpy(o + (size_t)(16 * j + q) * ld + (16 * i + dhi), out, sizeof(out));
        }
}

#define LOAD_V(blk) load_v_tr(v_reg, v_base + (size_t)(blk) * KV_BLOCK_SIZE * KV_ROW_STRIDE, KV_ROW_STRIDE)

using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

#define SCHED_BARRIER(mask, cnt, group) __builtin_amdgcn_sched_group_barrier(mask, cnt, group)

template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

template<int Pairs, int EXP_CNT, int Group>
__device__ __forceinline__ void sched_barrier_exp_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(EXP_MASK, EXP_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_exp_pairs<Pairs - 1, EXP_CNT, Group>();
}


// Lazy rescale: O and the row sum are rescaled only when the running row max
// rises by more than this (log2 units) somewhere in the wave.
constexpr float RESCALE_THRESHOLD_C = 8.0f;  // https://github.com/Dao-AILab/flash-attention/blob/add175637c5d54b74bc25372e49ce282d6f236fc/flash_attn/cute/flash_fwd_sm100.py#L1229

template<typename RV>
__device__ __forceinline__ int rv_all_below(const RV& prev, const RV& cur, float T) {
    int ok = 1;
    #pragma unroll
    for (int o = 0; o < RV::outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < RV::inner_dim; ++i) {
            ok &= (float(cur.data[o][i]) - float(prev.data[o][i]) <= T);
        }
    }
    return ok;  // 1 or 0, per lane
}
// Wave-uniform: true iff EVERY lane says "all below"
__device__ __forceinline__ int wave_all_ok(int lane_ok) {
    return __all(lane_ok);  // 1 fast vote; no ds_bpermute ladder
}

template<int D, typename T=bf16, typename L=row_l, typename S=S16> using qo_tile = rt<T, Q_BLOCK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=S16> using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=S16> using kv_tile = rt<T, KV_BLOCK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=S16> using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L, S>;
template<int D, typename T=float, typename L=col_l, typename S=S16> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>;

template<int D> struct attn_globals {
    _gl_QKVO Qg, Kg, Vg, Og;
    gl<float, -1, -1, -1, -1> L_vec;
    hipStream_t stream;
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }  // no LDS: K/Q/V all go straight to registers
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {

    // Consecutive workgroups go to different KV heads.
    const int head_idx = (blockIdx.x % ATTN_H_KV) * GROUP_SIZE + (blockIdx.x / ATTN_H_KV);
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    // K/V base pointers for this (batch, kv head).
    const bf16* k_base = (bf16*)&g.Kg[{batch_idx, 0, head_idx_kv, 0}];
    const bf16* v_base = (bf16*)&g.Vg[{batch_idx, 0, head_idx_kv, 0}];

    const int num_tiles = ATTN_N / KV_BLOCK_SIZE;

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qo_tile_transposed<D, bf16> q_reg_transposed;
    kv_tile_transposed<D, bf16> k_reg_transposed;

    kv_tile<D, bf16, col_l, S16> v_reg;
    qo_tile_transposed<D, float, col_l, S16> o_reg; // Output tile.
    attn_tile<D, float, col_l, S16> att_block[2]; // attention tile, in float.
    attn_tile<D, bf16, col_l, S16> att_block_bf16;
    typename attn_tile<D, float, col_l, S16>::row_vec max_vec, norm_vec, max_vec_prev, scale_vec;

    hk::zero(o_reg);
    hk::zero(norm_vec);
    hk::ones(scale_vec);

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);

    load_t_direct(q_reg_transposed,
                  (const bf16*)&g.Qg[{batch_idx, tile_idx * Q_BLOCK_SIZE, head_idx, 0}],
                  ATTN_H * ATTN_D, TEMPERATURE_SCALE);

    // Prologue: K0, QK0 and the first half of its softmax, then K1.
    LOAD_KT(0);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);

    hk::zero(att_block[0]);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);

    col_max(max_vec, att_block[0]);
    hk::copy(max_vec_prev, max_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2_half<0>(att_block[0]);
    __builtin_amdgcn_sched_barrier(0);

    // Waves 4-7 take a branch with no work in it. It stays because it is a
    // scheduling-region boundary: without it the compiler reorders the prologue.
    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
    }

    __builtin_amdgcn_sched_barrier(0);
    LOAD_KT(1);
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);

    int pending_scale = 0;


    // hot loop
    #pragma unroll 2
    for (int j = 3; j < num_tiles - 1; j += 2) {
        // Cluster 0:
        //      QK1
        hk::zero(att_block[1]);
        if (pending_scale) {
            hk::mul(norm_vec, norm_vec, scale_vec);
        }
        mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
        hk::copy(subtile_inplace<32>(att_block_bf16, 0), subtile_inplace<32>(att_block[0], 0));
        //      Finish softmax for QK0
        exp2_half<1>(att_block[0]);
        col_sum(norm_vec, att_block[0], norm_vec);
        hk::copy(subtile_inplace<32>(att_block_bf16, 1), subtile_inplace<32>(att_block[0], 1));
        sched_barrier_exp_pairs<6, 3, 1>();
        sched_barrier_pairs<10, 5, 1>();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1:
        //      Load V0 into registers
        LOAD_V(j - 3);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2:
        //      A0V0
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 0), subtile_inplace<16>(att_block_bf16, 0), o_reg);
        //      Partial softmax for QK1
        col_max(max_vec, att_block[1], max_vec_prev);
        sched_barrier_pairs<4, 5, 2>();
        int lane_ok = rv_all_below(max_vec_prev, max_vec, RESCALE_THRESHOLD_C);
        int all_ok  = wave_all_ok(lane_ok);
        if (__builtin_expect(all_ok, 1)) {
            hk::copy(max_vec, max_vec_prev);
            pending_scale = 0;
        } else {
            hk::sub(scale_vec, max_vec_prev, max_vec);   // diff = prev - new
            hk::exp2(scale_vec, scale_vec);
            mul_col(o_reg, o_reg, scale_vec);
            hk::copy(max_vec_prev, max_vec);
            pending_scale = 1;
        }
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 1), subtile_inplace<16>(att_block_bf16, 1), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 2), subtile_inplace<16>(att_block_bf16, 2), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 3), subtile_inplace<16>(att_block_bf16, 3), o_reg);
        sub_col(att_block[1], att_block[1], max_vec);
        exp2_half<0>(att_block[1]);
        sched_barrier_pairs<6, 5, 2>();
        sched_barrier_exp_pairs<6, 3, 2>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3:
        //      Load K2 into registers
        LOAD_KT(j - 1);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4:
        //      QK2
        hk::zero(att_block[0]);
        if (pending_scale) {
            hk::mul(norm_vec, norm_vec, scale_vec);
        }
        mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
        //      Finish softmax for QK1
        exp2_half<1>(att_block[1]);
        col_sum(norm_vec, att_block[1], norm_vec);
        hk::copy(att_block_bf16, att_block[1]);
        sched_barrier_exp_pairs<6, 3, 3>();
        sched_barrier_pairs<10, 5, 3>();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5:
        //      Load V1 into registers
        LOAD_V(j - 2);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6:
        //      A1V1
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 0), subtile_inplace<16>(att_block_bf16, 0), o_reg);
        //      Partial softmax for QK2
        col_max(max_vec, att_block[0], max_vec_prev);
        sched_barrier_pairs<4, 5, 4>();
        lane_ok = rv_all_below(max_vec_prev, max_vec, RESCALE_THRESHOLD_C);
        all_ok  = wave_all_ok(lane_ok);
        if (__builtin_expect(all_ok, 1)) {
            hk::copy(max_vec, max_vec_prev);
            pending_scale = 0;
        } else {
            hk::sub(scale_vec, max_vec_prev, max_vec);   // diff = prev - new
            hk::exp2(scale_vec, scale_vec);
            mul_col(o_reg, o_reg, scale_vec);
            hk::copy(max_vec_prev, max_vec);
            pending_scale = 1;
        }
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 1), subtile_inplace<16>(att_block_bf16, 1), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 2), subtile_inplace<16>(att_block_bf16, 2), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 3), subtile_inplace<16>(att_block_bf16, 3), o_reg);
        sub_col(att_block[0], att_block[0], max_vec);
        exp2_half<0>(att_block[0]);
        sched_barrier_pairs<6, 5, 4>();
        sched_barrier_exp_pairs<6, 3, 4>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7:
        //      Load K3 into registers
        LOAD_KT(j);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0:
    //      QK3
    hk::zero(att_block[1]);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK2
    exp2_half<1>(att_block[0]);
    hk::mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[0], norm_vec);
    hk::copy(att_block_bf16, att_block[0]);
    sched_barrier_exp_pairs<6, 3, 5>();
    sched_barrier_pairs<10, 5, 5>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1:
    //      Load V2 into registers
    LOAD_V(num_tiles - 4);
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2:
    //      A2V2
    __builtin_amdgcn_s_setprio(1);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    //      Partial softmax for QK3
    col_max(max_vec, att_block[1], max_vec_prev);
    hk::sub(scale_vec, max_vec_prev, max_vec);
    hk::copy(max_vec_prev, max_vec);
    hk::exp2(scale_vec, scale_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2_half<0>(att_block[1]);
    sched_barrier_pairs<10, 5, 6>();
    sched_barrier_exp_pairs<6, 3, 6>();
    __builtin_amdgcn_sched_barrier(0);
    mul_col(o_reg, o_reg, scale_vec);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3:
    //      Load K4 into registers
    LOAD_KT(num_tiles - 2);
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4:
    //      QK4
    hk::zero(att_block[0]);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
    //      Finish softmax for QK3
    exp2_half<1>(att_block[1]);
    hk::mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[1], norm_vec);
    hk::copy(att_block_bf16, att_block[1]);
    sched_barrier_exp_pairs<6, 3, 7>();
    sched_barrier_pairs<10, 5, 7>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5:
    //      Load V3 into registers
    LOAD_V(num_tiles - 3);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 6:
    //      A3V3
    __builtin_amdgcn_s_setprio(1);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    //      Partial softmax for QK4
    col_max(max_vec, att_block[0], max_vec_prev);
    hk::sub(scale_vec, max_vec_prev, max_vec);
    hk::copy(max_vec_prev, max_vec);
    hk::exp2(scale_vec, scale_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2_half<0>(att_block[0]);
    sched_barrier_pairs<10, 5, 8>();
    sched_barrier_exp_pairs<6, 3, 8>();
    __builtin_amdgcn_sched_barrier(0);
    mul_col(o_reg, o_reg, scale_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7:
    //      Load K5 into registers
    LOAD_KT(num_tiles - 1);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8:
    //      QK5
    hk::zero(att_block[1]);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK4
    exp2_half<1>(att_block[0]);
    hk::mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[0], norm_vec);
    hk::copy(att_block_bf16, att_block[0]);
    sched_barrier_exp_pairs<6, 3, 9>();
    sched_barrier_pairs<10, 5, 9>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 9:
    //      Load V4 into registers
    LOAD_V(num_tiles - 2);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 10:
    //      A4V4
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    //      Full softmax for QK5
    col_max(max_vec, att_block[1], max_vec_prev);
    hk::sub(scale_vec, max_vec_prev, max_vec);
    hk::copy(max_vec_prev, max_vec);
    hk::exp2(scale_vec, scale_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2_half<0>(att_block[1]);
    sched_barrier_pairs<10, 5, 10>();
    sched_barrier_exp_pairs<6, 3, 10>();
    __builtin_amdgcn_sched_barrier(0);

    exp2_half<1>(att_block[1]);
    hk::mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[1], norm_vec);
    hk::copy(att_block_bf16, att_block[1]);
    __builtin_amdgcn_sched_barrier(0);
    mul_col(o_reg, o_reg, scale_vec);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 11:
    //      Load V5 into registers
    LOAD_V(num_tiles - 1);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12:
    //      A5V5
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    div_col(o_reg, o_reg, norm_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    store_o_acc((bf16*)&g.Og[{batch_idx, tile_idx * Q_BLOCK_SIZE, head_idx, 0}],
                o_reg, ATTN_H * ATTN_D);

    // multiply by ln(2)
    hk::mul(max_vec, max_vec, 0.69314718056f);
    hk::log(norm_vec, norm_vec);
    hk::add(norm_vec, norm_vec, max_vec);
    store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
}

template<int D>
void dispatch_micro(attn_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<D><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}


template __global__ void attend_ker<64>(const attn_globals<64>);

// ===========================================================================
// HARNESS_MAIN — CPU-referenced correctness gate + timing.
//
// Tensor layout, fixed here by construction so the kernel's coords have a
// single source of truth:
//     Q, O   (B, N, H,    D)
//     K, V   (B, N, H_KV, D)
//     L_vec  (B, H, 1, N)
// This is what the kernel's index math assumes: v_base = &Vg[{b,0,h_kv,0}]
// walks N with stride H_KV*D.
//
// The reference is exact softmax attention in fp32 from bf16-rounded inputs.
// The kernel is online-softmax with bf16 P, so the two differ by bf16 rounding
// on P and by summation order; the tolerance below is set for that, not tight.
// ===========================================================================
#ifdef HARNESS_MAIN
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

static inline float bf16_rne(float x) {
    uint32_t u; __builtin_memcpy(&u, &x, 4);
    uint32_t r = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    float o; __builtin_memcpy(&o, &r, 4); return o;
}
static inline uint16_t to_bf16(float x) {
    uint32_t u; __builtin_memcpy(&u, &x, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static inline float from_bf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float o; __builtin_memcpy(&o, &u, 4); return o;
}

int main(int argc, char **argv) {
    const int iters  = argc > 1 ? atoi(argv[1]) : 0;   // 0 = verify only
    const int verify = argc > 2 ? atoi(argv[2]) : 1;
    // Correctness modes. Each feeds inputs that isolate one part of the kernel.
    //   0: random Q, K, V.
    //   1: Q = 0, so P is exactly uniform and O = mean over kv of V: the V load
    //      and the PV GEMM, without the softmax or the QK GEMM.
    //   2: Q = 0, V[k][d] = k%64: O reveals which kv rows were read.
    //   3: Q = 0, V[k][d] = d: O[q][d] must equal d for any correct normalisation,
    //      whatever kv rows were read. Isolates the d/lane mapping and the
    //      softmax denominator.
    //   4: random Q, K (so the running max changes and the lazy rescale fires),
    //      V[k][d] = d: O[q][d] = d only if O and the row sum are rescaled
    //      consistently.
    //   5: random Q, K, V[k][d] = k/64: fails only if P is paired with the wrong
    //      kv block.
    //   6: random Q, K, V[k][d] = k%64: fails only if P is paired with the wrong
    //      row inside a block.
    const int mode   = argc > 3 ? atoi(argv[3]) : 0;
    const int B = ATTN_B, H = ATTN_H, HKV = ATTN_H_KV, N = ATTN_N, Dh = ATTN_D;
    const size_t nQ = (size_t)B*N*H*Dh, nKV = (size_t)B*N*HKV*Dh, nL = (size_t)B*H*N;

    printf("attn fwd gfx1201  B=%d H=%d H_KV=%d N=%d D=%d  bf16 non-causal\n",
           B, H, HKV, N, Dh);

    std::vector<float> Qf(nQ), Kf(nKV), Vf(nKV);
    srand(1234);
    auto rnd = [&]{ return bf16_rne((rand()/(float)RAND_MAX)*2.f - 1.f); };
    for (auto &x : Qf) x = rnd();
    for (auto &x : Kf) x = rnd();
    for (auto &x : Vf) x = rnd();
    if (mode == 1) std::fill(Qf.begin(), Qf.end(), 0.f);
    if (mode == 3) {
        std::fill(Qf.begin(), Qf.end(), 0.f);
        for (int b=0;b<B;b++) for (int k=0;k<N;k++) for (int h=0;h<HKV;h++)
          for (int i=0;i<Dh;i++)
            Vf[((size_t)b*N+k)*HKV*Dh + h*Dh + i] = (float)i;
    }
    if (mode == 4) {
        for (int b=0;b<B;b++) for (int k=0;k<N;k++) for (int h=0;h<HKV;h++)
          for (int i=0;i<Dh;i++)
            Vf[((size_t)b*N+k)*HKV*Dh + h*Dh + i] = (float)i;
    }
    if (mode == 5 || mode == 6) {
        for (int b=0;b<B;b++) for (int k=0;k<N;k++) for (int h=0;h<HKV;h++)
          for (int i=0;i<Dh;i++)
            Vf[((size_t)b*N+k)*HKV*Dh + h*Dh + i] =
                bf16_rne((float)(mode == 5 ? k/64 : k%64));
    }
    if (mode == 2) {
        std::fill(Qf.begin(), Qf.end(), 0.f);
        for (int b=0;b<B;b++) for (int k=0;k<N;k++) for (int h=0;h<HKV;h++)
          for (int i=0;i<Dh;i++)
            Vf[((size_t)b*N+k)*HKV*Dh + h*Dh + i] = bf16_rne((float)(k % 64));
    }

    std::vector<uint16_t> Qh(nQ), Kh(nKV), Vh(nKV);
    for (size_t i=0;i<nQ;i++)  Qh[i] = to_bf16(Qf[i]);
    for (size_t i=0;i<nKV;i++){Kh[i] = to_bf16(Kf[i]); Vh[i] = to_bf16(Vf[i]);}

    void *dQ,*dK,*dV,*dO,*dL;
    hipMalloc(&dQ, nQ*2);  hipMalloc(&dK, nKV*2); hipMalloc(&dV, nKV*2);
    hipMalloc(&dO, nQ*2);  hipMalloc(&dL, nL*4);
    hipMemcpy(dQ,Qh.data(),nQ*2,hipMemcpyHostToDevice);
    hipMemcpy(dK,Kh.data(),nKV*2,hipMemcpyHostToDevice);
    hipMemcpy(dV,Vh.data(),nKV*2,hipMemcpyHostToDevice);
    hipMemset(dO,0,nQ*2); hipMemset(dL,0,nL*4);

    attn_globals<ATTN_D> g{
        _gl_QKVO((bf16*)dQ, B, N, H,   Dh),
        _gl_QKVO((bf16*)dK, B, N, HKV, Dh),
        _gl_QKVO((bf16*)dV, B, N, HKV, Dh),
        _gl_QKVO((bf16*)dO, B, N, H,   Dh),
        gl<float,-1,-1,-1,-1>((float*)dL, B, H, 1, N),
        0
    };

    dispatch_micro<ATTN_D>(g);
    hipError_t le = hipGetLastError();      // launch-time errors land HERE, not in sync
    if (le != hipSuccess) { printf("  LAUNCH FAILED: %s\n", hipGetErrorString(le)); return 2; }
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("  EXEC FAILED: %s\n", hipGetErrorString(e)); return 2; }

    if (verify) {
        std::vector<uint16_t> Oh(nQ);
        hipMemcpy(Oh.data(), dO, nQ*2, hipMemcpyDeviceToHost);
        const float scale = 1.0f / sqrtf((float)Dh);
        // Model the kernel's Q path: it scales Q in fp32 by (1/sqrt(D))*log2(e)
        // and rounds the product to bf16 before the GEMM. QS holds that; dividing
        // the dot product by log2(e) puts the scores back in natural-log units.
        const float TSCALE  = 0.125f * 1.44269504089f;
        const float INV_L2E = 1.0f / 1.44269504089f;
        std::vector<float> QS(nQ);
        for (size_t i = 0; i < nQ; i++) QS[i] = bf16_rne(Qf[i] * TSCALE);
        long bad = 0, checked = 0; double maxerr = 0, sumerr = 0, sumref = 0, sumsign = 0;
        // Sample (b,h) pairs so the reference stays cheap at full size; every
        // q and d within a sampled pair is checked.
        const int bstep = std::max(1, B/2), hstep = std::max(1, H/4);
        #pragma omp parallel for collapse(2) reduction(+:bad,checked,sumerr,sumref,sumsign) reduction(max:maxerr)
        for (int b = 0; b < B; b += bstep)
        for (int h = 0; h < H; h += hstep) {
            const int hkv = h / (H/HKV);
            std::vector<float> s(N), acc(Dh);
            for (int q = 0; q < N; q++) {
                float m = -INFINITY;
                for (int k = 0; k < N; k++) {
                    float d = 0;
                    for (int i = 0; i < Dh; i++)
                        d += QS[((size_t)b*N+q)*H*Dh + h*Dh + i]
                           * Kf[((size_t)b*N+k)*HKV*Dh + hkv*Dh + i];
                    s[k] = d*INV_L2E; m = fmaxf(m, s[k]);
                }
                float sum = 0;
                for (int k = 0; k < N; k++) { s[k] = expf(s[k]-m); sum += s[k]; }
                std::fill(acc.begin(), acc.end(), 0.f);
                for (int k = 0; k < N; k++) {
                    float p = bf16_rne(s[k]/sum);
                    for (int i = 0; i < Dh; i++)
                        acc[i] += p * Vf[((size_t)b*N+k)*HKV*Dh + hkv*Dh + i];
                }
                for (int i = 0; i < Dh; i++) {
                    float got = from_bf16(Oh[((size_t)b*N+q)*H*Dh + h*Dh + i]);
                    float err = fabsf(got - acc[i]);
                    if (!(err <= 3e-2f*fmaxf(1.f, fabsf(acc[i])))) bad++;
                    checked++; sumerr += err; if (err > maxerr) maxerr = err;
                    sumref += fabsf(acc[i]); sumsign += (got - acc[i]);
                }
            }
        }
        printf("  max_abs_err=%.5f  mean_abs_err=%.5f  bad=%ld/%ld   %s\n",
               maxerr, sumerr/checked, bad, checked, bad ? "*** FAIL ***" : "PASS");
        // A nonzero mean signed error points at a systematic fault rather than
        // rounding, which is zero-mean.
        printf("  mean|O|=%.5f  -> mean_rel_err=%.3f%%   mean_SIGNED_err=%.3e (bias)\n",
               sumref/checked, 100.0*(sumerr/checked)/(sumref/checked), sumsign/checked);
        {   // L_vec is the logsumexp of the scores: it depends only on Q and K,
            // so it checks the QK+softmax half of the kernel on its own.
            std::vector<float> Lg(nL); hipMemcpy(Lg.data(), dL, nL*4, hipMemcpyDeviceToHost);
            double lmax = 0, lsum = 0; long lbad = 0;
            for (int q = 0; q < N; q++) {
                float m = -INFINITY; std::vector<float> sc(N);
                for (int k = 0; k < N; k++) { float d=0;
                    for (int i=0;i<Dh;i++) d += Qf[((size_t)0*N+q)*H*Dh+0*Dh+i]
                                              * Kf[((size_t)0*N+k)*HKV*Dh+0*Dh+i];
                    sc[k]=d*scale; m=fmaxf(m,sc[k]); }
                double su=0; for(int k=0;k<N;k++) su += exp(sc[k]-m);
                double ref_lse = log(su) + m;
                double got = Lg[(size_t)0*H*N + 0*N + q];
                double e = fabs(got - ref_lse);
                if (!(e <= 2e-2*fmax(1.0, fabs(ref_lse)))) lbad++;
                lsum += e; if (e > lmax) lmax = e;
            }
            printf("  L_vec (logsumexp, QK+softmax only): max_err=%.5f mean=%.5f bad=%ld/%d  %s\n",
                   lmax, lsum/N, lbad, N, lbad ? "*** FAIL ***" : "PASS");
            bad += lbad;
        }
        if (bad) return 1;
    }

    if (iters > 0) {
        for (int i = 0; i < 3; i++) dispatch_micro<ATTN_D>(g);
        hipDeviceSynchronize();
        hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
        hipEventRecord(t0);
        for (int i = 0; i < iters; i++) dispatch_micro<ATTN_D>(g);
        hipEventRecord(t1); hipEventSynchronize(t1);
        float ms = 0; hipEventElapsedTime(&ms, t0, t1); ms /= iters;
        // 2 GEMMs, each 2*N*N*D flops per (batch, head), non-causal
        double fl = 4.0 * (double)B * H * (double)N * N * Dh;
        printf("  %.4f ms/iter   %.1f TFLOP/s\n", ms, fl/(ms*1e-3)/1e12);
    }
    return 0;
}
#endif
