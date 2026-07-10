#include "kittens.cuh"
#include <fstream>
#include <chrono>
#include <cmath>


#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#ifndef ATTN_B
constexpr int ATTN_B = 16; // batch size
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 8;  // number of heads
#endif

#ifndef ATTN_D
constexpr int ATTN_D = 128; // dimension
#endif

#ifndef ATTN_F
constexpr int ATTN_F = 128;  // number of features
#endif

#ifndef ATTN_N
constexpr int ATTN_N = 1024; // sequence length
#endif

constexpr int CHUNK_SIZE = 32;
constexpr int V_CHUNK_SIZE = 32;

using namespace kittens;

using G = kittens::group<NUM_WARPS>;

template<int F, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using q_tile = rt<T, CHUNK_SIZE, F, L, S>;
template<int F, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using q_tile_transposed = rt<T, F, CHUNK_SIZE, L, S>;
template<int F, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using k_tile = rt<T, CHUNK_SIZE, F, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using v_tile = rt<T, CHUNK_SIZE, V_CHUNK_SIZE, L, S>;
template<int F, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using k_tile_transposed = rt<T, F, CHUNK_SIZE, L, S>;

using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

struct attn_globals {
    _gl_QKVO Qg, Kg, K_split_g, Vg, Og;
    _gl_QKVO ODEBUGg;

    uintptr_t slopes;

    hipStream_t stream;
    dim3 block() {return dim3(NUM_THREADS);}
    dim3 grid() {return dim3(ATTN_D / V_CHUNK_SIZE, ATTN_H, ATTN_B);}
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void attn_kernel(const attn_globals globals, int N)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // smem
    st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s> (&q_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s>, 2>();
    st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s>, 2>();
    st_bf<CHUNK_SIZE, V_CHUNK_SIZE, st_32x32_s> (&v_smem)[2] = al.allocate<st_bf<CHUNK_SIZE, V_CHUNK_SIZE, st_32x32_s>, 2>();
    st_bf<ATTN_F, V_CHUNK_SIZE, st_32x32_s> (&kv_state_smem) = al.allocate<st_bf<ATTN_F, V_CHUNK_SIZE, st_32x32_s>>();

    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int v_block_idx = blockIdx.x;
    float slope = reinterpret_cast<float*>(globals.slopes)[head_idx];
    int blocks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const int tic = 0, toc = 1;

    // Initialize all of the register tiles.
    q_tile<ATTN_F, bf16> q_reg;
    q_tile_transposed<ATTN_F, bf16> q_reg_transposed;
    k_tile<ATTN_F, bf16> k_reg;
    k_tile_transposed<ATTN_F, bf16> k_reg_transposed;
    v_tile<ATTN_D, bf16, col_l, rt_32x32_s> v_reg;

    using T = typename st_bf<CHUNK_SIZE, ATTN_F, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>(); // 16
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;             // 16 * 64 * 8 = 8192
    constexpr int memcpy_per_tile_q_k = CHUNK_SIZE * ATTN_F * sizeof(T) / bytes_per_memcpy;
    constexpr int memcpy_per_tile_v = CHUNK_SIZE * V_CHUNK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_Q[memcpy_per_tile_q_k];
    uint32_t swizzled_offsets_V[memcpy_per_tile_v];
    uint32_t swizzled_offsets_K[memcpy_per_tile_q_k];
    G::prefill_swizzled_offsets<1, false>(q_smem[0], globals.Qg, swizzled_offsets_Q);
    G::prefill_swizzled_offsets<1, false>(k_smem[0], globals.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], globals.Vg, swizzled_offsets_V);

    for (int block = 0; block < blocks; block++) {
        // Load Q, K, V tiles from global memory to shared memory
        G::load<1, false>(q_smem[tic], globals.Qg, {batch_idx, block, head_idx, 0}, swizzled_offsets_Q);
        G::load<1, false>(k_smem[tic], globals.Kg, {batch_idx, block, head_idx, 0}, swizzled_offsets_K);
        G::load<1, false>(v_smem[tic], globals.Vg, {batch_idx, block, head_idx, v_block_idx}, swizzled_offsets_V);

        // smem to reg
        asm volatile("s_waitcnt vmcnt(2)");
        load(q_reg, q_smem[tic]);
        asm volatile("s_waitcnt vmcnt(1)");
        load(k_reg, k_smem[tic]);
        asm volatile("s_waitcnt vmcnt(0)");
        load(v_reg, v_smem[tic]);
    
        // TODO
    }
}