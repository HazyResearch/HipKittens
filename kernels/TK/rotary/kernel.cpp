#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define NUM_WORKERS (4) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

constexpr int ATTN_B = 16;
constexpr int ATTN_H = 16;
constexpr int N = 1024;
constexpr int HEAD_DIM = 128;                   
constexpr float rope_embd_fraction = 1.0f;

// Make these constexpr so they can be used as template parameters
constexpr int ROPE_DIM = static_cast<int>(rope_embd_fraction * HEAD_DIM);
constexpr int HALF_ROPE_DIM = (ROPE_DIM / 2);
constexpr int EXCESS_DIM = HEAD_DIM - ROPE_DIM; 
constexpr int BLOCK_SIZE = 32;
constexpr int DOT_SLICE = 32;

using namespace kittens;

#define tile_1xFULL_ROPE_D st<bf16, BLOCK_SIZE, ROPE_DIM>
#define tile_1xHALF_ROPE_D st<bf16, BLOCK_SIZE, HALF_ROPE_DIM>
#define tile_1xEXCESS_ROPE_D st<bf16, BLOCK_SIZE, EXCESS_DIM>

#define reg_tile_1xFULL_ROPE_D rt<bf16, BLOCK_SIZE, ROPE_DIM>
#define reg_tile_1xHALF_ROPE_D rt<bf16, BLOCK_SIZE, HALF_ROPE_DIM>
#define reg_tile_1xEXCESS_ROPE_D rt<bf16, BLOCK_SIZE, EXCESS_DIM>

template<int _d_model> struct rotary_globals {
    static constexpr int d_model = _d_model;

    // global descriptors
    using x_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1>;
    using sin_gl = gl<bf16, -1, -1, -1, -1>;
    using cos_gl = gl<bf16, -1, -1, -1, -1>;

    // global pointers
    x_gl x;
    o_gl o;
    sin_gl sin;
    cos_gl cos;

    dim3 grid() { return dim3(ATTN_B, ATTN_H); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 2*2*BLOCK_SIZE*ROPE_DIM*NUM_WORKERS + 2*2*2*BLOCK_SIZE*HALF_ROPE_DIM*NUM_WORKERS; }
};

template<int D>
__device__ inline void apply_rotary_embedding(reg_tile_1xFULL_ROPE_D &x_reg,
                                              const reg_tile_1xHALF_ROPE_D &cos_reg,
                                              const reg_tile_1xHALF_ROPE_D &sin_reg) {
    reg_tile_1xHALF_ROPE_D x1, x2, temp1, temp2;
    constexpr int half_dim_tiles = HALF_ROPE_DIM / BLOCK_SIZE;  // Should be 1 for D=64
    // Copy first half and second half
    #pragma unroll
    for(int i = 0; i < half_dim_tiles; i++) {
        #pragma unroll
        for(int j = 0; j < reg_tile_1xHALF_ROPE_D::packed_per_thread; j++){
            x1.tiles[0][i].data[j] = x_reg.tiles[0][i].data[j];                    // First half: elements 0-31
            x2.tiles[0][i].data[j] = x_reg.tiles[0][i + half_dim_tiles].data[j];   // Second half: elements 32-63
        }
    }
    // Apply rotary embedding transformation: new_x1 = x1 * cos - x2 * sin   new_x2 = x2 * cos + x1 * sin
    mul(temp1, x1, cos_reg);  // Compute x1 * cos
    mul(temp2, x2, sin_reg);  // Compute x2 * sin
    sub(temp1, temp1, temp2); // new_x1 = x1 * cos - x2 * sin
    mul(temp2, x2, cos_reg);  // Compute x2 * cos
    mul(x1, x1, sin_reg);     // Compute x1 * sin
    add(temp2, temp2, x1);    // new_x2 = x2 * cos + x1 * sin
    #pragma unroll
    for(int i = 0; i < half_dim_tiles; i++) {
        #pragma unroll
        for(int j = 0; j < reg_tile_1xHALF_ROPE_D::packed_per_thread; j++) {
            x_reg.tiles[0][i].data[j] = temp1.tiles[0][i].data[j];                    // First half
            x_reg.tiles[0][i + half_dim_tiles].data[j] = temp2.tiles[0][i].data[j];   // Second half  
        }
    }
}

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void _fused_rotary(const rotary_globals<D> g) {
    auto warpid = kittens::warpid();
    auto lane = kittens::laneid();

    const int b = blockIdx.x;
    const int h = blockIdx.y;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    tile_1xFULL_ROPE_D (&x_s)[NUM_WORKERS][2] = al.allocate<tile_1xFULL_ROPE_D, NUM_WORKERS, 2>();    
    tile_1xHALF_ROPE_D (&cos_s)[NUM_WORKERS][2] = al.allocate<tile_1xHALF_ROPE_D, NUM_WORKERS, 2>(); 
    tile_1xHALF_ROPE_D (&sin_s)[NUM_WORKERS][2] = al.allocate<tile_1xHALF_ROPE_D, NUM_WORKERS, 2>(); 
    reg_tile_1xFULL_ROPE_D x_reg;
    reg_tile_1xHALF_ROPE_D cos_reg, sin_reg;

    int tic = 0;
    int toc = 1;
    using T = typename tile_1xFULL_ROPE_D::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * WARP_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * ROPE_DIM * sizeof(T) / bytes_per_memcpy;
    constexpr int memcpy_per_tile_cos_sin = BLOCK_SIZE * HALF_ROPE_DIM * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_x[memcpy_per_tile];
    uint32_t swizzled_offsets_cos[memcpy_per_tile_cos_sin];
    uint32_t swizzled_offsets_sin[memcpy_per_tile_cos_sin];
    prefill_swizzled_offsets<2, false>(x_s[warpid][tic], g.x, swizzled_offsets_x);
    prefill_swizzled_offsets<2, false>(cos_s[warpid][tic], g.cos, swizzled_offsets_cos);
    prefill_swizzled_offsets<2, false>(sin_s[warpid][tic], g.sin, swizzled_offsets_sin);
    const lds_lane_ofs lane_ofs = prefill_swizzled_offsets(x_reg, x_s[warpid][0]);
    const lds_lane_ofs lane_ofs_cos = prefill_swizzled_offsets(cos_reg, cos_s[warpid][0]);
    const lds_lane_ofs lane_ofs_sin = prefill_swizzled_offsets(sin_reg, sin_s[warpid][0]);

    load(x_s[warpid][tic], g.x, {b, h, warpid, 0}, swizzled_offsets_x);
    load(cos_s[warpid][tic], g.cos, {0, 0, warpid, 0}, swizzled_offsets_cos);  // cos and sin are shared across batch/head
    load(sin_s[warpid][tic], g.sin, {0, 0, warpid, 0}, swizzled_offsets_sin);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    constexpr int n_blocks = N / (NUM_WORKERS * BLOCK_SIZE);
    #pragma unroll
    for (int block = 0; block < n_blocks-1; block++, tic ^= 1, toc ^= 1) {

        load(x_s[warpid][toc], g.x, {b, h, (block + 1)*NUM_WORKERS + warpid, 0}, swizzled_offsets_x);
        load(cos_s[warpid][toc], g.cos, {0, 0, (block + 1)*NUM_WORKERS + warpid, 0}, swizzled_offsets_cos);  // cos and sin are shared across batch/head
        load(sin_s[warpid][toc], g.sin, {0, 0, (block + 1)*NUM_WORKERS + warpid, 0}, swizzled_offsets_sin);
        
        load(x_reg, x_s[warpid][tic], lane_ofs);
        load(cos_reg, cos_s[warpid][tic], lane_ofs_cos);
        load(sin_reg, sin_s[warpid][tic], lane_ofs_sin);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        apply_rotary_embedding<D>(x_reg, cos_reg, sin_reg);
        store(g.o, x_reg, {b, h, block*NUM_WORKERS + warpid, 0});
    }

    // Epilogue
    load(x_reg, x_s[warpid][tic], lane_ofs);
    load(cos_reg, cos_s[warpid][tic], lane_ofs_cos);
    load(sin_reg, sin_s[warpid][tic], lane_ofs_sin);
    apply_rotary_embedding<D>(x_reg, cos_reg, sin_reg);
    store(g.o, x_reg, {b, h, (n_blocks-1)*NUM_WORKERS + warpid, 0});
}

template<int D>
void dispatch_rotary(rotary_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)_fused_rotary<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    _fused_rotary<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_rotary<HEAD_DIM>>(m, "dispatch_rotary", 
        &rotary_globals<HEAD_DIM>::x, 
        &rotary_globals<HEAD_DIM>::o, 
        &rotary_globals<HEAD_DIM>::sin, 
        &rotary_globals<HEAD_DIM>::cos
    );
}