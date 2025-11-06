#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define NUM_WORKERS (1) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

constexpr int ATTN_B = 16;
constexpr int ATTN_H = 16;
constexpr int ATTN_N = 2048;
constexpr int HEAD_DIM = 128;                   
constexpr float rope_embd_fraction = 1.0f;

constexpr int ROPE_DIM = static_cast<int>(rope_embd_fraction * HEAD_DIM);
constexpr int HALF_ROPE_DIM = (ROPE_DIM / 2);
constexpr int EXCESS_DIM = HEAD_DIM - ROPE_DIM; 
constexpr int BLOCK_SIZE = 32;

using namespace kittens;

#define tile_1xEXCESS_ROPE_D st<bf16, BLOCK_SIZE, EXCESS_DIM>
#define reg_tile_1xEXCESS_ROPE_D rt<bf16, BLOCK_SIZE, EXCESS_DIM>

template<int _d_model> struct rotary_globals {
    static constexpr int d_model = _d_model;
    using x_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1>;
    using sin_gl = gl<bf16, -1, -1, -1, -1>;
    using cos_gl = gl<bf16, -1, -1, -1, -1>;
    x_gl x;
    o_gl o;
    sin_gl sin;
    cos_gl cos;

    dim3 grid() { return dim3((ATTN_B*ATTN_H), ATTN_N/(BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return (0); }
};

template<int D>
__device__ __forceinline__ void apply_rotary_embedding(rt<bf16, BLOCK_SIZE, ROPE_DIM> &x_reg,
                                              const rt<bf16, BLOCK_SIZE, HALF_ROPE_DIM> &cos_reg,
                                              const rt<bf16, BLOCK_SIZE, HALF_ROPE_DIM> &sin_reg) {
    rt<bf16, BLOCK_SIZE, HALF_ROPE_DIM> temp1, temp2, temp3;
    constexpr int half_dim_tiles = HALF_ROPE_DIM / BLOCK_SIZE;
    #pragma unroll
    for(int i = 0; i < half_dim_tiles; i++) {
        #pragma unroll
        for(int j = 0; j < rt<bf16, BLOCK_SIZE, HALF_ROPE_DIM>::packed_per_thread; j++) {
            auto x1_val = x_reg.tiles[0][i].data[j];
            auto x2_val = x_reg.tiles[0][i + half_dim_tiles].data[j];
            auto cos_val = cos_reg.tiles[0][i].data[j];
            auto sin_val = sin_reg.tiles[0][i].data[j];
            
            // Compute new values directly
            temp1.tiles[0][i].data[j] = __hsub2(__hmul2(x1_val, cos_val), __hmul2(x2_val, sin_val));
            temp2.tiles[0][i].data[j] = __hadd2(__hmul2(x2_val, cos_val), __hmul2(x1_val, sin_val));
        }
    }
    
    // Single write-back pass
    #pragma unroll
    for(int i = 0; i < half_dim_tiles; i++) {
        #pragma unroll
        for(int j = 0; j < rt<bf16, BLOCK_SIZE, HALF_ROPE_DIM>::packed_per_thread; j++) {
            x_reg.tiles[0][i].data[j] = temp1.tiles[0][i].data[j];
            x_reg.tiles[0][i + half_dim_tiles].data[j] = temp2.tiles[0][i].data[j];
        }
    }
}

template<int D> __launch_bounds__(NUM_THREADS, 4)
__global__ void tk_fused_rotary(const rotary_globals<D> g) {
    auto warpid = kittens::warpid();
    auto lane = kittens::laneid();
    const int b = blockIdx.x/ATTN_H;
    const int h = blockIdx.x%ATTN_H;
    const int n = blockIdx.y;

    rt<bf16, BLOCK_SIZE, ROPE_DIM> x_reg;
    rt<bf16, BLOCK_SIZE, HALF_ROPE_DIM> cos_reg, sin_reg;

    load(cos_reg, g.cos, {0, 0, n, 0});
    load(sin_reg, g.sin, {0, 0, n, 0});
    load(x_reg, g.x, {b, h, n, 0});
    asm volatile("s_waitcnt lgkmcnt(0)");
    apply_rotary_embedding<D>(x_reg, cos_reg, sin_reg);
    store(g.o, x_reg, {b, h, n, 0}); 
}

template<int D>
void dispatch_rotary(rotary_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)tk_fused_rotary<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    tk_fused_rotary<D><<<g.grid(), g.block(), mem_size>>>(g);
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

