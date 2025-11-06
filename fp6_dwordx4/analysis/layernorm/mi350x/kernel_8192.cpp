#include "kittens.cuh"
#include <rocrand_kernel.h>
#include <hip/hip_cooperative_groups.h>
#include "pyutils/pyutils.cuh"


constexpr int B = 16;
constexpr int H = 16;
constexpr int N = 8192;
constexpr int HEAD_D = 128;
constexpr int D = HEAD_D * H;
constexpr float DROPOUT_P = 0.01;

#define NUM_WORKERS (4) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

using G = kittens::group<NUM_WORKERS>;
using namespace kittens;

template<kittens::ducks::rv::all T>
__device__ void dropout_mask(T &dst, float keep_prob) {
    unsigned long long seed = 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    rocrand_state_philox4x32_10 state;
    rocrand_init(clock64(), idx, 0, &state);

    #pragma unroll
    for ( int i = 0 ; i < dst.outer_dim ; i ++ ) { 
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            float rand = rocrand_uniform(&state);
            if (rand < keep_prob) {
                dst[i][j] = base_types::constants<bf16>::zero();
            }
        }
    }
    // mul(dst, dst, __float2bfloat16(1/(1-keep_prob)));
}

template<kittens::ducks::sv::all T>
__device__ void dropout_mask(T &dst, float keep_prob) {
    unsigned long long seed = 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    rocrand_state_philox4x32_10 state;
    rocrand_init(seed, idx, 0, &state);

    #pragma unroll
    for(int cur = laneid(); cur < T::length; cur+=WARP_THREADS) {
        float rand = rocrand_uniform(&state);
        if (rand < keep_prob) {
            dst[cur] = base_types::constants<bf16>::zero();
        }
    }
    // mul(dst, dst, __float2bfloat16(1/(1-keep_prob)));
}

template<int _d_model> struct norm_globals {
    static constexpr int d_model = _d_model;

    // global descriptors
    using x_gl            = gl<bf16, -1, -1, -1, -1>;
    using residual_gl     = gl<bf16, -1, -1, -1, -1>;
    using o_gl            = gl<bf16, -1, -1, -1, -1>;
    using o_resid_gl      = gl<bf16, -1, -1, -1, -1>;
    using norm_weight_gl  = gl<bf16, -1, -1, -1, -1>;
    using norm_bias_gl    = gl<bf16, -1, -1, -1, -1>;

    // global pointers
    x_gl x;
    residual_gl residual;
    o_gl o;
    o_resid_gl o_resid;
    norm_weight_gl norm_weight;
    norm_bias_gl norm_bias;

    const int n_per_tile = 4;
    const int n_tile_size = N / n_per_tile;

    dim3 grid() { return dim3(n_tile_size, B, 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<int D> //__launch_bounds__(NUM_THREADS, 2)
__global__ void layernorm_tk(const norm_globals<D> g) {

    auto warpid = kittens::warpid();
    const int batch = blockIdx.y;
    const int seq_start = blockIdx.x*g.n_per_tile;

    constexpr int d_model = D;
    rv<bf16, d_model, naive_l> residual_s_reg, x_s_reg, norm_weight_s_reg, norm_bias_s_reg;
    load(x_s_reg, g.x, {0, batch, seq_start + warpid, 0});
 
    bf16 mean = __float2bfloat16(0.0f);
    bf16 var  = __float2bfloat16(0.0f);      
    if constexpr (DROPOUT_P > 0.0f) {
        dropout_mask(x_s_reg, DROPOUT_P); 
    }
    load(residual_s_reg, g.residual, {0, batch, seq_start + warpid, 0});
    if constexpr (DROPOUT_P > 0.0f) {
        constexpr float scale = 1.0f / (1.0f - DROPOUT_P);
        mul(x_s_reg, x_s_reg, __float2bfloat16(scale));
    }
    add(residual_s_reg, residual_s_reg, x_s_reg);   
    store(g.o_resid, residual_s_reg, {0, batch, seq_start + warpid, 0});

    // mean and variance
    sum(mean, residual_s_reg); 
    constexpr float dim_scale = 1.0f / d_model;
    mean = mean * __float2bfloat16(dim_scale);
    sub(residual_s_reg, residual_s_reg, mean);  
    load(norm_weight_s_reg, g.norm_weight, {0,0,0,0});
    mul(x_s_reg, residual_s_reg, residual_s_reg);
    sum(var, x_s_reg);
    var = var * __float2bfloat16(dim_scale);
    var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-05f))));

    // compute norm
    div(residual_s_reg, residual_s_reg, var);
    load(norm_bias_s_reg, g.norm_bias, {0,0,0,0});
    mul(residual_s_reg, residual_s_reg, norm_weight_s_reg); 
    add(residual_s_reg, residual_s_reg, norm_bias_s_reg);
    store(g.o, residual_s_reg, {0, batch, seq_start+warpid, 0});
}

template<int D>
void dispatch_micro(norm_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)layernorm_tk<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    layernorm_tk<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro<D>>(m, "dispatch_micro", 
        &norm_globals<D>::x, 
        &norm_globals<D>::residual, 
        &norm_globals<D>::o, 
        &norm_globals<D>::o_resid, 
        &norm_globals<D>::norm_weight, 
        &norm_globals<D>::norm_bias
    );
}


