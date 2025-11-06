#include "kittens.cuh"
#include <rocrand_kernel.h>
#include <hip/hip_cooperative_groups.h>
#include "pyutils/pyutils.cuh"


constexpr int B = 16;
constexpr int H = 16;
constexpr int N = 1024;
constexpr int HEAD_D = 64;
constexpr int D = HEAD_D * H;


#define NUM_WORKERS (4) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

using namespace kittens;

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
    mul(dst, dst, __float2bfloat16(1/(1-keep_prob)));
}

template<int _d_model> struct norm_globals {
    static constexpr int d_model = _d_model;
    static constexpr int dropout_p = 0.01;

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
    size_t dynamic_shared_memory() { return NUM_WORKERS*D*sizeof(bf16)*2*1 + D*sizeof(bf16)*2; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void layernorm_tk(const norm_globals<D> g) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    const int batch = blockIdx.y;
    const int seq_start = blockIdx.x*g.n_per_tile;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    static constexpr int d_model = D;
    using vec_smem_1xD = sv<bf16, d_model>;
    vec_smem_1xD (&x_s)[NUM_WORKERS] = al.allocate<vec_smem_1xD,NUM_WORKERS>();
    vec_smem_1xD (&residual_s)[NUM_WORKERS] = al.allocate<vec_smem_1xD,NUM_WORKERS>();  
    vec_smem_1xD (&norm_weight_s) = al.allocate<vec_smem_1xD>(); 
    vec_smem_1xD (&norm_bias_s  ) = al.allocate<vec_smem_1xD>();                  

    // global loads
    if (warpid == 0) {
        load(norm_bias_s, g.norm_bias, {0,0,0,0});
        load(norm_weight_s, g.norm_weight, {0,0,0,0});
    }
    __syncthreads();
 
    bf16 mean = __float2bfloat16(0.0f);
    bf16 var  = __float2bfloat16(0.0f);      

    int idx = seq_start + warpid;
    load(x_s[warpid], g.x, {0, batch, idx, 0});
    load(residual_s[warpid], g.residual, {0, batch, idx, 0});
    __syncthreads();
    
    dropout_mask(x_s[warpid], g.dropout_p); 
    add(residual_s[warpid], residual_s[warpid], x_s[warpid]);    
    store(g.o_resid, residual_s[warpid], {0, batch, seq_start+warpid, 0});

    sum(mean, residual_s[warpid]);
    mean = mean / __float2bfloat16(d_model);
    sub(residual_s[warpid], residual_s[warpid], mean);  
    mul(x_s[warpid], residual_s[warpid], residual_s[warpid]);
    sum(var, x_s[warpid]);
    var = var / __float2bfloat16(d_model);
    var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-05f))));

    // compute norm
    div(residual_s[warpid], residual_s[warpid], var);
    mul(residual_s[warpid], residual_s[warpid], norm_weight_s); 
    add(residual_s[warpid], residual_s[warpid], norm_bias_s);
    __syncthreads();

    // save output
    store(g.o, residual_s[warpid], {0, batch, seq_start+warpid, 0});
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

