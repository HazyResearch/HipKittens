#include "kittens.cuh"
#include <random>

using namespace kittens;

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    std::cerr << "HIP error " << hipGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(1);} } while(0)

// Kernel
constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 16; // number of heads
constexpr int ATTN_H_KV = 16; // number of heads for key and value
constexpr int ATTN_N = 8192; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int Q_BLOCK_SIZE = 32; // q block size
constexpr int KV_BLOCK_SIZE = 64; // kv block size

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;
using _gl_L = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

template<int D, typename T=bf16, typename L=row_l> using qo_tile = rt<T, Q_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L>;
template<int D, typename T=bf16, typename L=row_l> using kv_tile = rt<T, KV_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=accum_col_l> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L>;

template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    _gl_L L_vec;
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row>, 2>();
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col> (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col>, 2>();
    
    // const int head_idx = (blockIdx.x % 8) * 8 + (blockIdx.x / 8);
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.z;
    const int GROUP_SIZE = ATTN_H / ATTN_H_KV;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    const int num_tiles = ATTN_N / KV_BLOCK_SIZE;

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
    qo_tile_transposed<D, bf16> q_reg_transposed;
    kv_tile<D, bf16> k_reg;
    kv_tile_transposed<D, bf16> k_reg_transposed;

    kv_tile<D, bf16, accum_col_l> v_reg;
    qo_tile_transposed<D, float, accum_col_l> o_reg; // Output tile.
    attn_tile<D, float, accum_col_l> att_block[2]; // attention tile, in float.
    attn_tile<D, bf16, accum_col_l> att_block_bf16;
    typename attn_tile<D, float, accum_col_l>::row_vec max_vec, norm_vec, max_vec_prev;

    using T = typename st_bf<KV_BLOCK_SIZE, ATTN_D>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * ATTN_D * sizeof(T) / bytes_per_memcpy;

    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swizzled_offsets_V);
    const lds_lane_ofs lane_ofs = prefill_swizzled_offsets(k_reg, k_smem[0]);

    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_K);

    // Pre-scale Q by temperature
    qo_tile<D, float> q_reg_fl;
    load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);  // Use sqrtf for clarity
    copy(q_reg, q_reg_fl);
    swap_layout_and_transpose(q_reg_transposed, q_reg);

    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec_prev);

    // All warps then collaboratively load in the first slice of V (V0) and the second slice of K (K1) into shared memory
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_K);
    // All warps then load in the first slice of K (K0)
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_V);
    load(k_reg, k_smem[0], lane_ofs);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Each warp performs QK0
    asm volatile("s_waitcnt lgkmcnt(0)");
    zero(att_block[0]);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);

    // Each warp performs a partial softmax of QK0 (i.e. some of the online softmax up until but not including the second exponential scaling of the attention block likely)
    // col_max(max_vec, att_block[0], max_vec);
    col_max(max_vec, att_block[0]);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0], att_block[0]);

    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    // All warps then load in the second slice of K (K1)
    load(k_reg, k_smem[1], lane_ofs);
    // All warps then collaboratively load in the third slice of K (K2) into shared memory
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0}, swizzled_offsets_K);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    // All warps then collaboratively load in the second slice of V (V1) into shared memory 
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_V);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // hot loop
    // #pragma unroll  // for some reason unroll makes it slower
    for (int j = 3; j < num_tiles - 1; j += 2) {
        // Cluster 0:
        //      QK1
        asm volatile("s_waitcnt lgkmcnt(0)");
        zero(att_block[1]);
        mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
        //      Finish softmax for QK0
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block[0], norm_vec);
        copy(att_block_bf16, att_block[0]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1:
        //      Load K3 into shared 
        G::load<1, false>(k_smem[1], g.Kg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_K);
        //      Load V0 into registers
        load(v_reg, v_smem[0]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2:
        //      A0V0
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mul_col(o_reg, o_reg, max_vec_prev);
        mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
        //      Partial softmax for QK1
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att_block[1], max_vec);
        sub_col(att_block[1], att_block[1], max_vec);
        exp2(att_block[1], att_block[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3:
        //      Load V2 into shared
        G::load<1, false>(v_smem[0], g.Vg, {batch_idx, j - 1, head_idx_kv, 0}, swizzled_offsets_V);
        //      Load K2 into registers
        load(k_reg, k_smem[0], lane_ofs);
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4:
        //      QK2
        asm volatile("s_waitcnt lgkmcnt(0)");
        zero(att_block[0]);
        mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
        //      Finish softmax for QK1
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block[1], norm_vec);
        copy(att_block_bf16, att_block[1]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5:
        //      Load K4 into shared
        G::load<1, false>(k_smem[0], g.Kg, {batch_idx, j + 1, head_idx_kv, 0}, swizzled_offsets_K);
        //      Load V1 into registers
        load(v_reg, v_smem[1]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6:
        //      A1V1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mul_col(o_reg, o_reg, max_vec_prev);
        mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
        //      Partial softmax for QK2
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att_block[0], max_vec);
        sub_col(att_block[0], att_block[0], max_vec);
        exp2(att_block[0], att_block[0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7:
        //      Load V3 into shared
        G::load<1, false>(v_smem[1], g.Vg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_V);
        //      Load K3 into registers
        load(k_reg, k_smem[1], lane_ofs);
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0:
    //      QK3
    asm volatile("s_waitcnt lgkmcnt(0)");
    zero(att_block[1]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK2
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);

    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1:
    //      Load K5 into shared
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_K);
    //      Load V2 into registers
    load(v_reg, v_smem[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2:
    //      A2V2
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    mul_col(o_reg, o_reg, max_vec_prev);
    //      Partial softmax for QK3
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[1], max_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1], att_block[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3:
    //      Load V4 into shared
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, num_tiles - 2, head_idx_kv, 0}, swizzled_offsets_V);
    //      Load K4 into registers
    load(k_reg, k_smem[0], lane_ofs);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4:
    //      QK4
    asm volatile("s_waitcnt lgkmcnt(0)");
    zero(att_block[0]);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
    //      Finish softmax for QK3
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);

    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5:
    //      Load V3 into registers
    load(v_reg, v_smem[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 6:
    //      A3V3
    asm volatile("s_waitcnt lgkmcnt(0)");
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    mul_col(o_reg, o_reg, max_vec_prev);
    //      Partial softmax for QK4
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[0], max_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0], att_block[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7:
    //      Load V5 into shared
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_V);
    //      Load K5 into registers
    load(k_reg, k_smem[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8:
    //      QK5
    asm volatile("s_waitcnt lgkmcnt(0)");
    zero(att_block[1]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK4
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev); 
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 9:
    //      Load V4 into registers
    load(v_reg, v_smem[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 10:
    //      A4V4
    asm volatile("s_waitcnt lgkmcnt(0)");
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    mul_col(o_reg, o_reg, max_vec_prev);
    //      Full softmax for QK5
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[1], max_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1], att_block[1]);
    sub(max_vec_prev, max_vec_prev, max_vec);
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 11:
    //      Load V5 into registers
    load(v_reg, v_smem[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12:
    //      A5V5
    asm volatile("s_waitcnt lgkmcnt(0)");
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    mul_col(o_reg, o_reg, max_vec_prev);
    div_col(o_reg, o_reg, norm_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Conclusion
    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    qo_tile<D, float, accum_row_l> o_reg_transposed;
    swap_layout_and_transpose(o_reg_transposed, o_reg);
    store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});

    // multiply by ln(2)
    mul(max_vec, max_vec, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
}


__device__ inline bool is_nan(bf16 val) {
    return val != val;
}

__global__ void detect_nan_kernel(bf16* O, float* L, int O_size, int L_size, bool* nan_found) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Check O arrays for NaN
    for (int i = tid; i < O_size; i += total_threads) {
        if (O[i] != O[i]) {
            printf("NaN found in O arrays at index %d: O=%f\n", 
                   i, float(O[i]));
            *nan_found = true;
            return;  // Early exit
        }
    }
    
    // Check L arrays for NaN
    for (int i = tid; i < L_size; i += total_threads) {
        if (L[i] != L[i]) {
            printf("NaN found in L arrays at index %d: L=%f\n", 
                   i, L[i]);
            *nan_found = true;
            return;  // Early exit
        }
    }
}

__global__ void nan_micro() {
    bf16 nan_val = bf16(0.0f) / bf16(0.0f);
    
    // Convert to get bit representation
    union { bf16 b; uint16_t i; } bits;
    bits.b = nan_val;
    
    printf("bf16 bit representation: 0x%x\n", bits.i);
    printf("NaN value (as float): %f\n", (float) nan_val);
    printf("nan != nan: %d\n", nan_val != nan_val);
    printf("is_nan: %d\n", is_nan(nan_val));
    
    // Also test float conversion preservation
    float as_float = (float) nan_val;
    printf("float bit representation: 0x%x\n", as_float);
    printf("float nan != nan: %d\n", as_float != as_float);
}

int main() {
    std::cout << "=== Hammer FWD Test ===" << std::endl;

    bf16 *Q = new bf16[ATTN_B * ATTN_H * ATTN_N * ATTN_D];
    bf16 *K = new bf16[ATTN_B * ATTN_H * ATTN_N * ATTN_D];
    bf16 *V = new bf16[ATTN_B * ATTN_H * ATTN_N * ATTN_D];
    bf16 *O = new bf16[ATTN_B * ATTN_H * ATTN_N * ATTN_D];
    float *L = new float[ATTN_B * ATTN_H * ATTN_N];

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0f, 1.0f);

    // Initialize with different values
    for (int i = 0; i < ATTN_B * ATTN_H * ATTN_N * ATTN_D; i++) {
        Q[i] = bf16(dis(gen));
        K[i] = bf16(dis(gen));
        V[i] = bf16(dis(gen));
        O[i] = bf16(0.0f);
    }

    for (int i = 0; i < ATTN_B * ATTN_H * ATTN_N; i++) {
        L[i] = 0.0f;
    }

    // copy Q, K, V to device
    bf16 *d_Q, *d_K, *d_V;
    hipMalloc(&d_Q, ATTN_B * ATTN_H * ATTN_N * ATTN_D * sizeof(bf16));
    hipMalloc(&d_K, ATTN_B * ATTN_H * ATTN_N * ATTN_D * sizeof(bf16));
    hipMalloc(&d_V, ATTN_B * ATTN_H * ATTN_N * ATTN_D * sizeof(bf16));
    hipMemcpy(d_Q, Q, ATTN_B * ATTN_H * ATTN_N * ATTN_D * sizeof(bf16), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, ATTN_B * ATTN_H * ATTN_N * ATTN_D * sizeof(bf16), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V, ATTN_B * ATTN_H * ATTN_N * ATTN_D * sizeof(bf16), hipMemcpyHostToDevice);
    
    bf16 *d_O;
    float *d_L;
    hipMalloc(&d_O, ATTN_B * ATTN_H * ATTN_N * ATTN_D * sizeof(bf16));
    hipMalloc(&d_L, ATTN_B * ATTN_H * ATTN_N * sizeof(float));

    hipMemcpy(d_O, O, ATTN_B * ATTN_H * ATTN_N * ATTN_D * sizeof(bf16), hipMemcpyHostToDevice);
    hipMemcpy(d_L, L, ATTN_B * ATTN_H * ATTN_N * sizeof(float), hipMemcpyHostToDevice);

    _gl_QKVO input_gl_q(d_Q, ATTN_B, ATTN_N, ATTN_H, ATTN_D);
    _gl_QKVO input_gl_k(d_K, ATTN_B, ATTN_N, ATTN_H, ATTN_D);
    _gl_QKVO input_gl_v(d_V, ATTN_B, ATTN_N, ATTN_H, ATTN_D);
    _gl_QKVO output_gl_o(d_O, ATTN_B, ATTN_N, ATTN_H, ATTN_D);
    _gl_L output_gl_l(d_L, ATTN_B, ATTN_H, 1, ATTN_N);

    attn_globals<ATTN_D> g{input_gl_q, input_gl_k, input_gl_v, output_gl_o, output_gl_l};
    // Allocate GPU memory for NaN detection
    bool* d_nan_found;
    hipMalloc(&d_nan_found, sizeof(bool));
    
    const int NUM_ITERS = 200000;
    const int O_size = ATTN_B * ATTN_H * ATTN_N * ATTN_D;
    const int L_size = ATTN_B * ATTN_H * ATTN_N;
    
    for (int i = 0; i < NUM_ITERS; i++) {

        attend_ker<ATTN_D><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
        hipDeviceSynchronize();
        HIP_CHECK(hipGetLastError());
        
        // Initialize NaN detector to false
        bool initial_nan_found = false;
        hipMemcpy(d_nan_found, &initial_nan_found, sizeof(bool), hipMemcpyHostToDevice);
        
        // Launch NaN detection kernel
        dim3 nan_grid(256);    // 256 blocks
        dim3 nan_block(256);   // 256 threads per block
        detect_nan_kernel<<<nan_grid, nan_block>>>(d_O, d_L, 
                                                   O_size, L_size, d_nan_found);
        hipDeviceSynchronize();
        HIP_CHECK(hipGetLastError());
        
        // Get result back to host
        bool h_nan_found;
        hipMemcpy(&h_nan_found, d_nan_found, sizeof(bool), hipMemcpyDeviceToHost);
        
        if (h_nan_found) {
            std::cout << "NaN detected at iteration " << i << std::endl;
            break;
        }
        
        if (i % 1000 == 0 && i > 0) {
            std::cout << "Iteration " << i << " completed - no NaNs detected" << std::endl;
        }
    }

    // free memory
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] O;
    delete[] L;

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_O);
    hipFree(d_L);
    hipFree(d_nan_found);

    return 0;
}