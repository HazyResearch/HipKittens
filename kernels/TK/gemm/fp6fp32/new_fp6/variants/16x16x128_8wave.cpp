
#include "kittens.cuh"
#include "16x16x128_utils.cpp"
#include <random>
#include <omp.h>
#include <cstring>
#include <iomanip>
using namespace kittens;


using din = fp6_e2m3;
using dout = half;

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    std::cerr << "HIP error " << hipGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(1);} } while(0)

constexpr int BLOCK_SIZE_M     = 256;
constexpr int BLOCK_SIZE_N     = 256;  
constexpr int K_STEP           = 128;

constexpr int HALF_BLOCK_SIZE_M = BLOCK_SIZE_M / 2;
constexpr int HALF_BLOCK_SIZE_N = BLOCK_SIZE_N / 2;

constexpr int WARPS_ROW = 2;
constexpr int WARPS_COL = 4;

constexpr int REG_BLOCK_M      = BLOCK_SIZE_M / 2 / WARPS_ROW;
constexpr int REG_BLOCK_N      = BLOCK_SIZE_N / 2 / WARPS_COL;

#define NUM_WARPS (WARPS_ROW * WARPS_COL)
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 8192
#define K 8192
#define N 8192
constexpr int k_iters = K / K_STEP; // K iterations

using _gl_A = gl<din, -1, -1, -1, -1>;
using _gl_B = gl<din, -1, -1, -1, -1>;
using _gl_C = gl<dout, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

__host__ __device__ inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    dim3 grid()  { return dim3((N / BLOCK_SIZE_N), (M / BLOCK_SIZE_M)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

constexpr int axis = 2;

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_f6<HALF_BLOCK_SIZE_M, K_STEP>;
    using ST_B = st_f6<HALF_BLOCK_SIZE_N, K_STEP>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    using RT_A = rt_f6<REG_BLOCK_M, K_STEP>;
    using RT_B = rt_f6<REG_BLOCK_N, K_STEP>;
    using RT_C = rt_fl<REG_BLOCK_M, REG_BLOCK_N, ducks::rt_layout::accumulator>;

    RT_A a;
    RT_B b0;
    RT_B b1;
    RT_C cA;
    RT_C cB;
    RT_C cC;
    RT_C cD;

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int block_m = block_row * BLOCK_SIZE_M;
    int block_n = block_col * BLOCK_SIZE_N;

    int warp_m = (warpid() / WARPS_COL);
    int warp_n = (warpid() % WARPS_COL);

    zero(cA);
    zero(cB);
    zero(cC);
    zero(cD);

    int tic = 0, toc = 1;

    load_global_to_shared_fp6<axis, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col * 2, 0}, Bs[tic][0]);
    load_global_to_shared_fp6<axis, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row * 2, 0}, As[tic][0]);
    load_global_to_shared_fp6<axis, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col * 2 + 1, 0}, Bs[tic][1]);
    load_global_to_shared_fp6<axis, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row * 2 + 1, 0}, As[tic][1]);

    load_global_to_shared_fp6<axis, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row * 2, 1}, As[toc][0]);
    load_global_to_shared_fp6<axis, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col * 2, 1}, Bs[toc][0]);
    load_global_to_shared_fp6<axis, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col * 2 + 1, 1}, Bs[toc][1]);

    asm volatile("s_waitcnt vmcnt(12)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto bs_subtile0 = kittens::subtile_inplace<BLOCK_SIZE_N / WARPS_COL / 2, K_STEP>(Bs[tic][0], {warp_n, 0});
    load_lds_reg_row_fp6(b0, bs_subtile0);

    asm volatile("s_waitcnt vmcnt(10)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    if (warp_m == 1) {
        __builtin_amdgcn_s_barrier();
    }

    // {   
    //     constexpr int k = 0;

    //     auto as_subtile0 = kittens::subtile_inplace<BLOCK_SIZE_M / WARPS_ROW / 2, K_STEP>(As[tic][0], {warp_m, 0});
    //     load_lds_reg_row_fp6(a, as_subtile0);
    //     load_global_to_shared_fp6<axis, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row * 2 + 1, k + 1}, As[toc][1]);

    //     asm volatile("s_waitcnt lgkmcnt(0)");
    //     asm volatile("s_waitcnt vmcnt(10)");
    //     __builtin_amdgcn_s_barrier();
    //     __builtin_amdgcn_sched_barrier(0);

    //     __builtin_amdgcn_s_setprio(1);
    //     mma_ABt(cA, a, b0, cA);
    //     __builtin_amdgcn_s_setprio(0);

    //     __builtin_amdgcn_sched_barrier(0);
    //     __builtin_amdgcn_s_barrier();
    //     __builtin_amdgcn_sched_barrier(0);

    //     auto bs_subtile1 = kittens::subtile_inplace<BLOCK_SIZE_N / WARPS_COL / 2, K_STEP>(Bs[tic][1], {warp_n, 0});
    //     load_lds_reg_row_fp6(b1, bs_subtile1);
    //     load_global_to_shared_fp6<axis, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row * 2, k + 2}, As[tic][0]);

    //     asm volatile("s_waitcnt lgkmcnt(0)");
    //     asm volatile("s_waitcnt vmcnt(10)");
    //     __builtin_amdgcn_s_barrier();
    //     __builtin_amdgcn_sched_barrier(0);

    //     __builtin_amdgcn_s_setprio(1);
    //     mma_ABt(cB, a, b1, cB);
    //     __builtin_amdgcn_s_setprio(0);

    //     __builtin_amdgcn_sched_barrier(0);
    //     __builtin_amdgcn_s_barrier();
    //     __builtin_amdgcn_sched_barrier(0);

    //     auto as_subtile1 = kittens::subtile_inplace<BLOCK_SIZE_M / WARPS_ROW / 2, K_STEP>(As[tic][1], {warp_m, 0});
    //     load_lds_reg_row_fp6(a, as_subtile1);
    //     load_global_to_shared_fp6<axis, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col * 2, k + 2}, Bs[tic][0]);

    //     asm volatile("s_waitcnt lgkmcnt(0)");
    //     asm volatile("s_waitcnt vmcnt(10)");
    //     __builtin_amdgcn_s_barrier();
    //     __builtin_amdgcn_sched_barrier(0);

    //     __builtin_amdgcn_s_setprio(1);
    //     mma_ABt(cC, a, b0, cC);
    //     __builtin_amdgcn_s_setprio(0);

    //     __builtin_amdgcn_sched_barrier(0);
    //     __builtin_amdgcn_s_barrier();
    //     __builtin_amdgcn_sched_barrier(0);

    //     auto bs_subtile0 = kittens::subtile_inplace<BLOCK_SIZE_N / WARPS_COL / 2, K_STEP>(Bs[toc][0], {warp_n, 0});
    //     load_lds_reg_row_fp6(b0, bs_subtile0);
    //     load_global_to_shared_fp6<axis, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col * 2 + 1, k + 2}, Bs[tic][1]);

    //     __builtin_amdgcn_sched_barrier(0);
    //     asm volatile("s_waitcnt vmcnt(10)");
    //     __builtin_amdgcn_s_barrier();
    //     __builtin_amdgcn_sched_barrier(0);

    //     __builtin_amdgcn_s_setprio(1);
    //     mma_ABt(cD, a, b1, cD);
    //     __builtin_amdgcn_s_setprio(0);

    //     __builtin_amdgcn_sched_barrier(0);
    //     __builtin_amdgcn_s_barrier();
    //     __builtin_amdgcn_sched_barrier(0);
    // }

    // tic^=1, toc^=1;

    // Inner loop over K dimension
    #pragma unroll
    for (int k = 0; k < k_iters - 2; k++, tic^=1, toc^=1) {

        auto as_subtile0 = kittens::subtile_inplace<BLOCK_SIZE_M / WARPS_ROW / 2, K_STEP>(As[tic][0], {warp_m, 0});
        load_lds_reg_row_fp6(a, as_subtile0);
        load_global_to_shared_fp6<axis, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row * 2 + 1, k + 1}, As[toc][1]);

        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(10)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile1 = kittens::subtile_inplace<BLOCK_SIZE_N / WARPS_COL / 2, K_STEP>(Bs[tic][1], {warp_n, 0});
        load_lds_reg_row_fp6(b1, bs_subtile1);
        load_global_to_shared_fp6<axis, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row * 2, k + 2}, As[tic][0]);

        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(10)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto as_subtile1 = kittens::subtile_inplace<BLOCK_SIZE_M / WARPS_ROW / 2, K_STEP>(As[tic][1], {warp_m, 0});
        load_lds_reg_row_fp6(a, as_subtile1);
        load_global_to_shared_fp6<axis, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col * 2, k + 2}, Bs[tic][0]);

        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(10)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile0 = kittens::subtile_inplace<BLOCK_SIZE_N / WARPS_COL / 2, K_STEP>(Bs[toc][0], {warp_n, 0});
        load_lds_reg_row_fp6(b0, bs_subtile0);
        load_global_to_shared_fp6<axis, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col * 2 + 1, k + 2}, Bs[tic][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(10)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    {
        constexpr int k = k_iters - 2;

        auto as_subtile0 = kittens::subtile_inplace<BLOCK_SIZE_M / WARPS_ROW / 2, K_STEP>(As[tic][0], {warp_m, 0});
        load_lds_reg_row_fp6(a, as_subtile0);
        load_global_to_shared_fp6<axis, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row * 2 + 1, k + 1}, As[toc][1]);

        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(10)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile1 = kittens::subtile_inplace<BLOCK_SIZE_N / WARPS_COL / 2, K_STEP>(Bs[tic][1], {warp_n, 0});
        load_lds_reg_row_fp6(b1, bs_subtile1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto as_subtile1 = kittens::subtile_inplace<BLOCK_SIZE_M / WARPS_ROW / 2, K_STEP>(As[tic][1], {warp_m, 0});
        load_lds_reg_row_fp6(a, as_subtile1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile0 = kittens::subtile_inplace<BLOCK_SIZE_N / WARPS_COL / 2, K_STEP>(Bs[toc][0], {warp_n, 0});
        load_lds_reg_row_fp6(b0, bs_subtile0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    tic^=1, toc^=1;

    {
        auto as_subtile0 = kittens::subtile_inplace<BLOCK_SIZE_M / WARPS_ROW / 2, K_STEP>(As[tic][0], {warp_m, 0});
        load_lds_reg_row_fp6(a, as_subtile0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile1 = kittens::subtile_inplace<BLOCK_SIZE_N / WARPS_COL / 2, K_STEP>(Bs[tic][1], {warp_n, 0});
        load_lds_reg_row_fp6(b1, bs_subtile1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto as_subtile1 = kittens::subtile_inplace<BLOCK_SIZE_M / WARPS_ROW / 2, K_STEP>(As[tic][1], {warp_m, 0});
        load_lds_reg_row_fp6(a, as_subtile1);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);
    }

    if (warp_m == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store(g.c, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(g.c, cB, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(g.c, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(g.c, cD, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
}

void pack(uint32_t *output, const din *input, int size) {

    for (int i = 0; i < size * 6 / 32; i++) {
        output[i] = 0;
    }

    for (int i = 0; i < size; i++) {
        const uint8_t tmp = *reinterpret_cast<const uint8_t*>(&input[i]);
        const uint32_t v = static_cast<uint32_t>(tmp & 0x3Fu);
        const int bit_pos = i * 6;
        const int word_idx = bit_pos >> 5;
        const int bit_off = bit_pos & 31;

        output[word_idx] |= (v << bit_off);
        const int spill = bit_off + 6 - 32;
        if (spill > 0) {
            output[word_idx + 1] |= (v >> (6 - spill));
        }
    }
}


int main() {
    std::cout << "=== FP6 Packed GEMM Test ===\n";
    
    din *h_input_a = new din[M * K];
    din *h_input_b = new din[N * K];
    dout *h_output = new dout[M * N];

    // Benchmarking variables
    double gemm_flops = 2.0 * double(M) * double(N) * double(K);
    hipStream_t stream;
    HIP_CHECK( hipStreamCreate(&stream) );
    hipEvent_t start, stop;
    HIP_CHECK( hipEventCreate(&start) );
    HIP_CHECK( hipEventCreate(&stop) );

    // Calculate sizes for packed FP6 data
    int total_bytes_a = ( M * K * 6 ) / 8;
    int total_bytes_b = ( N * K * 6 ) / 8;
    int total_words_a = ( M * K * 6 ) / 32;
    int total_words_b = ( N * K * 6 ) / 32;

    // Allocate packed arrays
    uint32_t *h_input_a_packed = new uint32_t[total_words_a];
    uint32_t *h_input_b_packed = new uint32_t[total_words_b];

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0f, 1.0f);

    // Initialize with different values
    for (int i = 0; i < M * K; i++) {
        h_input_a[i] = din(dis(gen));
        h_input_b[i] = din(dis(gen));
    }
    
    // Pack the input data
    std::cout << "Packing input data...\n";
    pack(h_input_a_packed, h_input_a, M * K);
    pack(h_input_b_packed, h_input_b, N * K);
    
    // Print first few packed words for debugging
    std::cout << "First 4 packed words of A: ";
    for (int i = 0; i < 4 && i < total_words_a; i++) {
        std::cout << "0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << h_input_a_packed[i] << " ";
    }
    std::cout << std::dec << "\n\n";

    din *d_input_a_packed;
    din *d_input_b_packed;
    dout *d_output;
    hipMalloc(&d_input_a_packed, total_bytes_a);
    hipMalloc(&d_input_b_packed, total_bytes_b);
    hipMalloc(&d_output, M * N * sizeof(dout));

    // Copy packed data to device
    hipMemcpy(d_input_a_packed, h_input_a_packed, total_bytes_a, hipMemcpyHostToDevice);
    hipMemcpy(d_input_b_packed, h_input_b_packed, total_bytes_b, hipMemcpyHostToDevice);

    _gl_A input_gl_a(d_input_a_packed, 1, 1, M, K);
    _gl_B input_gl_b(d_input_b_packed, 1, 1, N, K);
    _gl_C output_gl(d_output, 1, 1, M, N);
    micro_globals globals{input_gl_a, input_gl_b, output_gl};

    // Warmup
    // Warmup
    const int WARMUP_REPS = 500;
    for (int r = 0; r < WARMUP_REPS; ++r) { 
        micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory(), stream>>>(globals);
    }
    hipDeviceSynchronize();

    // Timed kernel-only loop
    const int REPS = 100;
    std::vector<float> times_ms;
    times_ms.reserve(REPS);
    for (int r = 0; r < REPS; ++r) {
        HIP_CHECK( hipEventRecord(start, stream) );
        micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory(), stream>>>(globals);
        HIP_CHECK( hipEventRecord(stop, stream) );
        HIP_CHECK( hipEventSynchronize(stop) );
        float ms = 0.0f;
        HIP_CHECK( hipEventElapsedTime(&ms, start, stop) );
        times_ms.push_back(ms);
    }

    float sum_ms = 0.f, best_ms = 1e30f;
    for (float t : times_ms) { sum_ms += t; best_ms = std::min(best_ms, t); }
    float avg_ms = sum_ms / times_ms.size();
    double tflops_best = (gemm_flops / (best_ms * 1e-3)) / 1e12;
    double tflops_avg  = (gemm_flops / (avg_ms  * 1e-3)) / 1e12;
    std::cout << "Kernel time (best): " << best_ms << " ms,  TFLOPs: " << tflops_best << "\n";
    std::cout << "Kernel time (avg ): " << avg_ms  << " ms,  TFLOPs: " << tflops_avg  << "\n";

    
    hipMemcpy(h_output, d_output, M * N * sizeof(dout), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    // Check for kernel errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // CPU reference: compute A * B^T
    std::cout << "Computing CPU reference...\n";
    half *cpu_result = new half[M * N];
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += float(h_input_a[i * K + k]) * float(h_input_b[j * K + k]);
            }
            cpu_result[i * N + j] = half(sum);
        }
    }
    
    // Compare results
    int errors = 0;
    int num_printed = 0;
    int num_printed_correct = 0;
    float max_diff = 0.0f;
    float total_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float h_output_float = float(h_output[i]);
        const float rtol = 0.1f;   // ~u with a little margin
        const float atol = 1e-2f;   // floor for tiny expected values
        float diff = fabs(float(cpu_result[i]) - h_output_float);
        float threshold = rtol * fabs(float(cpu_result[i])) + atol;
        max_diff = std::max(max_diff, diff);
        total_diff += diff;
        if (diff > threshold) {
            ++errors;
            if (num_printed < 5) {
                int row = i / N;
                int col = i % N;
                std::cout << "[" << row << "," << col << "] CPU: " << float(cpu_result[i]) 
                          << " GPU: " << h_output_float 
                          << " (diff: " << diff << " / threshold: " << threshold << ")\n";
                num_printed++;
            }
        } else {
            if (num_printed_correct < 5) {
                int row = i / N;
                int col = i % N;
                std::cout << "[" << row << "," << col << "] CPU: " << float(cpu_result[i]) 
                          << " GPU: " << h_output_float 
                          << " (diff: " << diff << " / threshold: " << threshold << ")\n";
                num_printed_correct++;
            }
        }
    }

    std::cout << "Average diff: " << total_diff / (M * N) << std::endl;
    std::cout << "Max diff: " << max_diff << std::endl;
    std::cout << "Errors: " << errors << "/" << (M * N) << std::endl;
    if (errors < 100) {
        std::cout << "GEMM test PASSED" << std::endl;
    } else {
        std::cout << "GEMM test FAILED" << std::endl;
    }
    
    // Cleanup
    delete[] cpu_result;
    delete[] h_input_a;
    delete[] h_input_b;
    delete[] h_input_a_packed;
    delete[] h_input_b_packed;
    delete[] h_output;
    hipFree(d_input_a_packed);
    hipFree(d_input_b_packed);
    hipFree(d_output);
    return 0;
}