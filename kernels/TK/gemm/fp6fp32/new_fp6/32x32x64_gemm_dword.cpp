
#include "kittens.cuh"
#include "dword_utils.cpp"
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
  


constexpr int BLOCK_SIZE       = 256;  
constexpr int K_STEP           = 128;
constexpr int REG_BLOCK_M      = BLOCK_SIZE / 2;
constexpr int REG_BLOCK_N      = BLOCK_SIZE / 4;
constexpr int DOT_SLICE        = 64;
              

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 8192
#define K 8192
#define N 8192

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
    dim3 grid()  { return dim3((N / BLOCK_SIZE) * (M / BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_f6<BLOCK_SIZE, K_STEP> (&As)[3] = al.allocate<st_f6<BLOCK_SIZE, K_STEP>, 3>();
    st_f6<BLOCK_SIZE, K_STEP> (&Bs)[3] = al.allocate<st_f6<BLOCK_SIZE, K_STEP>, 3>();

    uintptr_t A_base = reinterpret_cast<uintptr_t>(&As[0]);
    uintptr_t B_base = reinterpret_cast<uintptr_t>(&Bs[0]);

    st_f6<BLOCK_SIZE, K_STEP> *As_ptrs[3] = {
        reinterpret_cast<st_f6<BLOCK_SIZE, K_STEP>*>(A_base + (reinterpret_cast<uintptr_t>(&As[0]) - A_base) * 6 / 8),
        reinterpret_cast<st_f6<BLOCK_SIZE, K_STEP>*>(A_base + (reinterpret_cast<uintptr_t>(&As[1]) - A_base) * 6 / 8),
        reinterpret_cast<st_f6<BLOCK_SIZE, K_STEP>*>(A_base + (reinterpret_cast<uintptr_t>(&As[2]) - A_base) * 6 / 8)
    };

    st_f6<BLOCK_SIZE, K_STEP> *Bs_ptrs[3] = {
        reinterpret_cast<st_f6<BLOCK_SIZE, K_STEP>*>(B_base + (reinterpret_cast<uintptr_t>(&Bs[0]) - B_base) * 6 / 8),
        reinterpret_cast<st_f6<BLOCK_SIZE, K_STEP>*>(B_base + (reinterpret_cast<uintptr_t>(&Bs[1]) - B_base) * 6 / 8),
        reinterpret_cast<st_f6<BLOCK_SIZE, K_STEP>*>(B_base + (reinterpret_cast<uintptr_t>(&Bs[2]) - B_base) * 6 / 8)
    };

    rt_f6<REG_BLOCK_M, DOT_SLICE> A_tile[2];
    rt_f6<REG_BLOCK_N, DOT_SLICE> B_tile[2];
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, ducks::rt_layout::accumulator> C_accum;
    zero(C_accum);

    // Original WGID.
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int NUM_XCDS = 8;
    const int CUS_PER_XCD = 32;
    const int NUM_CUS = CUS_PER_XCD * NUM_XCDS;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    // Swizzle for better L2 within the same XCD.
    const int WGM = 8;
    const int num_pid_m = ceil_div(M, BLOCK_SIZE);
    const int num_pid_n = ceil_div(N, BLOCK_SIZE);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    const int row = pid_m; 
    const int col = pid_n; 

    // Info
    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int num_tiles = K / K_STEP;

    int tic = 0;
    int toc = 2;
    constexpr int bytes_per_thread = 4;
    constexpr int memcpy_per_tile = (BLOCK_SIZE * K_STEP * 6 / 8) / (bytes_per_thread * NUM_THREADS);


    // Register array to store swizzled global addresses for each thread.
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    uint32_t swizzled_offsets_A[memcpy_per_tile];

    prefill_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_A, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row, 0}, *As_ptrs[tic], swizzled_offsets_A);
    prefill_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_B, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col, 0}, *Bs_ptrs[tic], swizzled_offsets_B);

    // Load first tile into shared memory
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_A, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row, 0}, *As_ptrs[0], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_B, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col, 0}, *Bs_ptrs[0], swizzled_offsets_B);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // Load second tile into shared memory
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_A, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row, 1}, *As_ptrs[1], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_B, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col, 1}, *Bs_ptrs[1], swizzled_offsets_B);
    // Load first tile into registers
    load_lds_reg_row_fp6_shuffled(B_tile[0], subtile_inplace<REG_BLOCK_N, DOT_SLICE>(*Bs_ptrs[tic], {warp_col, 0}));
    load_lds_reg_row_fp6_shuffled(A_tile[0], subtile_inplace<REG_BLOCK_M, DOT_SLICE>(*As_ptrs[tic], {warp_row, 0}));
    // __builtin_amdgcn_s_waitcnt(0);
    // asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    // #pragma unroll
    for (int tile = 0; tile < num_tiles - 2; ++tile) {

        // Cluster 0
        load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_A, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row, tile+2}, *As_ptrs[toc], swizzled_offsets_A);
        load_lds_reg_row_fp6_shuffled(A_tile[1], subtile_inplace<REG_BLOCK_M, DOT_SLICE>(*As_ptrs[tic], {warp_row, 1}));
        load_lds_reg_row_fp6_shuffled(B_tile[1], subtile_inplace<REG_BLOCK_N, DOT_SLICE>(*Bs_ptrs[tic], {warp_col, 1}));
        // asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, A_tile[0], B_tile[0], C_accum);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_B, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col, tile+2}, *Bs_ptrs[toc], swizzled_offsets_B);
        tic = (tic + 1) % 3;
        toc = (toc + 1) % 3;
        load_lds_reg_row_fp6_shuffled(B_tile[0], subtile_inplace<REG_BLOCK_N, DOT_SLICE>(*Bs_ptrs[tic], {warp_col, 0}));
        load_lds_reg_row_fp6_shuffled(A_tile[0], subtile_inplace<REG_BLOCK_M, DOT_SLICE>(*As_ptrs[tic], {warp_row, 0}));
        // asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3 (compute)
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, A_tile[1], B_tile[1], C_accum);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0
    // __builtin_amdgcn_sched_barrier(0);
    load_lds_reg_row_fp6_shuffled(B_tile[1], subtile_inplace<REG_BLOCK_N, DOT_SLICE>(*Bs_ptrs[tic], {warp_col, 1}));
    load_lds_reg_row_fp6_shuffled(A_tile[1], subtile_inplace<REG_BLOCK_M, DOT_SLICE>(*As_ptrs[tic], {warp_row, 1}));
    // asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();   
    __builtin_amdgcn_sched_barrier(0); 

    // Cluster 1
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum, A_tile[0], B_tile[0], C_accum);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2 (load)
    tic = (tic + 1) % 3;
    load_lds_reg_row_fp6_shuffled(B_tile[0], subtile_inplace<REG_BLOCK_N, DOT_SLICE>(*Bs_ptrs[tic], {warp_col, 0}));
    load_lds_reg_row_fp6_shuffled(A_tile[0], subtile_inplace<REG_BLOCK_M, DOT_SLICE>(*As_ptrs[tic], {warp_row, 0}));
    // asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3 (compute)
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum, A_tile[1], B_tile[1], C_accum);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 0
    // __builtin_amdgcn_sched_barrier(0);
    load_lds_reg_row_fp6_shuffled(B_tile[1], subtile_inplace<REG_BLOCK_N, DOT_SLICE>(*Bs_ptrs[tic], {warp_col, 1}));
    load_lds_reg_row_fp6_shuffled(A_tile[1], subtile_inplace<REG_BLOCK_M, DOT_SLICE>(*As_ptrs[tic], {warp_row, 1}));
    // asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();   
    __builtin_amdgcn_sched_barrier(0); 

    // Cluster 1
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum, A_tile[0], B_tile[0], C_accum);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2 (load)
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3 (compute)
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum, A_tile[1], B_tile[1], C_accum);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store(g.c, C_accum, {0, 0, row * 2 + warp_row, col * 4 + warp_col});
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
    const int WARMUP_REPS = 1000;
    for (int r = 0; r < WARMUP_REPS; ++r) { 
        micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory(), stream>>>(globals);
    }
    hipDeviceSynchronize();

    // Timed kernel-only loop
    const int REPS = 200;
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
        } 
        // else {
        if (num_printed_correct < 5) {
                int row = i / N;
                int col = i % N;
                std::cout << "[" << row << "," << col << "] CPU: " << float(cpu_result[i]) 
                          << " GPU: " << h_output_float 
                          << " (diff: " << diff << " / threshold: " << threshold << ")\n";
                num_printed_correct++;
        }
        // }
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