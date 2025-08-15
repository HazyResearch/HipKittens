#include "kittens.cuh"
#include <random>
#include <omp.h>

using namespace kittens;

// Timing result structure
struct TimingResult {
    float best_time_ms;
    float avg_time_ms;
    double best_tflops;
    double avg_tflops;
    int timing_iterations;
};

// #define DUMP_TO_CSV

#define PROFILE

#define HipCheckError()    __hipCheckError( __FILE__, __LINE__ )
inline void __hipCheckError( const char *file, const int line ) {
    hipError_t err = hipGetLastError();
    if ( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = hipDeviceSynchronize();
    if( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() with sync failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
}

#ifndef PROFILE

template <typename T>
void dump_to_csv(const char* filename, const T& data, int rows, int cols) {
    FILE* f = fopen(filename, "w");
    if (f) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                fprintf(f, "%f", float(data[i * cols + j]));
                if (j < cols - 1) fprintf(f, ",");
            }
            fprintf(f, "\n");
        }
        fclose(f);
    } else {
        printf("Failed to open %s for writing\n", filename);
    }
}

template <int M, int N, int K>
__global__ void matmul_device_ref(const kittens::gl<fp8e4m3, 1, 1, M, K> A, const kittens::gl<fp8e4m3, 1, 1, N, K> B, const kittens::gl<float, 1, 1, M, N> C) {
    static_assert(M % 64 == 0, "M must be a multiple of 64");
    static_assert(N % 64 == 0, "N must be a multiple of 64");
    static_assert(K % 64 == 0, "K must be a multiple of 64");

    constexpr int k_iters = K / 64;
    constexpr int n_iters = N / 64; // thread-block iters
    constexpr int m_iters = M / 64; // thread-block iters

    rt_fp8e4m3<32, 64> a;
    rt_fp8e4m3<32, 64> b;
    rt_fl<32, 32, kittens::ducks::rt_layout::accumulator> c;

    constexpr int total_iters = n_iters * m_iters;

    for (int i = blockIdx.x; i < total_iters; i += gridDim.x) {
        constexpr int warps_per_block_dim = 2;
        // Convert linear block index to 2D coordinates in the grid
        int block_m = i / n_iters;  // which row of blocks
        int block_n = i % n_iters;  // which column of blocks
        
        // Map warps within the block to sub-blocks
        int i_m = block_m * warps_per_block_dim + warpid() / warps_per_block_dim;
        int i_n = block_n * warps_per_block_dim + warpid() % warps_per_block_dim;

        zero(c);
        for (int k = 0; k < k_iters; k++) {
            load(a, A, {0, 0, i_m, k});
            load(b, B, {0, 0, i_n, k});
            mma_ABt(c, a, b, c);
            store(C, c, {0, 0, i_m, i_n});
        }
    }
}

template <int M, int N, int K, int CUs>
TimingResult matmul_ref(const std::vector<fp8e4m3>& a, const std::vector<fp8e4m3>& b, std::vector<float>& c, 
                       int warmup_iters = 3, int timing_iters = 20) {
    constexpr int threads_per_warp = 64;
    constexpr int warps_per_cu = 4;
    constexpr int threads_per_block = threads_per_warp * warps_per_cu;

    // Ensure input vectors have correct size
    if (a.size() != M * K) {
        fprintf(stderr, "Error: Input vector 'a' size %zu does not match expected M*K=%d\n", a.size(), M*K);
        return {0, 0, 0, 0, 0};
    }
    if (b.size() != N * K) {
        fprintf(stderr, "Error: Input vector 'b' size %zu does not match expected N*K=%d\n", b.size(), N*K);
        return {0, 0, 0, 0, 0};
    }

    // Resize output vector
    c.resize(M * N);

    // Allocate device memory
    fp8e4m3 *d_a, *d_b;
    float *d_c;
    hipMalloc(&d_a, M*K*sizeof(fp8e4m3));
    hipMalloc(&d_b, N*K*sizeof(fp8e4m3));
    hipMalloc(&d_c, M*N*sizeof(float));
    HipCheckError();

    // Copy data to device
    hipMemcpy(d_a, a.data(), M*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b.data(), N*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
    hipMemset(d_c, 0, M*N*sizeof(float));
    HipCheckError();

    // Create global memory objects
    kittens::gl<fp8e4m3, 1, 1, M, K> A(d_a, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<fp8e4m3, 1, 1, N, K> B(d_b, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<float, 1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);
    
    // Warmup iterations
    for (int i = 0; i < warmup_iters; i++) {
        hipMemset(d_c, 0, M*N*sizeof(float));
        matmul_device_ref<M, N, K><<<CUs, threads_per_block>>>(A, B, C);
        hipDeviceSynchronize();
    }
    
    // Create HIP events for precise kernel timing
    hipEvent_t start_event, stop_event;
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);
    
    // Timed kernel-only loop
    std::vector<float> times_ms;
    times_ms.reserve(timing_iters);
    for (int r = 0; r < timing_iters; ++r) {
        hipMemset(d_c, 0, M*N*sizeof(float));
        hipEventRecord(start_event, 0);
        matmul_device_ref<M, N, K><<<CUs, threads_per_block>>>(A, B, C);
        hipEventRecord(stop_event, 0);
        hipEventSynchronize(stop_event);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start_event, stop_event);
        times_ms.push_back(ms);
    }
    
    // Calculate best and average times
    float sum_ms = 0.f, best_ms = 1e30f;
    for (float t : times_ms) { 
        sum_ms += t; 
        best_ms = std::min(best_ms, t); 
    }
    float avg_ms = sum_ms / times_ms.size();
    
    // Calculate TFLOPS (2*M*N*K operations)
    double total_ops = 2.0 * M * N * K;
    double best_tflops = (total_ops / (best_ms * 1e-3)) / 1e12;
    double avg_tflops = (total_ops / (avg_ms * 1e-3)) / 1e12;
    
    // Cleanup events
    hipEventDestroy(start_event);
    hipEventDestroy(stop_event);
    HipCheckError();

    // Copy result back to host
    hipMemcpy(c.data(), d_c, M*N*sizeof(float), hipMemcpyDeviceToHost);
    HipCheckError();

    // Free device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    HipCheckError();
    
    return {best_ms, avg_ms, best_tflops, avg_tflops, timing_iters};
}

#endif

template <int M, int N, int K>
__global__ void matmul_device(const kittens::gl<fp8e4m3, 1, 1, M, K> A, const kittens::gl<fp8e4m3, 1, 1, N, K> B, const kittens::gl<float, 1, 1, M, N> C) {
    // Each threadblock computes 256x256 output tile
    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCK_K = 64;
    constexpr int blocks_per_row = M / BLOCK_SIZE; // Number of blocks per matrix row
    constexpr int blocks_per_col = N / BLOCK_SIZE; // Number of blocks per matrix col
    constexpr int total_blocks_needed = blocks_per_row * blocks_per_col; // Total blocks needed
    constexpr int k_iters = K / BLOCK_K; // K iterations

    // Shared memory tiles: 128x64 for A and B
    __shared__ st<fp8e4m3, BLOCK_SIZE, BLOCK_K> As;
    __shared__ st<fp8e4m3, BLOCK_SIZE, BLOCK_K> Bs;

    // Register tiles: 64x64 per warp
    rt_fp8e4m3<64, 64> a;
    rt_fp8e4m3<128, 64> b;
    rt_fl<64, 128, kittens::ducks::rt_layout::accumulator> c;

    // Calculate how many outer iterations needed based on available threadblocks
    int outer_iters = (total_blocks_needed + gridDim.x - 1) / gridDim.x;

    // Outer loop: iterate until all matrix blocks are covered
    for (int outer = 0; outer < outer_iters; outer++) {
        // Calculate which block this threadblock should work on
        int global_block_id = outer * gridDim.x + blockIdx.x;

        // Early exit if this threadblock has no work in this iteration
        if (global_block_id >= total_blocks_needed) continue;

        // Convert linear block ID to 2D coordinates
        int block_row = global_block_id / blocks_per_col;
        int block_col = global_block_id % blocks_per_col;
        int block_m = block_row * BLOCK_SIZE;
        int block_n = block_col * BLOCK_SIZE;

        // Warp arrangement within threadblock: 4x2 warps covering 256x256
        int warp_m = (warpid() / 2); // warp row: 0 to 3
        int warp_n = (warpid() % 2); // warp col: 0 to 1

        zero(c);

        // Inner loop over K dimension
        for (int k = 0; k < k_iters; k++) {
            // Cooperatively load 128x64 tiles into shared memory
            // All 4 warps participate in loading
            load<2, false, kittens::ducks::rt_layout::row>(As, A, {0, 0, block_row, k});
            load<2, false, kittens::ducks::rt_layout::row>(Bs, B, {0, 0, block_col, k});

            // CRITICAL: Ensure all warps complete loading before any reads from shared memory
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Each warp loads its 64x64 portion from shared memory using subtiles
            auto as_subtile = kittens::subtile_inplace<64, 64>(As, {warp_m, 0});
            auto bs_subtile = kittens::subtile_inplace<128, 64>(Bs, {warp_n, 0});
            load(a, as_subtile);
            load(b, bs_subtile);

            // Ensure shared-to-register loads complete before MMA
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_sched_barrier(0);

            // Compute: C += A * B^T
            mma_ABt(c, a, b, c);
            
            // CRITICAL: Ensure all warps finish using shared memory before next iteration overwrites
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }

        // Store result: each warp stores its 64x64 result
        store(C, c, {0, 0, block_row * 4 + warp_m, block_col * 2 + warp_n});
    }
}

template <int M, int N, int K, int CUs>
TimingResult matmul_host(const std::vector<fp8e4m3>& a, const std::vector<fp8e4m3>& b, std::vector<float>& c, 
                        int warmup_iters = 3, int timing_iters = 20) {
    constexpr int threads_per_warp = 64;
    constexpr int warps_per_cu = 8;
    constexpr int threads_per_block = threads_per_warp * warps_per_cu;
    
    // Ensure input vectors have correct size
    if (a.size() != M * K) {
        fprintf(stderr, "Error: Input vector 'a' size %zu does not match expected M*K=%d\n", a.size(), M*K);
        return {0, 0, 0, 0, 0};
    }
    if (b.size() != N * K) {
        fprintf(stderr, "Error: Input vector 'b' size %zu does not match expected N*K=%d\n", b.size(), N*K);
        return {0, 0, 0, 0, 0};
    }
    
    // Resize output vector
    c.resize(M * N);
    
    // Allocate device memory
    fp8e4m3 *d_a, *d_b;
    float *d_c;
    hipMalloc(&d_a, M*K*sizeof(fp8e4m3));
    hipMalloc(&d_b, N*K*sizeof(fp8e4m3));
    hipMalloc(&d_c, M*N*sizeof(float));
    HipCheckError();
    
    // Copy data to device
    hipMemcpy(d_a, a.data(), M*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b.data(), N*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
    hipMemset(d_c, 0, M*N*sizeof(float));
    HipCheckError();
    
    // Create global memory objects
    kittens::gl<fp8e4m3, 1, 1, M, K> A(d_a, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<fp8e4m3, 1, 1, N, K> B(d_b, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<float, 1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);
    
    // Warmup iterations
    for (int i = 0; i < warmup_iters; i++) {
        hipMemset(d_c, 0, M*N*sizeof(float));
        matmul_device<M, N, K><<<CUs, threads_per_block>>>(A, B, C);
        hipDeviceSynchronize();
    }
    
    // Create HIP events for precise kernel timing
    hipEvent_t start_event, stop_event;
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);
    
    // Timed kernel-only loop
    std::vector<float> times_ms;
    times_ms.reserve(timing_iters);
    for (int r = 0; r < timing_iters; ++r) {
        hipMemset(d_c, 0, M*N*sizeof(float));
        hipEventRecord(start_event, 0);
        matmul_device<M, N, K><<<CUs, threads_per_block>>>(A, B, C);
        hipEventRecord(stop_event, 0);
        hipEventSynchronize(stop_event);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start_event, stop_event);
        times_ms.push_back(ms);
    }
    
    // Calculate best and average times
    float sum_ms = 0.f, best_ms = 1e30f;
    for (float t : times_ms) { 
        sum_ms += t; 
        best_ms = std::min(best_ms, t); 
    }
    float avg_ms = sum_ms / times_ms.size();
    
    // Calculate TFLOPS (2*M*N*K operations)
    double total_ops = 2.0 * M * N * K;
    double best_tflops = (total_ops / (best_ms * 1e-3)) / 1e12;
    double avg_tflops = (total_ops / (avg_ms * 1e-3)) / 1e12;
    
    // Cleanup events
    hipEventDestroy(start_event);
    hipEventDestroy(stop_event);
    HipCheckError();
    
    // Copy result back to host
    hipMemcpy(c.data(), d_c, M*N*sizeof(float), hipMemcpyDeviceToHost);
    HipCheckError();
    
    // Free device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    HipCheckError();
    
    return {best_ms, avg_ms, best_tflops, avg_tflops, timing_iters};
}

// Random initialization function
template <int M, int N, int K>
void random_init(std::vector<fp8e4m3>& a_host, std::vector<fp8e4m3>& b_host) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < M*K; i++) {
        a_host[i] = fp8e4m3(dis(gen));
    }
    for (int i = 0; i < N*K; i++) {
        b_host[i] = fp8e4m3(dis(gen));
    }
}

#ifndef PROFILE

// Identity matrix initialization for easier debugging
// For A*B^T with identity matrices, result should be identity matrix
template <int M, int N, int K>
void identity_init(std::vector<fp8e4m3>& a_host, std::vector<fp8e4m3>& b_host) {
    // Initialize A to identity matrix (M x K)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            if (i == j) {
                a_host[i * K + j] = fp8e4m3(1.0f);
            } else {
                a_host[i * K + j] = fp8e4m3(0.0f);
            }
        }
    }
    
    // Initialize B to identity matrix (N x K)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            if (i == j) {
                b_host[i * K + j] = fp8e4m3(1.0f);
            } else {
                b_host[i * K + j] = fp8e4m3(0.0f);
            }
        }
    }
}

#endif

int main() {
    // Reduced problem size for faster timing
    constexpr int M = 8192;  // 256 threadblocks needed for 2048x2048
    constexpr int N = 8192;  
    constexpr int K = 8192;  // Smaller K for reasonable timing
    constexpr int CUs = 256; // 256 threadblocks (1 outer iteration)
    
    // Timing parameters to keep total runtime reasonable  
    constexpr int warmup_iters = 2;
    constexpr int timing_iters = 20;

    printf("Matrix dimensions: %dx%dx%d, CUs: %d\n", M, N, K, CUs);
    printf("Warmup iterations: %d, Timing iterations: %d\n\n", warmup_iters, timing_iters);

    // Initialize input matrices
    std::vector<fp8e4m3> a_host(M*K);
    std::vector<fp8e4m3> b_host(N*K);
    std::vector<float> c_ref(M*N);
    std::vector<float> c_host(M*N);

    // Test with random matrices now that the kernel works
    random_init<M, N, K>(a_host, b_host);

    #ifndef PROFILE

    // Compute reference result with timing
    printf("Running reference kernel (matmul_device_ref)...\n");
    TimingResult ref_timing = matmul_ref<M, N, K, CUs>(a_host, b_host, c_ref, warmup_iters, timing_iters);

    #endif

    // Compute test result with timing
    printf("Running optimized kernel (matmul_device)...\n");
    TimingResult host_timing = matmul_host<M, N, K, CUs>(a_host, b_host, c_host, warmup_iters, timing_iters);

    #ifndef PROFILE

    bool success = true;
    // Compare GPU result (c_host) with CPU reference (c_ref)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            // c_host is row major: [row*N + col]
            // c_ref is row major: [row*N + col]
            float c_val = c_host[row * N + col];
            float c_ref_val = c_ref[row * N + col];
            float diff = std::abs(c_val - c_ref_val);
            if (diff > 0.4f) {
                printf("Mismatch at (row=%d, col=%d): c_host = %f, c_ref = %f, diff = %f\n", row, col, c_val, c_ref_val, diff);
                success = false;
                break;
            }
        }
        if (!success) {
            break;
        }
    }
    // Performance comparison and results
    printf("\n=== PERFORMANCE RESULTS ===\n");
    
    printf("Reference kernel (matmul_device_ref):\n");
    printf("  Kernel time (best): %.3f ms,  TFLOPS: %.2f\n", ref_timing.best_time_ms, ref_timing.best_tflops);
    printf("  Kernel time (avg ): %.3f ms,  TFLOPS: %.2f\n", ref_timing.avg_time_ms, ref_timing.avg_tflops);
    
    printf("\nOptimized kernel (matmul_device):\n");
    printf("  Kernel time (best): %.3f ms,  TFLOPS: %.2f\n", host_timing.best_time_ms, host_timing.best_tflops);
    printf("  Kernel time (avg ): %.3f ms,  TFLOPS: %.2f\n", host_timing.avg_time_ms, host_timing.avg_tflops);
    
    printf("\nSpeedup (best): %.2fx\n", ref_timing.best_time_ms / host_timing.best_time_ms);
    printf("Speedup (avg ): %.2fx\n", ref_timing.avg_time_ms / host_timing.avg_time_ms);
    
    if (success) {
        printf("\nCorrectness: PASSED\n");
        #ifdef DUMP_TO_CSV
        dump_to_csv("a_host.csv", a_host, M, K);
        dump_to_csv("b_host.csv", b_host, N, K);
        dump_to_csv("c_host.csv", c_host, M, N);
        dump_to_csv("c_ref.csv", c_ref, M, N);
        #endif
    } else {
        printf("\nCorrectness: FAILED\n");
        dump_to_csv("a_host.csv", a_host, M, K);
        dump_to_csv("b_host.csv", b_host, N, K);
        dump_to_csv("c_host.csv", c_host, M, N);
        dump_to_csv("c_ref.csv", c_ref, M, N);
    }

    #endif

    return 0;
}