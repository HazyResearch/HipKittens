#include "kittens.cuh"
#include <random>
#include <omp.h>

using namespace kittens;

// #define DUMP_TO_CSV

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
__global__ void matmul(const kittens::gl<fp8e4m3, 1, 1, M, K> A, const kittens::gl<fp8e4m3, 1, 1, N, K> B, const kittens::gl<float, 1, 1, M, N> C) {
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

int main() {
    // Allow different M and N dimensions
    constexpr int M = 8192;  // Can be any multiple of 64
    constexpr int N = 8192;  // Can be any multiple of 64
    constexpr int K = 8192;
    constexpr int threads_per_warp = 64;
    constexpr int warps_per_cu = 4;
    constexpr int threads_per_block = threads_per_warp * warps_per_cu;
    constexpr int CUs = 256;

    fp8e4m3 *d_a, *d_b;
    float *d_c;
    hipMalloc(&d_a, M*K*sizeof(fp8e4m3));
    hipMalloc(&d_b, N*K*sizeof(fp8e4m3));
    hipMalloc(&d_c, M*N*sizeof(float));
    HipCheckError();

    std::vector<fp8e4m3> a_host(M*K);
    std::vector<fp8e4m3> b_host(N*K);
    // Randomly initialize b_host with values in [-1.0, 1.0]
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> c_ref(M*N, 0.0f);
    
    // Initialize b_host (N x K matrix)
    for (int i = 0; i < N*K; i++) {
        b_host[i] = fp8e4m3(dis(gen));
    }

    // Initialize a_host to identity matrix
    // a_host is M x K, but we want to set a_host[i*K + k] = 1.0 if i == k, else 0.0
    // However, since K may be larger than M, we only set the diagonal up to min(M, K)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            if (i == k) {
                a_host[i*K + k] = fp8e4m3(1.0f);
            } else {
                a_host[i*K + k] = fp8e4m3(0.0f);
            }
        }
    }
    
    // Compute reference result: C = A * B^T
    // A is M x K (identity in first min(M,K) diagonal), B is N x K
    // C should be M x N
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = float(a_host[i*K + k]);
                float b_val = float(b_host[j*K + k]);
                sum += a_val * b_val;
            }
            c_ref[i*N + j] = sum;
        }
    }

    hipMemcpy(d_a, a_host.data(), M*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b_host.data(), N*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
    HipCheckError();

    kittens::gl<fp8e4m3, 1, 1, M, K> A(d_a, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<fp8e4m3, 1, 1, N, K> B(d_b, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<float, 1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);
    matmul<M, N, K><<<CUs, threads_per_block>>>(A, B, C);
    HipCheckError();

    std::vector<float> c_host(M*N);
    hipMemcpy(c_host.data(), d_c, M*N*sizeof(float), hipMemcpyDeviceToHost);

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
    // }
    if (success) {
        printf("Test passed\n");
        #ifdef DUMP_TO_CSV
        dump_to_csv("a_host.csv", a_host, M, K);
        dump_to_csv("b_host.csv", b_host, N, K);
        dump_to_csv("c_host.csv", c_host, M, N);
        dump_to_csv("c_ref.csv", c_ref, M, N);
        #endif
    } else {
        printf("Test failed\n");
        dump_to_csv("a_host.csv", a_host, M, K);
        dump_to_csv("b_host.csv", b_host, N, K);
        dump_to_csv("c_host.csv", c_host, M, N);
        dump_to_csv("c_ref.csv", c_ref, M, N);
    }
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    return 0;
}