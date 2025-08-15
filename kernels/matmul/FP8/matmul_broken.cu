#include "kittens.cuh"
#include <random>

using namespace kittens;

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

constexpr int BLOCK_SIZE = 256;
constexpr int K_STEP = 128;
constexpr int BLOCK_WARPS_M = 4;
constexpr int BLOCK_WARPS_N = 2;
constexpr int REG_BLOCK_M = BLOCK_SIZE / BLOCK_WARPS_M;
constexpr int REG_BLOCK_N = BLOCK_SIZE / BLOCK_WARPS_N;
constexpr int DOT_SLICE = 128;

#define NUM_WARPS 8
#define NUM_THREADS (WARP_THREADS * NUM_WARPS)

#define M 8192
#define N 8192
#define K 8192

using _gl_A = gl<fp8e4m3, 1, 1, M, K>;
using _gl_B = gl<fp8e4m3, 1, 1, N, K>;
using _gl_C = gl<float, 1, 1, M, N>;

using G = kittens::group<NUM_WARPS>;

__host__ __device__ inline constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

struct globals {
    _gl_A A;
    _gl_B B;
    _gl_C C;
    dim3 grid() { return dim3((N / BLOCK_SIZE) * (M / BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2) void matmul_device(const globals g) {
    extern __shared__ alignment_dummy __shm[];
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    shared_allocator al((int*)&__shm[0]);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    st_fp8e4m3<BLOCK_SIZE, K_STEP> (&As)[2] = al.allocate<st_fp8e4m3<BLOCK_SIZE, K_STEP>, 2>();
    st_fp8e4m3<BLOCK_SIZE, K_STEP> (&Bs)[2] = al.allocate<st_fp8e4m3<BLOCK_SIZE, K_STEP>, 2>();
    // __shared__ st_fp8e4m3<BLOCK_SIZE, K_STEP> As[2];
    // __shared__ st_fp8e4m3<BLOCK_SIZE, K_STEP> Bs[2];
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    rt_fp8e4m3<REG_BLOCK_M, DOT_SLICE> A_tile;
    rt_fp8e4m3<REG_BLOCK_N, DOT_SLICE> B_tile;
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, kittens::ducks::rt_layout::accumulator> C_accum;
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    zero(C_accum);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    int wgid = blockIdx.x; // flat grid
    const int NUM_WGS = gridDim.x;
    constexpr int NUM_XCDs = 8;
    constexpr int CUS_PER_XCD = 32;
    constexpr int NUM_CUS = NUM_XCDs * CUS_PER_XCD;

    // TODO: add XCD/L2 locality for workgroups
    constexpr int num_pid_m = ceil_div(M, BLOCK_SIZE);
    constexpr int num_pid_n = ceil_div(N, BLOCK_SIZE);
    int pid_m = wgid / num_pid_n;
    int pid_n = wgid % num_pid_n;

    // Assign the tile's row/column based on the pid_m and pid_n
    const int row = pid_m;
    const int col = pid_n;

    const int warp_row = warpid() / BLOCK_WARPS_N;
    const int warp_col = warpid() % BLOCK_WARPS_N;
    constexpr int num_tiles = K / K_STEP;

    int tic = 0;
    int toc = 1;
    using T = typename st_fp8e4m3<BLOCK_SIZE, K_STEP>::dtype; // fp8e4m3
    constexpr int bytes_per_thread = BLOCK_SIZE * K_STEP * sizeof(T) / NUM_THREADS;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);
        printf("warp_row: %d, warp_col: %d\n", warp_row, warp_col);
        printf("row: %d, col: %d\n", row, col);
        printf("REG_BLOCK_M: %d, DOT_SLICE: %d\n", REG_BLOCK_M, DOT_SLICE);
    }

    if (blockIdx.x == 1 && threadIdx.x == 0) {
        printf("blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);
        printf("warp_row: %d, warp_col: %d\n", warp_row, warp_col);
        printf("row: %d, col: %d\n", row, col);
        printf("REG_BLOCK_M: %d, DOT_SLICE: %d\n", REG_BLOCK_M, DOT_SLICE);
    }

    if (blockIdx.x == 0 && threadIdx.x == 64) {
        printf("blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);
        printf("warp_row: %d, warp_col: %d\n", warp_row, warp_col);
        printf("row: %d, col: %d\n", row, col);
        printf("REG_BLOCK_M: %d, DOT_SLICE: %d\n", REG_BLOCK_M, DOT_SLICE);
    }

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_sched_barrier(0);
    
    // Load first tile into shared memory
    load<2, false, kittens::ducks::rt_layout::row, st_fp8e4m3<BLOCK_SIZE, K_STEP>, _gl_A, coord<st_fp8e4m3<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(As[tic], g.A, {0, 0, row, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    load<2, false, kittens::ducks::rt_layout::row, st_fp8e4m3<BLOCK_SIZE, K_STEP>, _gl_B, coord<st_fp8e4m3<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(Bs[tic], g.B, {0, 0, col, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);


    for (int tile = 0; tile < num_tiles - 1; ++tile, tic^=1, toc^=1) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
        // Load to register for tic and load to shared for toc
        load<2, false, kittens::ducks::rt_layout::row, st_fp8e4m3<BLOCK_SIZE, K_STEP>, _gl_A, coord<st_fp8e4m3<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(As[toc], g.A, {0, 0, row, tile + 1});
        __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
        auto as_subtile = kittens::subtile_inplace<REG_BLOCK_M, DOT_SLICE>(As[tic], {warp_row, 0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        if (blockIdx.x == 0 && threadIdx.x == 0 && tile == 0) {
            printf("warp_row: %d\n", warp_row);
            printf("as_subtile 0: %f\n", float(as_subtile.data[0]));
            printf("as_subtile 1: %f\n", float(as_subtile.data[1]));
            printf("as_subtile 128: %f\n", float(as_subtile.data[128]));
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        load(A_tile, as_subtile);
        __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
        load<2, false, kittens::ducks::rt_layout::row, st_fp8e4m3<BLOCK_SIZE, K_STEP>, _gl_B, coord<st_fp8e4m3<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(Bs[toc], g.B, {0, 0, col, tile + 1});
        __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
        auto bs_subtile = kittens::subtile_inplace<REG_BLOCK_N, DOT_SLICE>(Bs[tic], {warp_col, 0});
        __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
        load(B_tile, bs_subtile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        if (blockIdx.x == 0 && threadIdx.x == 0 && tile == 0) {
            printf("warp_row: %d\n", warp_row);
            printf("B_tile 0: %f\n", float4(B_tile.tiles[0][0].data[0]).x);
            printf("B_tile 1: %f\n", float4(B_tile.tiles[0][0].data[0]).y);
        }

        if (blockIdx.x == 0 && threadIdx.x == 0 && tile == 0) {
            printf("warp_row: %d\n", warp_row);
            printf("C_accum 0: %f\n", float(C_accum.tiles[0][0].data[0].x));
        }

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, A_tile, B_tile, C_accum);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        if (blockIdx.x == 0 && threadIdx.x == 0 && tile == 0) {
            printf("warp_row: %d\n", warp_row);
            printf("C_accum 0: %f\n", float(C_accum.tiles[0][0].data[0].x));
        }
        break;
    }

    // need to do last step without new loads to shared
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    // auto as_subtile = kittens::subtile_inplace<REG_BLOCK_M, DOT_SLICE>(As[tic], {warp_row, 0});
    // __builtin_amdgcn_s_waitcnt(0);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);
    // load(A_tile, as_subtile);
    // __builtin_amdgcn_s_waitcnt(0);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);
    // auto bs_subtile = kittens::subtile_inplace<REG_BLOCK_N, DOT_SLICE>(Bs[tic], {warp_col, 0});
    // __builtin_amdgcn_s_waitcnt(0);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);
    // load(B_tile, bs_subtile);
    // __builtin_amdgcn_s_waitcnt(0);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);
    // asm volatile("s_waitcnt lgkmcnt(0)");
    // __builtin_amdgcn_s_waitcnt(0);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);
    // __builtin_amdgcn_s_setprio(1);
    // mma_ABt(C_accum, A_tile, B_tile, C_accum);
    // __builtin_amdgcn_s_setprio(0);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);

    // store to C
    store(g.C, C_accum, {0, 0, row*BLOCK_WARPS_M+warp_row, col*BLOCK_WARPS_N+warp_col});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

void matmul_host(const std::vector<fp8e4m3>& a, const std::vector<fp8e4m3>& b, std::vector<float>& c) {
    // Ensure input vectors have correct size
    if (a.size() != M * K) {
        fprintf(stderr, "Error: Input vector 'a' size %zu does not match expected M*K=%d\n", a.size(), M*K);
        return;
    }
    if (b.size() != N * K) {
        fprintf(stderr, "Error: Input vector 'b' size %zu does not match expected N*K=%d\n", b.size(), N*K);
        return;
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
    
    // Create globals struct and populate with device pointers
    globals g {
        _gl_A(d_a, nullptr, nullptr, nullptr, nullptr),
        _gl_B(d_b, nullptr, nullptr, nullptr, nullptr),
        _gl_C(d_c, nullptr, nullptr, nullptr, nullptr)
    };
    
    // Launch kernel
    matmul_device<<<1, g.block(), g.dynamic_shared_memory()>>>(g);
    hipDeviceSynchronize();
    HipCheckError();
    
    // Copy result back to host
    hipMemcpy(c.data(), d_c, M*N*sizeof(float), hipMemcpyDeviceToHost);
    HipCheckError();
    
    // Free device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    HipCheckError();
}

// Random initialization function
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

// Identity matrix initialization for easier debugging
// For A*B^T with identity matrices, result should be identity matrix
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

int main() {
    printf("Matrix dimensions: %dx%dx%d\n", M, N, K);

    // Initialize input matrices
    std::vector<fp8e4m3> a_host(M*K, 0.0f);
    std::vector<fp8e4m3> b_host(N*K, 0.0f);
    std::vector<float> c_host(M*N, 0.0f);

    // Initialize with random data
    // random_init(a_host, b_host);
    // identity_init(a_host, b_host);
    
    printf("Running matmul kernel...\n");
    
    // Run the kernel
    matmul_host(a_host, b_host, c_host);
    
    printf("Kernel execution completed successfully!\n");
    printf("First few results: c[0]=%f, c[1]=%f, c[2]=%f\n", 
           c_host[0], c_host[1], c_host[2]);
    
    // INSERT_YOUR_CODE
    FILE* f = fopen("c_host_fast.csv", "w");
    if (f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                fprintf(f, "%f", c_host[i * N + j]);
                if (j < N - 1) fprintf(f, ",");
            }
            fprintf(f, "\n");
        }
        fclose(f);
        printf("Dumped c_host to c_host_fast.csv\n");
    } else {
        printf("Failed to open c_host_fast.csv for writing\n");
    }
    
    return 0;
}