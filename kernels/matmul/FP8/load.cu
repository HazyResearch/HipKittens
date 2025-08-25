#include "kittens.cuh"

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


constexpr int M = 32;
constexpr int N = 128;
constexpr int NUM_WARPS = 2;

__global__ inline void load_gl_to_st(gl<fp8e4m3, 1, 1, M, N> dst, gl<fp8e4m3, 1, 1, M, N> src)
{
    __shared__ st<fp8e4m3, M, N> As;
    rt_fp8e4m3<M, N/NUM_WARPS> a;

    load<2, false, kittens::ducks::rt_layout::row, st<fp8e4m3, M, N>, kittens::gl<fp8e4m3, 1, 1, M, N>, coord<st<fp8e4m3, M, N>>, NUM_WARPS*WARP_THREADS>(As, src, {0, 0, 0, 0});

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto as_subtile = kittens::subtile_inplace<M, N/NUM_WARPS>(As, {0, warpid() % 2});
    load(a, as_subtile);

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);

    store(dst, a, {0, 0, 0, warpid() % 2});
}

int main() {
    int warps = NUM_WARPS;
    int threads = warps * WARP_THREADS;

    fp8e4m3 *d_a, *d_b;
    hipMalloc(&d_a, M*N*sizeof(fp8e4m3));
    hipMalloc(&d_b, M*N*sizeof(fp8e4m3));
    HipCheckError();

    std::vector<fp8e4m3> a(M*N);
    for (int i = 0; i < M * N; ++i) {
        a[i] = fp8e4m3(float(i));
    }
    
    hipMemcpy(d_a, a.data(), M*N*sizeof(fp8e4m3), hipMemcpyHostToDevice);
    hipMemset(d_b, 0, M*N*sizeof(fp8e4m3));
    HipCheckError();

    kittens::gl<fp8e4m3, 1, 1, M, N> A(d_a, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<fp8e4m3, 1, 1, M, N> B(d_b, nullptr, nullptr, nullptr, nullptr);

    load_gl_to_st<<<1, threads>>>(B, A);
    HipCheckError();

    std::vector<fp8e4m3> b(M*N);
    hipMemcpy(b.data(), d_b, M*N*sizeof(fp8e4m3), hipMemcpyDeviceToHost);

    // Dump a to a_out.csv and b to b_out.csv, no leading zero padding, no clamping
    {
        FILE* fa = fopen("a_out.csv", "w");
        FILE* fb = fopen("b_out.csv", "w");
        if (!fa || !fb) {
            fprintf(stderr, "Error opening output files!\n");
            if (fa) fclose(fa);
            if (fb) fclose(fb);
        } else {
            // Dump a
            for (int i = 0; i < M * N; ++i) {
                float val = float(a[i]);
                int intval = static_cast<int>(val + 0.5f); // round to nearest int
                fprintf(fa, "%d", intval);
                if ((i + 1) % N == 0)
                    fprintf(fa, "\n");
                else
                    fprintf(fa, ",");
            }
            // Dump b
            for (int i = 0; i < M * N; ++i) {
                float val = float(b[i]);
                int intval = static_cast<int>(val + 0.5f); // round to nearest int
                fprintf(fb, "%d", intval);
                if ((i + 1) % N == 0)
                    fprintf(fb, "\n");
                else
                    fprintf(fb, ",");
            }
            fclose(fa);
            fclose(fb);
        }
    }
    printf("\n");

    // INSERT_YOUR_CODE
    // Check if b is equal to a and print first mismatch if any
    bool all_equal = true;
    for (int i = 0; i < M * N; ++i) {
        float a_val = float(a[i]);
        float b_val = float(b[i]);
        if (a_val != b_val) {
            printf("Mismatch at index %d: a = %f, b = %f\n", i, a_val, b_val);
            all_equal = false;
            break;
        }
    }
    if (all_equal) {
        printf("b is equal to a\n");
    }

    return 0;
}