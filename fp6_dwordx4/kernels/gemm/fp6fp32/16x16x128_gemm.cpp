
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
constexpr int REG_BLOCK_M      = BLOCK_SIZE_M / 2;
constexpr int REG_BLOCK_N      = BLOCK_SIZE_N / 2;
              

#define NUM_WARPS 4
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
    dim3 grid()  { return dim3((N / BLOCK_SIZE_N) * (M / BLOCK_SIZE_M)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

/**
   * @brief Transform a workgroup ID to a new workgroup ID based on the chunk size and number of XCDs.
   * @param workgroup_id The original workgroup ID.
   * @param num_workgroups The total number of workgroups.
   * @param num_xcds The number of XCDs.
   * @param chunk_size The chunk size.
   * @return The new workgroup ID.
   */
   __host__ __device__ inline int chiplet_transform_chunked(
    int workgroup_id, 
    int num_workgroups,
    int num_xcds,
    int chunk_size 
) {
    // Current XCD
    int xcd = workgroup_id % num_xcds;

    // Largest full (NUM_XCDS*CHUNK_SIZE)-aligned block
    int block = num_xcds * chunk_size;
    int limit = (num_workgroups / block) * block;

    // If pid beyond the last full block, leave unchanged
    if (workgroup_id > limit) return workgroup_id;

    // Local PID (within round-robin assignment)
    int local_pid    = workgroup_id / num_xcds;
    int chunk_idx    = local_pid / chunk_size;
    int pos_in_chunk = local_pid % chunk_size;

    // New PID
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_f6<REG_BLOCK_M, K_STEP> (&As)[4] = al.allocate<st_f6<REG_BLOCK_M, K_STEP>, 4>();
    st_f6<REG_BLOCK_N, K_STEP> (&Bs)[4] = al.allocate<st_f6<REG_BLOCK_N, K_STEP>, 4>();

    uintptr_t A_base = reinterpret_cast<uintptr_t>(&As[0]);
    uintptr_t B_base = reinterpret_cast<uintptr_t>(&Bs[0]);

    st_f6<REG_BLOCK_M, K_STEP> *As_ptrs[4] = {
        reinterpret_cast<st_f6<REG_BLOCK_M, K_STEP>*>(A_base + (reinterpret_cast<uintptr_t>(&As[0]) - A_base) * 6 / 8),
        reinterpret_cast<st_f6<REG_BLOCK_M, K_STEP>*>(A_base + (reinterpret_cast<uintptr_t>(&As[1]) - A_base) * 6 / 8),
        reinterpret_cast<st_f6<REG_BLOCK_M, K_STEP>*>(A_base + (reinterpret_cast<uintptr_t>(&As[2]) - A_base) * 6 / 8),
        reinterpret_cast<st_f6<REG_BLOCK_M, K_STEP>*>(A_base + (reinterpret_cast<uintptr_t>(&As[3]) - A_base) * 6 / 8)
    };

    st_f6<REG_BLOCK_N, K_STEP> *Bs_ptrs[4] = {
        reinterpret_cast<st_f6<REG_BLOCK_N, K_STEP>*>(B_base + (reinterpret_cast<uintptr_t>(&Bs[0]) - B_base) * 6 / 8),
        reinterpret_cast<st_f6<REG_BLOCK_N, K_STEP>*>(B_base + (reinterpret_cast<uintptr_t>(&Bs[1]) - B_base) * 6 / 8),
        reinterpret_cast<st_f6<REG_BLOCK_N, K_STEP>*>(B_base + (reinterpret_cast<uintptr_t>(&Bs[2]) - B_base) * 6 / 8),
        reinterpret_cast<st_f6<REG_BLOCK_N, K_STEP>*>(B_base + (reinterpret_cast<uintptr_t>(&Bs[3]) - B_base) * 6 / 8)
    };

    rt_f6<REG_BLOCK_M/2, K_STEP> A_tile[2];
    rt_f6<REG_BLOCK_N/2, K_STEP> B_tile[2];
    rt_fl<REG_BLOCK_M/2, REG_BLOCK_N/2, ducks::rt_layout::accumulator> C_accum[2][2];
    zero(C_accum[0][0]);
    zero(C_accum[0][1]);
    zero(C_accum[1][0]);
    zero(C_accum[1][1]);

    // Original WGID.
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS  = gridDim.x * gridDim.y;
    const int NUM_XCDS = 8;
    const int WGM = 8;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    // Swizzle for better L2 within the same XCD.
    const int num_pid_m = ceil_div(M, BLOCK_SIZE_M); // 7680 / 192 = 40
    const int num_pid_n = ceil_div(N, BLOCK_SIZE_N); // 7680 / 256 = 30
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    int row = pid_m; 
    int col = pid_n;

    // Info
    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;
    const int num_tiles = K / K_STEP;

    int tic = 0;
    int toc = 1;
    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile_A = (REG_BLOCK_M * K_STEP * 6 / 8) / (bytes_per_thread * NUM_THREADS);
    constexpr int memcpy_per_tile_B = (REG_BLOCK_N * K_STEP * 6 / 8) / (bytes_per_thread * NUM_THREADS);

    // Register array to store swizzled global addresses for each thread.
    uint32_t swizzled_offsets_A[memcpy_per_tile_A];
    uint32_t swizzled_offsets_B[memcpy_per_tile_B];

    prefill_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row, 0}, *As_ptrs[0], swizzled_offsets_A);
    prefill_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col, 0}, *Bs_ptrs[0], swizzled_offsets_B);

    const uint32_t addrA = prefill_swizzled_offset_fp6(A_tile[0], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[0], {warp_row, 0}));
    // const uint32_t addrA1 = prefill_swizzled_offset_fp6(A_tile[1], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[1], {warp_row, 0}));
    // const uint32_t addrA2 = prefill_swizzled_offset_fp6(A_tile[0], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[2], {warp_row, 0}));
    // const uint32_t addrA3 = prefill_swizzled_offset_fp6(A_tile[1], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[3], {warp_row, 0}));
    const uint32_t addrB = prefill_swizzled_offset_fp6(B_tile[0], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[0], {warp_col, 0}));
    // const uint32_t addrB1 = prefill_swizzled_offset_fp6(B_tile[1], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[1], {warp_col, 0}));
    // const uint32_t addrB2 = prefill_swizzled_offset_fp6(B_tile[0], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[2], {warp_col, 0}));
    // const uint32_t addrB3 = prefill_swizzled_offset_fp6(B_tile[1], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[3], {warp_col, 0}));
    // uint32_t addrA[4] = {addrA0, addrA1, addrA2, addrA3};
    // uint32_t addrB[4] = {addrB0, addrB1, addrB2, addrB3};

    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row*2, 0}, *As_ptrs[tic*2], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col*2, 0}, *Bs_ptrs[tic*2], swizzled_offsets_B);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col*2+1, 0}, *Bs_ptrs[tic*2+1], swizzled_offsets_B);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row*2+1, 0}, *As_ptrs[tic*2+1], swizzled_offsets_A);

    asm volatile("s_waitcnt vmcnt(9)");
    __builtin_amdgcn_s_barrier();

    load_lds_reg_row_fp6(A_tile[0], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[tic*2], {warp_row, 0}), addrA);

    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row*2, 1}, *As_ptrs[toc*2], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col*2, 1}, *Bs_ptrs[toc*2], swizzled_offsets_B);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col*2+1, 1}, *Bs_ptrs[toc*2+1], swizzled_offsets_B);

    asm volatile("s_waitcnt vmcnt(15)");
    __builtin_amdgcn_s_barrier();

    load_lds_reg_row_fp6(B_tile[0], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[tic*2], {warp_col, 0}), addrB);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row*2+1, 1}, *As_ptrs[toc*2+1], swizzled_offsets_A);

    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(12)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 2; ++tile, tic ^= 1, toc ^= 1) {
        load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row*2, tile+2}, *As_ptrs[tic*2], swizzled_offsets_A);
        load_lds_reg_row_fp6(B_tile[1], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[tic*2+1], {warp_col, 0}), addrB);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[0][0], A_tile[0], B_tile[0], C_accum[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");

        load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col*2, tile+2}, *Bs_ptrs[tic*2], swizzled_offsets_B);
        load_lds_reg_row_fp6(A_tile[1], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[tic*2+1], {warp_row, 0}), addrA);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[0][1], A_tile[0], B_tile[1], C_accum[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt vmcnt(12)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col*2+1, tile+2}, *Bs_ptrs[tic*2+1], swizzled_offsets_B);
        load_lds_reg_row_fp6(A_tile[0], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[toc*2], {warp_row, 0}), addrA);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[1][0], A_tile[1], B_tile[0], C_accum[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");

        load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row*2+1, tile+2}, *As_ptrs[tic*2+1], swizzled_offsets_A);
        load_lds_reg_row_fp6(B_tile[0], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[toc*2], {warp_col, 0}), addrB);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[1][1], A_tile[1], B_tile[1], C_accum[1][1]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt vmcnt(12)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }
    // epilogue num_tiles - 2
    {
        load_lds_reg_row_fp6(B_tile[1], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[tic*2+1], {warp_col, 0}), addrB);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[0][0], A_tile[0], B_tile[0], C_accum[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");

        load_lds_reg_row_fp6(A_tile[1], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[tic*2+1], {warp_row, 0}), addrA);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[0][1], A_tile[0], B_tile[1], C_accum[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt vmcnt(6)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        load_lds_reg_row_fp6(A_tile[0], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[toc*2], {warp_row, 0}), addrA);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[1][0], A_tile[1], B_tile[0], C_accum[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");

        load_lds_reg_row_fp6(B_tile[0], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[toc*2], {warp_col, 0}), addrB);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[1][1], A_tile[1], B_tile[1], C_accum[1][1]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        tic ^= 1;
        toc ^= 1;
    }
    // epilogue num_tiles - 1
    {
        load_lds_reg_row_fp6(B_tile[1], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[tic*2+1], {warp_col, 0}), addrB);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[0][0], A_tile[0], B_tile[0], C_accum[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");

        load_lds_reg_row_fp6(A_tile[1], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[tic*2+1], {warp_row, 0}), addrA);
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[0][1], A_tile[0], B_tile[1], C_accum[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(C_accum[1][0], A_tile[1], B_tile[0], C_accum[1][0]);
        mma_ABt(C_accum[1][1], A_tile[1], B_tile[1], C_accum[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }

    store(g.c, C_accum[0][0], {0, 0, (row * 2)*2 + warp_row, (col * 2)*2 + warp_col});
    store(g.c, C_accum[0][1], {0, 0, (row * 2)*2 + warp_row, (col * 2 + 1)*2 + warp_col});
    store(g.c, C_accum[1][0], {0, 0, (row * 2 + 1)*2+warp_row, (col * 2)*2 + warp_col});
    store(g.c, C_accum[1][1], {0, 0, (row * 2 + 1)*2+warp_row, (col * 2 + 1)*2+warp_col});
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

constexpr int ROTATING_BUFFER_COUNT = ((((1024*1024)/M)*512)/K); // 1000 MiB

// Random initialization function
void random_init(din* a_host, din* b_host, uint32_t seed = 42) {
    std::mt19937 gen(seed); // Seed for reproducibility
    std::normal_distribution<float> dis(-1.0f, 1.0f);
    // #pragma omp parallel for
    for (int i = 0; i < M*K; i++) {
        a_host[i] = din(dis(gen));
    }   
    // #pragma omp parallel for
    for (int i = 0; i < N*K; i++) {
        b_host[i] = din(dis(gen));
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

    // Print first few packed words for debugging (using first buffer)
    random_init(h_input_a, h_input_b, 42);
    pack(h_input_a_packed, h_input_a, M * K);
    pack(h_input_b_packed, h_input_b, N * K);
    
    std::cout << "First 4 packed words of A: ";
    for (int i = 0; i < 4 && i < total_words_a; i++) {
        std::cout << "0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << h_input_a_packed[i] << " ";
    }
    std::cout << std::dec << "\n\n";

    constexpr int block_count = ROTATING_BUFFER_COUNT;

    din *d_input_a_packed;
    din *d_input_b_packed;
    dout *d_output;
    hipMalloc(&d_input_a_packed, block_count * total_bytes_a);
    hipMalloc(&d_input_b_packed, block_count * total_bytes_b);
    hipMalloc(&d_output, M * N * sizeof(dout));

    // Pre-initialize all buffer sections with random data on host
    printf("Initializing %d rotating buffer sections (%zu MB total, A+B only)...\n",
           block_count,
           (block_count * (M*K*6/8 + N*K*6/8) + M*N*sizeof(half)) / (1024*1024));

    for (int block = 0; block < block_count; ++block) {
        // Generate random data with different seed for each buffer
        random_init(h_input_a, h_input_b, 42 + block);

        pack(h_input_a_packed, h_input_a, M * K);
        pack(h_input_b_packed, h_input_b, N * K);

        // Copy to offset position in device memory
        hipMemcpy(reinterpret_cast<uint32_t*>(d_input_a_packed) + block * total_words_a, h_input_a_packed, total_bytes_a, hipMemcpyHostToDevice);
        hipMemcpy(reinterpret_cast<uint32_t*>(d_input_b_packed) + block * total_words_b, h_input_b_packed, total_bytes_b, hipMemcpyHostToDevice);
    }
    hipDeviceSynchronize();
    printf("Buffer initialization complete.\n");

    // Warmup
    const int WARMUP_REPS = 500;
    for (int r = 0; r < WARMUP_REPS; ++r) { 
        int block_idx = r % block_count;
        din* d_a_current = reinterpret_cast<din*>(reinterpret_cast<uint32_t*>(d_input_a_packed) + block_idx * total_words_a);
        din* d_b_current = reinterpret_cast<din*>(reinterpret_cast<uint32_t*>(d_input_b_packed) + block_idx * total_words_b);

        hipMemset(d_output, 0, M*N*sizeof(half));

        _gl_A input_gl_a(d_a_current, 1, 1, M, K);
        _gl_B input_gl_b(d_b_current, 1, 1, N, K);
        _gl_C output_gl(d_output, 1, 1, M, N);

        micro_globals globals{input_gl_a, input_gl_b, output_gl};

        micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory(), stream>>>(globals);
    }
    hipDeviceSynchronize();

    // Timed kernel-only loop
    const int REPS = 100;
    std::vector<float> times_ms;
    times_ms.reserve(REPS);
    for (int r = 0; r < REPS; ++r) {
        int block_idx = r % block_count;
        din* d_a_current = reinterpret_cast<din*>(reinterpret_cast<uint32_t*>(d_input_a_packed) + block_idx * total_words_a);
        din* d_b_current = reinterpret_cast<din*>(reinterpret_cast<uint32_t*>(d_input_b_packed) + block_idx * total_words_b);

        hipMemset(d_output, 0, M*N*sizeof(half));

        _gl_A input_gl_a(d_a_current, 1, 1, M, K);
        _gl_B input_gl_b(d_b_current, 1, 1, N, K);
        _gl_C output_gl(d_output, 1, 1, M, N);

        micro_globals globals{input_gl_a, input_gl_b, output_gl};

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

    int last_buffer_idx = (REPS - 1) % ROTATING_BUFFER_COUNT;
    random_init(h_input_a, h_input_b, 42 + last_buffer_idx);

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