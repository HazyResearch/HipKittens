
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
constexpr int WARPS_COL = 2;

constexpr int REG_BLOCK_M      = BLOCK_SIZE_M / 2 / WARPS_ROW;
constexpr int REG_BLOCK_N      = BLOCK_SIZE_N / 2 / WARPS_COL;
              

#define NUM_WARPS 4
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

template<typename D, typename A, typename B, typename C>
__device__ inline void mma_ABt_base_wrapper(D& d_mma, const A& a_mma, const B& b_mma, const C& c_mma, int n, int m, int k) {
    using T = typename D::dtype;
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    mma_ABt_base(
        d_mma.tiles[n][m],
        a_mma.tiles[n][k],
        b_mma.tiles[m][k],
        c_mma.tiles[n][m]
    );
}

template <typename ST, int N_THREADS = WARP_THREADS>
__device__ inline void load_global_to_shared_direct_unit_fp6(int i, const uint8_t* lds_base, i32x4 srsrc, int row_stride)
{
    constexpr int bytes_per_thread = 16;

    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    const int warp_id = warpid();
    const int laneid = kittens::laneid();
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;

    const int shared_base_tile_cols = 128;
    const int shared_base_tile_rows = 16;

    const int bytes_per_shared_base_tile = shared_base_tile_cols * shared_base_tile_rows;
    const int shared_base_tiles_per_row = ST::cols / shared_base_tile_cols;


    const int byte_offset = ((i * num_warps + warp_id) * bytes_per_warp) + (laneid * bytes_per_thread);
    const int tile_id = byte_offset / bytes_per_shared_base_tile;
    const int tile_row_offset = tile_id / shared_base_tiles_per_row;
    const int tile_col_offset = tile_id % shared_base_tiles_per_row;

    const int base_tile_byte_offset = byte_offset % bytes_per_shared_base_tile;
    const int swizzled_base_tile_byte_offset = base_tile_byte_offset ^ (((base_tile_byte_offset % (8*128)) >> 7) << 4);
        
    const int swizzled_col_offset = (swizzled_base_tile_byte_offset % (shared_base_tile_cols * 1)) / 1;
    const int swizzled_row_offset = swizzled_base_tile_byte_offset / (shared_base_tile_cols * 1);

    const int col_offset_in_global = tile_col_offset * shared_base_tile_cols + swizzled_col_offset;
    const int row_offset_in_global = tile_row_offset * shared_base_tile_rows + swizzled_row_offset;

    const int offset_in_global = (row_offset_in_global * row_stride + col_offset_in_global) * 6 / 8;

    const uint8_t* lds_elem_ptr = lds_base + i * N_THREADS * bytes_per_thread;
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)reinterpret_cast<uintptr_t>(lds_elem_ptr);

    llvm_amdgcn_raw_buffer_load_lds(
        srsrc, // buffer resource
        lds_ptr,
        12, // 12 bytes is specified but the thread actually writes 16 bytes, last 4 garbage
        offset_in_global,
        0, 
        0, // instruction offset
        static_cast<index_t>(coherency::cache_all)); // cache coherency

}

template<typename ST_GL, typename GL_GL, typename ST, typename RT, typename RT_A, typename RT_B, typename RT_C, ducks::coord::tile COORD=coord<ST_GL>>
__device__ inline static void do_interleaved_cluster(
    ST_GL& dst_st, const GL_GL& src_gl, COORD idx,
    RT& dst, const ST& src, 
    RT_A& a, RT_B& b, RT_C& c
) {
    // __builtin_amdgcn_sched_barrier(0);
    mma_ABt_base_wrapper(c, a, b, c, 0, 0, 0);
    // __builtin_amdgcn_sched_barrier(0);
    const int laneid = kittens::laneid();
    const int warp_id = warpid();

    // global memory to shared memory loads setup
    using T_GL = typename ST_GL::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile =  ST_GL::rows * ST_GL::cols * sizeof(T_GL) / (bytes_per_thread * NUM_THREADS); 

    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    const int row_stride_gl_to_st = src_gl.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    auto* global_ptr = reinterpret_cast<const uint8_t*>(&src_gl[unit_coord]);
    i32x4 srsrc = make_srsrc(global_ptr, row_stride_gl_to_st * ST_GL::rows * 6 / 8);

    auto* lds_bytes = reinterpret_cast<uint8_t*>(&dst_st.data[0]);
    const uint8_t* lds_base = lds_bytes + warp_id * bytes_per_warp;

    // shared to register tile loads setup
    static_assert(RT::height == ST::height, "register tile / shared tile height");
    static_assert(RT::width  == ST::width,  "register tile / shared tile width");
    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    auto* lds_bytes_st_to_reg = reinterpret_cast<const uint8_t*>(&src.data[0]);
    const int tile_stride = 16 * 128;
    const int row_stride_st_to_reg = tile_stride * src.underlying_width;    
    const int row_offset = laneid % 16;
    const int col_byte_offset = 32 * (laneid / 16);
    uint32_t byte_offset = (row_offset * 128 + col_byte_offset);
    uint32_t swizzled_byte_offset = byte_offset ^ (((byte_offset % (8*128)) >> 7) << 4);
    uint32_t byte_offset_second = byte_offset + 16;
    uint32_t swizzled_byte_offset_second = byte_offset_second ^ (((byte_offset_second % (8*128)) >> 7) << 4);
    uint32_t addr_0 = reinterpret_cast<uintptr_t>(lds_bytes_st_to_reg + swizzled_byte_offset);
    uint32_t addr_1 = reinterpret_cast<uintptr_t>(lds_bytes_st_to_reg + swizzled_byte_offset_second);

    /*******
    * global to shared: 
    * shared to register: remember base tile is 16x128 so only i goes 0, 1, 2, 3 and j = 0.
    ********/
    {
        __builtin_amdgcn_sched_barrier(0);
        mma_ABt_base_wrapper(c, a, b, c, 0, 1, 0);
        __builtin_amdgcn_sched_barrier(0);

        {
            load_global_to_shared_direct_unit_fp6<ST_GL,NUM_THREADS>(0, lds_base, srsrc, row_stride_gl_to_st);
            constexpr int i = 0;
            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][0].data[0]))))
                : "v"(addr_0),
                "i"(i * row_stride_st_to_reg)
                : "memory"
            );

            // __builtin_amdgcn_sched_barrier(0);
            mma_ABt_base_wrapper(c, a, b, c, 0, 2, 0);
            // __builtin_amdgcn_sched_barrier(0);

            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][0].data[0]) + 12)))
                : "v"(addr_1),
                "i"(i * row_stride_st_to_reg)
                : "memory"
            );

            // __builtin_amdgcn_sched_barrier(0);
            mma_ABt_base_wrapper(c, a, b, c, 0, 3, 0);
            // __builtin_amdgcn_sched_barrier(0);
        }
        {
            load_global_to_shared_direct_unit_fp6<ST_GL,NUM_THREADS>(1, lds_base, srsrc, row_stride_gl_to_st);
            constexpr int i = 1;
            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][0].data[0]))))
                : "v"(addr_0),
                "i"(i * row_stride_st_to_reg)
                : "memory"
            );
            // __builtin_amdgcn_sched_barrier(0);
            mma_ABt_base_wrapper(c, a, b, c, 1, 0, 0);
            mma_ABt_base_wrapper(c, a, b, c, 1, 1, 0);
            // __builtin_amdgcn_sched_barrier(0);

            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][0].data[0]) + 12)))
                : "v"(addr_1),
                "i"(i * row_stride_st_to_reg)
                : "memory"
            );
            // __builtin_amdgcn_sched_barrier(0);
            mma_ABt_base_wrapper(c, a, b, c, 1, 2, 0);
            mma_ABt_base_wrapper(c, a, b, c, 1, 3, 0);
            // __builtin_amdgcn_sched_barrier(0);
        }
        {
            load_global_to_shared_direct_unit_fp6<ST_GL,NUM_THREADS>(2, lds_base, srsrc, row_stride_gl_to_st);
            constexpr int i = 2;
            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][0].data[0]))))
                : "v"(addr_0),
                "i"(i * row_stride_st_to_reg)
                : "memory"
            );
            // __builtin_amdgcn_sched_barrier(0);
            mma_ABt_base_wrapper(c, a, b, c, 2, 0, 0);
            mma_ABt_base_wrapper(c, a, b, c, 2, 1, 0);
            // __builtin_amdgcn_sched_barrier(0);

            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][0].data[0]) + 12)))
                : "v"(addr_1),
                "i"(i * row_stride_st_to_reg)
                : "memory"
            );
            // __builtin_amdgcn_sched_barrier(0);
            mma_ABt_base_wrapper(c, a, b, c, 2, 2, 0);
            mma_ABt_base_wrapper(c, a, b, c, 2, 3, 0);
            // __builtin_amdgcn_sched_barrier(0);
        }
        {
            load_global_to_shared_direct_unit_fp6<ST_GL,NUM_THREADS>(3, lds_base, srsrc, row_stride_gl_to_st);
            constexpr int i = 3;
            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][0].data[0]))))
                : "v"(addr_0),
                "i"(i * row_stride_st_to_reg)
                : "memory"
            );
            // __builtin_amdgcn_sched_barrier(0);
            mma_ABt_base_wrapper(c, a, b, c, 3, 0, 0);
            mma_ABt_base_wrapper(c, a, b, c, 3, 1, 0);
            // __builtin_amdgcn_sched_barrier(0);

            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][0].data[0]) + 12)))
                : "v"(addr_1),
                "i"(i * row_stride_st_to_reg)
                : "memory"
            );
            // __builtin_amdgcn_sched_barrier(0);
            mma_ABt_base_wrapper(c, a, b, c, 3, 2, 0);
            mma_ABt_base_wrapper(c, a, b, c, 3, 3, 0);
            // __builtin_amdgcn_sched_barrier(0);
        }
    }
}








__global__ __launch_bounds__(NUM_THREADS, 1)
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

    // Convert linear block ID to 2D coordinates
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int block_m = block_row * BLOCK_SIZE_M;
    int block_n = block_col * BLOCK_SIZE_N;

    // Info
    const int warp_id = kittens::warpid();
    const int warp_m = (warpid() / WARPS_COL);
    const int warp_n = (warpid() % WARPS_COL);

    int tic = 0;
    int toc = 1;
    constexpr int bytes_per_thread = 12;
    constexpr int memcpy_per_tile_A = (HALF_BLOCK_SIZE_M * K_STEP * 6 / 8) / (bytes_per_thread * NUM_THREADS);
    constexpr int memcpy_per_tile_B = (HALF_BLOCK_SIZE_N * K_STEP * 6 / 8) / (bytes_per_thread * NUM_THREADS);
    // Register array to store swizzled global addresses for each thread.
    uint32_t swizzled_offsets_A[memcpy_per_tile_A];
    uint32_t swizzled_offsets_B[memcpy_per_tile_B];
    prefill_swizzled_offsets_fp6<2, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row*WARPS_ROW, 0}, As[tic][0], swizzled_offsets_A);
    prefill_swizzled_offsets_fp6<2, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col*WARPS_COL, 0}, Bs[tic][0], swizzled_offsets_B);

    /***************************************************************/
    __builtin_amdgcn_sched_barrier(0);
    RT_A a[2];
    RT_B b[2];
    RT_C c[2][2];
    zero(c[0][0]);
    zero(c[0][1]);
    zero(c[1][0]);
    zero(c[1][1]);

    // initial loads 
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row*WARPS_ROW, 0}, As[tic][0], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col*WARPS_COL, 0}, Bs[tic][0], swizzled_offsets_B);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row*WARPS_ROW+1, 0}, As[tic][1], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col*WARPS_COL+1, 0}, Bs[tic][1], swizzled_offsets_B);

    // // more loads
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row*WARPS_ROW, 1}, As[toc][0], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col*WARPS_COL, 1}, Bs[toc][0], swizzled_offsets_B);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row*WARPS_ROW+1, 1}, As[toc][1], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col*WARPS_COL+1, 1}, Bs[toc][1], swizzled_offsets_B);

    __builtin_amdgcn_s_waitcnt(0);
    asm volatile("s_waitcnt vmcnt(28)"); 
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_subtile_0 = kittens::subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][0], {warp_m, 0});
    load_lds_reg_row_fp6(a[0], a_subtile_0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(24)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_subtile_0 = kittens::subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][0], {warp_n, 0});
    load_lds_reg_row_fp6(b[0], b_subtile_0);


    #pragma unroll
    for (int k = 0; k < k_iters - 2; ++k, tic ^= 1, toc ^= 1) {

        // cluster 0
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // cluster 1 (load, interleave, wait)
        // load
        auto bs_subtile_1 = kittens::subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][1], {warp_n, 0});
        // load_lds_reg_row_fp6(b[1], bs_subtile_1);
        // load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row*WARPS_ROW, k+2}, As[tic][0], swizzled_offsets_A);
        // // mma
        // __builtin_amdgcn_sched_barrier(0);
        // mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        // __builtin_amdgcn_sched_barrier(0);
        do_interleaved_cluster(
            As[tic][0], g.a, {0, 0, block_row*WARPS_ROW, k+2},
            b[1], bs_subtile_1, 
            a[0], b[0], c[0][0]
        );
        // wait
        // __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        // __builtin_amdgcn_sched_barrier(0);


        // cluster 2 (load, interleave, wait)
        auto a_subtile_1 = kittens::subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_m, 0});
        // load_lds_reg_row_fp6(a[1], a_subtile_1);
        // load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col*WARPS_COL, k+2}, Bs[tic][0], swizzled_offsets_B);
        // // mma
        // __builtin_amdgcn_sched_barrier(0);
        // mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        // __builtin_amdgcn_sched_barrier(0);
        do_interleaved_cluster(
            Bs[tic][0], g.b, {0, 0, block_col*WARPS_COL, k+2},
            a[1], a_subtile_1, 
            a[0], b[1], c[0][1]
        );
        // load 
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);


        // cluster 3 (load, interleave)
        auto a_subtile_0 = kittens::subtile_inplace<REG_BLOCK_M, K_STEP>(As[toc][0], {warp_m, 0});
        // load_lds_reg_row_fp6(a[0], a_subtile_0);
        // load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_A, _gl_A, coord<ST_A>, NUM_THREADS>(g.a, {0, 0, block_row*WARPS_ROW+1, k+2}, As[tic][1], swizzled_offsets_A);
        // // mma
        // __builtin_amdgcn_sched_barrier(0);
        // mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        // __builtin_amdgcn_sched_barrier(0);
        do_interleaved_cluster(
            As[tic][1], g.a, {0, 0, block_row*WARPS_ROW+1, k+2},
            a[0], a_subtile_0, 
            a[1], b[0], c[1][0]
        );


        // cluster 4 (load, interleave, wait)
        auto b_subtile_0 = kittens::subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[toc][0], {warp_n, 0});
        // load_lds_reg_row_fp6(b[0], b_subtile_0);
        // load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, ST_B, _gl_B, coord<ST_B>, NUM_THREADS>(g.b, {0, 0, block_col*WARPS_COL+1, k+2}, Bs[tic][1], swizzled_offsets_B);
        // // mma 
        // __builtin_amdgcn_sched_barrier(0);
        // mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        // __builtin_amdgcn_sched_barrier(0);
        do_interleaved_cluster(
            Bs[tic][1], g.b, {0, 0, block_col*WARPS_COL+1, k+2},
            b[0], b_subtile_0, 
            a[1], b[1], c[1][1]
        );

    }

    // Epilogue (k = k_iters - 2)
    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)"); 
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][1], {warp_n, 0});
        load_lds_reg_row_fp6(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_m, 0});
        load_lds_reg_row_fp6(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)"); 
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0 = kittens::subtile_inplace<REG_BLOCK_M, K_STEP>(As[toc][0], {warp_m, 0});
        load_lds_reg_row_fp6(a[0], a_subtile_0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);
        
        auto b_subtile_0 = kittens::subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[toc][0], {warp_n, 0});
        load_lds_reg_row_fp6(b[0], b_subtile_0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1;
        toc ^= 1;
    }

    // Epilogue (k = k_iters - 1)
    {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<REG_BLOCK_N, K_STEP>(Bs[tic][1], {warp_n, 0});
        load_lds_reg_row_fp6(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);
        
        
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<REG_BLOCK_M, K_STEP>(As[tic][1], {warp_m, 0});
        load_lds_reg_row_fp6(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);


    // Stores
    store(g.c, c[0][0], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(g.c, c[0][1], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
    store(g.c, c[1][0], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(g.c, c[1][1], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
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
    std::uniform_real_distribution<> dis(-0.0f, 4.0f);

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
    const int WARMUP_REPS = 10;
    for (int r = 0; r < WARMUP_REPS; ++r) { 
        micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory(), stream>>>(globals);
    }
    hipDeviceSynchronize();

    // Timed kernel-only loop
    const int REPS = 50;
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