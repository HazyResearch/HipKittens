
#include "kittens.cuh"
#include <random>
#include <omp.h>
#include <cstring>
#include <iomanip>
using namespace kittens;


using din = fp6;
using dout = bf16;

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
#define N 8192
#define K 8192
constexpr int k_iters = K / K_STEP;

using _gl_A = gl<din, -1, -1, -1, -1>;
using _gl_B = gl<din, -1, -1, -1, -1>;
using _gl_C = gl<dout, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    dim3 grid()  { return dim3((N / BLOCK_SIZE_N), (M / BLOCK_SIZE_M)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

constexpr int axis = 2;


__global__ __attribute__((amdgpu_num_vgpr(128))) __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_fp6<HALF_BLOCK_SIZE_M, K_STEP, st_16x128_s>; // TO CHECK
    using ST_B = st_fp6<HALF_BLOCK_SIZE_N, K_STEP, st_16x128_s>;
    ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>(); // TO CHECK
    ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

    using A_0_range          = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<224, 255>>, 8>; // 32 registers - v[224:255]
    using A_1_range          = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<192, 223>>, 8>;
    using B_0_range          = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<160, 191>>, 8>; // 32 registers - v[192:223]
    using B_1_range          = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<128, 159>>, 8>;
    using C_accum_00_range    = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<256, 319>>, 4>;
    using C_accum_01_range    = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<320, 383>>, 4>; // 64 registers - a[0:63]
    using C_accum_10_range    = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<384, 447>>, 4>;
    using C_accum_11_range    = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<448, 511>>, 4>;
    using C_vgpr_range       = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<128, 191>>, 4>; // 64 registers - v[128:191]
    using C_vgpr_bf16_range  = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<192, 223>>, 2>; // 32 registers - v[192:223]

    using clobber_range_v = ducks::rt::type_list<ducks::rt::range<128, 255>>;
    using clobber_range_a = ducks::rt::type_list<ducks::rt::range<256, 511>>;

    // Clobber the registers
    ducks::rt::clobber<clobber_range_a>();
    ducks::rt::clobber<clobber_range_v>();

    rt<fp6, 64, 128, row_l, rt_16x128_s, A_0_range> A_0;
    rt<fp6, 64, 128, row_l, rt_16x128_s, A_1_range> A_1;
    rt<fp6, 64, 128, row_l, rt_16x128_s, B_0_range> B_0;
    rt<fp6, 64, 128, row_l, rt_16x128_s, B_1_range> B_1;
    rt<float, 64, 64, col_l, rt_16x16_s, C_accum_00_range> C_accum_00;
    rt<float, 64, 64, col_l, rt_16x16_s, C_accum_01_range> C_accum_01;
    rt<float, 64, 64, col_l, rt_16x16_s, C_accum_10_range> C_accum_10;
    rt<float, 64, 64, col_l, rt_16x16_s, C_accum_11_range> C_accum_11;
    rt<float, 64, 64, col_l, rt_16x16_s, C_vgpr_range> C_vgpr_transposed;
    rt<bf16, 64, 64, row_l, rt_16x16_s, C_vgpr_bf16_range> C_vgpr_bf16;

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    const int warp_row = warpid() / 2;
    const int warp_col = warpid() % 2;

    int tic = 0, toc = 1;

    // Global to shared addresses
    constexpr int bytes_per_thread = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int memcpy_per_tile = ST_A::rows * ST_A::cols * sizeof(fp6) / (bytes_per_thread * NUM_THREADS);
    uint32_t global_to_shared_swizzled_offsets[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[tic][0], g.a, global_to_shared_swizzled_offsets);

    // Shared to register addresses
    ds_read_b96_addresses addresses;

    auto do_interleaved_cluster_prologue_Bshuffle = [&](auto& dst_st, const auto& src_gl, auto idx, auto& dst, const auto& src, auto& a, auto& b, auto& c, auto& shuffle) {
        shuffle_in_place<0, 0>(shuffle);
        macros::v_nop();
        macros::v_nop();
        mma_ABt<0, 0, 0>(c, b, a);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 0);
        mma_ABt<0, 1, 0>(c, b, a);
        addresses = get_address_fp6(dst, src);
        mma_ABt<0, 2, 0>(c, b, a);
        shuffle_in_place<1, 0>(shuffle);
        mma_ABt<0, 3, 0>(c, b, a);
        load_fp6<0, 0>(dst, src, addresses);
        mma_ABt<1, 0, 0>(c, b, a);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 1);
        mma_ABt<1, 1, 0>(c, b, a);
        shuffle_in_place<2, 0>(shuffle);
        mma_ABt<1, 2, 0>(c, b, a);
        load_fp6<1, 0>(dst, src, addresses);
        mma_ABt<1, 3, 0>(c, b, a);
        mma_ABt<2, 0, 0>(c, b, a);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 2);
        mma_ABt<2, 1, 0>(c, b, a);
        load_fp6<2, 0>(dst, src, addresses);
        mma_ABt<2, 2, 0>(c, b, a);
        shuffle_in_place<3, 0>(shuffle);
        mma_ABt<2, 3, 0>(c, b, a);
        load_fp6<3, 0>(dst, src, addresses);
        mma_ABt<3, 0, 0>(c, b, a);
        mma_ABt<3, 1, 0>(c, b, a);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 3);
        mma_ABt<3, 2, 0>(c, b, a);
        mma_ABt<3, 3, 0>(c, b, a);
    };

    auto do_interleaved_cluster_prologue_Ashuffle = [&](auto& dst_st, const auto& src_gl, auto idx, auto& dst, const auto& src, auto& a, auto& b, auto& c, auto& shuffle) {
        shuffle_in_place<0, 0>(shuffle);
        macros::v_nop();
        macros::v_nop();
        mma_ABt<0, 0, 0>(c, b, a);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 0);
        mma_ABt<1, 0, 0>(c, b, a);
        addresses = get_address_fp6(dst, src);
        mma_ABt<2, 0, 0>(c, b, a);
        shuffle_in_place<1, 0>(shuffle);
        mma_ABt<3, 0, 0>(c, b, a);
        load_fp6<0, 0>(dst, src, addresses);
        mma_ABt<0, 1, 0>(c, b, a);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 1);
        mma_ABt<1, 1, 0>(c, b, a);
        shuffle_in_place<2, 0>(shuffle);
        mma_ABt<2, 1, 0>(c, b, a);
        load_fp6<1, 0>(dst, src, addresses);
        mma_ABt<3, 1, 0>(c, b, a);
        mma_ABt<0, 2, 0>(c, b, a);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 2);
        mma_ABt<1, 2, 0>(c, b, a);
        load_fp6<2, 0>(dst, src, addresses);
        mma_ABt<2, 2, 0>(c, b, a);
        shuffle_in_place<3, 0>(shuffle);
        mma_ABt<3, 2, 0>(c, b, a);
        load_fp6<3, 0>(dst, src, addresses);
        mma_ABt<0, 3, 0>(c, b, a);
        mma_ABt<1, 3, 0>(c, b, a);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 3);
        mma_ABt<2, 3, 0>(c, b, a);
        mma_ABt<3, 3, 0>(c, b, a);
    };

    auto do_interleaved_cluster_loop_Bshuffle = [&](auto& dst_st, const auto& src_gl, auto idx, auto& dst, const auto& src, auto& a, auto& b, auto& c, auto& shuffle) {
        shuffle_in_place<0, 0>(shuffle);
        macros::v_nop();
        macros::v_nop();
        mma_ABt<0, 0, 0>(c, b, a, c);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 0);
        mma_ABt<0, 1, 0>(c, b, a, c);
        addresses = get_address_fp6(dst, src);
        mma_ABt<0, 2, 0>(c, b, a, c);
        shuffle_in_place<1, 0>(shuffle);
        mma_ABt<0, 3, 0>(c, b, a, c);
        load_fp6<0, 0>(dst, src, addresses);
        mma_ABt<1, 0, 0>(c, b, a, c);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 1);
        mma_ABt<1, 1, 0>(c, b, a, c);
        shuffle_in_place<2, 0>(shuffle);
        mma_ABt<1, 2, 0>(c, b, a, c);
        load_fp6<1, 0>(dst, src, addresses);
        mma_ABt<1, 3, 0>(c, b, a, c);
        mma_ABt<2, 0, 0>(c, b, a, c);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 2);
        mma_ABt<2, 1, 0>(c, b, a, c);
        load_fp6<2, 0>(dst, src, addresses);
        mma_ABt<2, 2, 0>(c, b, a, c);
        shuffle_in_place<3, 0>(shuffle);
        mma_ABt<2, 3, 0>(c, b, a, c);
        load_fp6<3, 0>(dst, src, addresses);
        mma_ABt<3, 0, 0>(c, b, a, c);
        mma_ABt<3, 1, 0>(c, b, a, c);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 3);
        mma_ABt<3, 2, 0>(c, b, a, c);
        mma_ABt<3, 3, 0>(c, b, a, c);
    };

    auto do_interleaved_cluster_loop_Ashuffle = [&](auto& dst_st, const auto& src_gl, auto idx, auto& dst, const auto& src, auto& a, auto& b, auto& c, auto& shuffle) {
        shuffle_in_place<0, 0>(shuffle);
        macros::v_nop();
        macros::v_nop();
        mma_ABt<0, 0, 0>(c, b, a, c);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 0);
        mma_ABt<1, 0, 0>(c, b, a, c);
        addresses = get_address_fp6(dst, src);
        mma_ABt<2, 0, 0>(c, b, a, c);
        shuffle_in_place<1, 0>(shuffle);
        mma_ABt<3, 0, 0>(c, b, a, c);
        load_fp6<0, 0>(dst, src, addresses);
        mma_ABt<0, 1, 0>(c, b, a, c);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 1);
        mma_ABt<1, 1, 0>(c, b, a, c);
        shuffle_in_place<2, 0>(shuffle);
        mma_ABt<2, 1, 0>(c, b, a, c);
        load_fp6<1, 0>(dst, src, addresses);
        mma_ABt<3, 1, 0>(c, b, a, c);
        mma_ABt<0, 2, 0>(c, b, a, c);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 2);
        mma_ABt<1, 2, 0>(c, b, a, c);
        load_fp6<2, 0>(dst, src, addresses);
        mma_ABt<2, 2, 0>(c, b, a, c);
        shuffle_in_place<3, 0>(shuffle);
        mma_ABt<3, 2, 0>(c, b, a, c);
        load_fp6<3, 0>(dst, src, addresses);
        mma_ABt<0, 3, 0>(c, b, a, c);
        mma_ABt<1, 3, 0>(c, b, a, c);
        G::load_once(dst_st, src_gl, idx, global_to_shared_swizzled_offsets, 3);
        mma_ABt<2, 3, 0>(c, b, a, c);
        mma_ABt<3, 3, 0>(c, b, a, c);
    };

    G::load_once(As[tic][0], g.a, {0, 0, row*2, 0}, global_to_shared_swizzled_offsets, 0);
    G::load_once(As[tic][0], g.a, {0, 0, row*2, 0}, global_to_shared_swizzled_offsets, 1);
    G::load_once(As[tic][0], g.a, {0, 0, row*2, 0}, global_to_shared_swizzled_offsets, 2);
    G::load_once(As[tic][0], g.a, {0, 0, row*2, 0}, global_to_shared_swizzled_offsets, 3);
    G::load_once(Bs[tic][0], g.b, {0, 0, col*2, 0}, global_to_shared_swizzled_offsets, 0);
    G::load_once(Bs[tic][0], g.b, {0, 0, col*2, 0}, global_to_shared_swizzled_offsets, 1);
    G::load_once(Bs[tic][0], g.b, {0, 0, col*2, 0}, global_to_shared_swizzled_offsets, 2);
    G::load_once(Bs[tic][0], g.b, {0, 0, col*2, 0}, global_to_shared_swizzled_offsets, 3);
    G::load_once(Bs[tic][1], g.b, {0, 0, col*2+1, 0}, global_to_shared_swizzled_offsets, 0);
    G::load_once(Bs[tic][1], g.b, {0, 0, col*2+1, 0}, global_to_shared_swizzled_offsets, 1);
    G::load_once(Bs[tic][1], g.b, {0, 0, col*2+1, 0}, global_to_shared_swizzled_offsets, 2);
    G::load_once(Bs[tic][1], g.b, {0, 0, col*2+1, 0}, global_to_shared_swizzled_offsets, 3);
    G::load_once(As[tic][1], g.a, {0, 0, row*2+1, 0}, global_to_shared_swizzled_offsets, 0);
    G::load_once(As[tic][1], g.a, {0, 0, row*2+1, 0}, global_to_shared_swizzled_offsets, 1);
    G::load_once(As[tic][1], g.a, {0, 0, row*2+1, 0}, global_to_shared_swizzled_offsets, 2);
    G::load_once(As[tic][1], g.a, {0, 0, row*2+1, 0}, global_to_shared_swizzled_offsets, 3);

    G::load_once(As[toc][0], g.a, {0, 0, row*2, 1}, global_to_shared_swizzled_offsets, 0);
    G::load_once(As[toc][0], g.a, {0, 0, row*2, 1}, global_to_shared_swizzled_offsets, 1);
    G::load_once(As[toc][0], g.a, {0, 0, row*2, 1}, global_to_shared_swizzled_offsets, 2);
    G::load_once(As[toc][0], g.a, {0, 0, row*2, 1}, global_to_shared_swizzled_offsets, 3);
    G::load_once(Bs[toc][0], g.b, {0, 0, col*2, 1}, global_to_shared_swizzled_offsets, 0);
    G::load_once(Bs[toc][0], g.b, {0, 0, col*2, 1}, global_to_shared_swizzled_offsets, 1);
    G::load_once(Bs[toc][0], g.b, {0, 0, col*2, 1}, global_to_shared_swizzled_offsets, 2);
    G::load_once(Bs[toc][0], g.b, {0, 0, col*2, 1}, global_to_shared_swizzled_offsets, 3);
    G::load_once(Bs[toc][1], g.b, {0, 0, col*2+1, 1}, global_to_shared_swizzled_offsets, 0);
    G::load_once(Bs[toc][1], g.b, {0, 0, col*2+1, 1}, global_to_shared_swizzled_offsets, 1);
    G::load_once(Bs[toc][1], g.b, {0, 0, col*2+1, 1}, global_to_shared_swizzled_offsets, 2);
    G::load_once(Bs[toc][1], g.b, {0, 0, col*2+1, 1}, global_to_shared_swizzled_offsets, 3);
    G::load_once(As[toc][1], g.a, {0, 0, row*2+1, 1}, global_to_shared_swizzled_offsets, 0);
    G::load_once(As[toc][1], g.a, {0, 0, row*2+1, 1}, global_to_shared_swizzled_offsets, 1);
    G::load_once(As[toc][1], g.a, {0, 0, row*2+1, 1}, global_to_shared_swizzled_offsets, 2);
    G::load_once(As[toc][1], g.a, {0, 0, row*2+1, 1}, global_to_shared_swizzled_offsets, 3);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(28)"); 
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_subtile_0 = kittens::subtile_inplace<64, 128>(As[tic][0], {warp_row, 0});
    addresses = get_address_fp6(A_0, a_subtile_0);
    load_fp6<0, 0>(A_0, a_subtile_0, addresses);
    load_fp6<1, 0>(A_0, a_subtile_0, addresses);
    load_fp6<2, 0>(A_0, a_subtile_0, addresses);
    load_fp6<3, 0>(A_0, a_subtile_0, addresses);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(24)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_subtile_0 = kittens::subtile_inplace<64, 128>(Bs[tic][0], {warp_col, 0});
    addresses = get_address_fp6(B_0, b_subtile_0);
    load_fp6<0, 0>(B_0, b_subtile_0, addresses);
    asm volatile("s_waitcnt lgkmcnt(0)");
    shuffle_in_place<0, 0>(A_0);
    load_fp6<1, 0>(B_0, b_subtile_0, addresses);
    shuffle_in_place<1, 0>(A_0);
    load_fp6<2, 0>(B_0, b_subtile_0, addresses);
    shuffle_in_place<2, 0>(A_0);
    load_fp6<3, 0>(B_0, b_subtile_0, addresses);
    shuffle_in_place<3, 0>(A_0);

    {
        constexpr int k = 0;
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto b_subtile_1 = kittens::subtile_inplace<64, 128>(Bs[tic][1], {warp_col, 0});
        do_interleaved_cluster_prologue_Bshuffle(As[tic][0], g.a, coord<ST_A>{0, 0, row*2, k+2}, B_1, b_subtile_1, A_0, B_0, C_accum_00, B_0);

        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_subtile_1 = kittens::subtile_inplace<64, 128>(As[tic][1], {warp_row, 0});
        do_interleaved_cluster_prologue_Bshuffle(Bs[tic][0], g.b, coord<ST_B>{0, 0, col*2, k+2}, A_1, a_subtile_1, A_0, B_1, C_accum_01, B_1);

        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_subtile_0 = kittens::subtile_inplace<64, 128>(As[toc][0], {warp_row, 0});
        do_interleaved_cluster_prologue_Ashuffle(Bs[tic][1], g.b, coord<ST_B>{0, 0, col*2+1, k+2}, A_0, a_subtile_0, A_1, B_0, C_accum_10, A_1);

        auto b_subtile_0 = kittens::subtile_inplace<64, 128>(Bs[toc][0], {warp_col, 0});
        do_interleaved_cluster_prologue_Ashuffle(As[tic][1], g.a, coord<ST_A>{0, 0, row*2+1, k+2}, B_0, b_subtile_0, A_1, B_1, C_accum_11, A_0);
    }
    tic ^= 1, toc ^= 1;
    #pragma unroll
    for (int k = 1; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto b_subtile_1 = kittens::subtile_inplace<64, 128>(Bs[tic][1], {warp_col, 0});
        // load(B_1, b_subtile_1);
        // G::load(As[tic][0], g.a, {0, 0, row*2, k+2});
        // shuffle_in_place(B_0);
        // mma_ABt(C_accum_00, B_0, A_0, C_accum_00);
        do_interleaved_cluster_loop_Bshuffle(As[tic][0], g.a, coord<ST_A>{0, 0, row*2, k+2}, B_1, b_subtile_1, A_0, B_0, C_accum_00, B_0);

        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_subtile_1 = kittens::subtile_inplace<64, 128>(As[tic][1], {warp_row, 0});
        // load(A_1, a_subtile_1);
        // G::load(Bs[tic][0], g.b, {0, 0, col*2, k+2});
        // shuffle_in_place(B_1);
        // mma_ABt(C_accum_01, B_1, A_0, C_accum_01);
        do_interleaved_cluster_loop_Bshuffle(Bs[tic][0], g.b, coord<ST_B>{0, 0, col*2, k+2}, A_1, a_subtile_1, A_0, B_1, C_accum_01, B_1);

        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_subtile_0 = kittens::subtile_inplace<64, 128>(As[toc][0], {warp_row, 0});
        // load(A_0, a_subtile_0);
        // G::load(Bs[tic][1], g.b, {0, 0, col*2+1, k+2});
        // shuffle_in_place(A_1);
        // mma_ABt(C_accum_10, B_0, A_1, C_accum_10);
        do_interleaved_cluster_loop_Ashuffle(Bs[tic][1], g.b, coord<ST_B>{0, 0, col*2+1, k+2}, A_0, a_subtile_0, A_1, B_0, C_accum_10, A_1);

        auto b_subtile_0 = kittens::subtile_inplace<64, 128>(Bs[toc][0], {warp_col, 0});
        // load(B_0, b_subtile_0);
        // G::load(As[tic][1], g.a, {0, 0, row*2+1, k+2});
        // asm volatile("s_waitcnt lgkmcnt(0)");
        // shuffle_in_place(A_0);
        // mma_ABt(C_accum_11, B_1, A_1, C_accum_11);
        do_interleaved_cluster_loop_Ashuffle(As[tic][1], g.a, coord<ST_A>{0, 0, row*2+1, k+2}, B_0, b_subtile_0, A_1, B_1, C_accum_11, A_0);
    }
    { // k = k_iters - 2
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto b_subtile_1 = kittens::subtile_inplace<64, 128>(Bs[tic][1], {warp_col, 0});
        load(B_1, b_subtile_1);
        shuffle_in_place(B_0);
        mma_ABt(C_accum_00, B_0, A_0, C_accum_00);

        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_subtile_1 = kittens::subtile_inplace<64, 128>(As[tic][1], {warp_row, 0});
        load(A_1, a_subtile_1);
        shuffle_in_place(B_1);
        mma_ABt(C_accum_01, B_1, A_0, C_accum_01);

        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_subtile_0 = kittens::subtile_inplace<64, 128>(As[toc][0], {warp_row, 0});
        load(A_0, a_subtile_0);
        shuffle_in_place(A_1);
        mma_ABt(C_accum_10, B_0, A_1, C_accum_10);

        auto b_subtile_0 = kittens::subtile_inplace<64, 128>(Bs[toc][0], {warp_col, 0});
        load(B_0, b_subtile_0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        shuffle_in_place(A_0);
        mma_ABt(C_accum_11, B_1, A_1, C_accum_11);
    }
    tic ^= 1, toc ^= 1;
    { // k = k_iters - 1
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto b_subtile_1 = kittens::subtile_inplace<64, 128>(Bs[tic][1], {warp_col, 0});
        load(B_1, b_subtile_1);
        shuffle_in_place(B_0);
        mma_ABt(C_accum_00, B_0, A_0, C_accum_00);

        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_subtile_1 = kittens::subtile_inplace<64, 128>(As[tic][1], {warp_row, 0});
        load(A_1, a_subtile_1);
        shuffle_in_place(B_1);
        mma_ABt(C_accum_01, B_1, A_0, C_accum_01);

        asm volatile("s_waitcnt lgkmcnt(0)");

        shuffle_in_place(A_1);
        mma_ABt(C_accum_10, B_0, A_1, C_accum_10);
        mma_ABt(C_accum_11, B_1, A_1, C_accum_11);
    }
    // macros::v_nop();
    // macros::v_nop();
    // macros::v_nop();
    // macros::v_nop();
    // macros::v_nop();
    // macros::v_nop();
    accvgpr_read(C_vgpr_transposed, C_accum_00);
    rt<float, 64, 64, row_l, rt_16x16_s, ducks::rt::transpose_2d<C_vgpr_range, 4, 4>> C_vgpr;
    copy(C_vgpr_bf16, C_vgpr);
    store(g.c, C_vgpr_bf16, {0, 0, row*4, col*4}, {0, 0, warp_row, warp_col});
    accvgpr_read(C_vgpr_transposed, C_accum_01);
    copy(C_vgpr_bf16, C_vgpr);
    store(g.c, C_vgpr_bf16, {0, 0, row*4, col*4+2}, {0, 0, warp_row, warp_col});
    accvgpr_read(C_vgpr_transposed, C_accum_10);
    copy(C_vgpr_bf16, C_vgpr);
    store(g.c, C_vgpr_bf16, {0, 0, row*4+2, col*4}, {0, 0, warp_row, warp_col});
    accvgpr_read(C_vgpr_transposed, C_accum_11);
    copy(C_vgpr_bf16, C_vgpr);
    store(g.c, C_vgpr_bf16, {0, 0, row*4+2, col*4+2}, {0, 0, warp_row, warp_col});
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
        // if (i / K == i % K) {
        //     h_input_a[i] = din(1.0f);
        //     h_input_b[i] = din(1.0f);
        // } else {
        //     h_input_a[i] = din(0.0f);
        //     h_input_b[i] = din(0.0f);
        // // }
        // h_input_a[i] = din(1.0f);
        // h_input_b[i] = din(1.0f);
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
    dout *cpu_result = new dout[M * N];
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += float(h_input_a[i * K + k]) * float(h_input_b[j * K + k]);
            }
            cpu_result[i * N + j] = dout(sum);
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
        float threshold = 0;//rtol * fabs(float(cpu_result[i])) + atol;
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