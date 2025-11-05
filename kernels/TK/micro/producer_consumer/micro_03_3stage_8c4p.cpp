#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;


/*******************************************************************************/

__device__ static inline void mfma323216_pc(      float2 (&D)[8],
                                         const bf16_2 (&A)[8],
                                         const bf16_2 (&B)[8],
                                         const float2 (&C)[8]) {
    // Cast to the correct vector types that the intrinsic expects
    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    
    *(floatx16_t*)C = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
        *(bf16x8_t*)A,
        *(bf16x8_t*)B,
        *(floatx16_t*)C,
        0, 0, 0
    );

    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
        *(bf16x8_t*)(A + 4),
        *(bf16x8_t*)(B + 4),
        *(floatx16_t*)C,
        0, 0, 0
    );
}

__device__ static inline void mma_ABt_base_pc(rt_base<float, ducks::rt_layout::accumulator_col> &d,
    const rt_base<bf16, ducks::rt_layout::row> &a,
    const rt_base<bf16, ducks::rt_layout::row> &b, // in row-major mode
    const rt_base<float, ducks::rt_layout::accumulator_col> &c) {
    mfma323216_pc(d.data, a.data, b.data, c.data);
}

template<ducks::rt::accumulator_col_layout D, ducks::rt::row_like A, ducks::rt::row_like B, ducks::rt::accumulator_col_layout C>
__device__ static inline void mma_ABt_pc(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base_pc(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base_pc(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load_pc(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
    static_assert((std::is_same_v<typename RT::layout, ducks::rt_layout::row> && std::is_same_v<typename ST::layout, ducks::st_layout::row>) 
    || (std::is_same_v<typename RT::layout, ducks::rt_layout::col> && std::is_same_v<typename ST::layout, ducks::st_layout::col>)
    || (std::is_same_v<typename RT::layout, ducks::rt_layout::accumulator_col> && std::is_same_v<typename ST::layout, ducks::st_layout::accumulator_col>
    || (std::is_same_v<typename RT::layout, ducks::rt_layout::accumulator_row> && std::is_same_v<typename ST::layout, ducks::st_layout::accumulator_row>)), "register tile and shared tile layout must match");

    // TODO: add support for fp8
    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    static_assert(sizeof(U) == 2, "only supporting 16-bit dtypes");

    const int laneid = kittens::laneid() % kittens::WARP_THREADS;

    const int subtile_stride = kittens::TILE_ROW_DIM<U> * kittens::TILE_COL_DIM<U> * sizeof(U) / 2;
    const int tile_stride = subtile_stride * 2;
    const int row_stride = tile_stride * ST::underlying_width;

    const int subtile_id = (laneid % 32) / 16;
    const int lane_col_byte_offset = (laneid / 32) * 16;
    const int lane_row_offset = (laneid % 16);
    const int lane_byte_offset = lane_row_offset * kittens::TILE_COL_DIM<U> * sizeof(U) + lane_col_byte_offset;
    const int next_lane_byte_offset = lane_byte_offset + 32;
    const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 8) << 4);
    const int swizzled_next_lane_byte_offset = next_lane_byte_offset ^ ((next_lane_byte_offset >> 8) << 4);
    const uint32_t addr = reinterpret_cast<uintptr_t>(&src.data[0]) + subtile_id * subtile_stride + swizzled_lane_byte_offset;
    const uint32_t next_addr = reinterpret_cast<uintptr_t>(&src.data[0]) + subtile_id * subtile_stride + swizzled_next_lane_byte_offset;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
       #pragma unroll
       for(int j = 0; j < dst.width; j++) {
            asm volatile(
                "ds_read_b128 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][j].data[0]))
                : "v"(addr), "i"(i * row_stride + j * tile_stride)
                : "memory"
            );
            asm volatile(
                "ds_read_b128 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][j].data[4]))
                : "v"(next_addr), "i"(i * row_stride + j * tile_stride)
                : "memory"
            );
        }
    }
}


/*******************************************************************************/

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;
constexpr int N_BLOCK = 4;
constexpr int DOT_SLICE = 32;

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<BLOCK_SIZE, DOT_SLICE, row_l>;
using B_slice = rt_bf<BLOCK_SIZE, DOT_SLICE, row_l>;

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
  }

#define M 9216
#define K 9216
#define N 9216

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a, b;
    gl<bf16, -1, -1, -1, -1> c;
    dim3 grid()  { return dim3((N / NEW_COL_BLOCK_SIZE), ( M / NEW_ROW_BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 3 * (M_BLOCK + N_BLOCK) * BLOCK_SIZE * BLOCK_SIZE * sizeof(bf16); } 
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&As)[3][M_BLOCK] =
    al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 3, M_BLOCK>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&Bs)[3][N_BLOCK] =
    al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 3, N_BLOCK>();

    rt_fl<BLOCK_SIZE, BLOCK_SIZE, accum_col_l> C_accum;
    zero(C_accum);

    int row = blockIdx.y * M_BLOCK;
    int col = blockIdx.x * N_BLOCK;

    int warp_id = kittens::warpid();
    const int local_warp_id = warp_id % 4;
    int warp_group_id = kittens::warpgroupid();
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);
    A_slice A_tile;
    const lds_lane_ofs lane_ofs = prefill_swizzled_offsets(A_tile, As[0][0]);

    int s = 0, n1 = 1, n2 = 2;
    if (is_producer) {
        // preload tile 0 into stage s
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[s][m],  g.a, {0,0, row+m, 0}, swizzled_offsets_A);
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[s][n],  g.b, {0,0, col+n, 0}, swizzled_offsets_B);
        // preload tile 1 into stage n1
    }
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();    
    __builtin_amdgcn_sched_barrier(0);
    if (is_producer) {
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[n1][m], g.a, {0,0, row+m, 1}, swizzled_offsets_A);
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[n1][n], g.b, {0,0, col+n, 1}, swizzled_offsets_B);
    }

    const int num_tiles = K / BLOCK_SIZE;
    #pragma unroll
    for (int tile = 0; tile < num_tiles-2; ++tile) {
        if (is_producer) {
            #pragma unroll
            for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[n2][m], g.a, {0,0, row+m, tile+2}, swizzled_offsets_A);
            #pragma unroll
            for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[n2][n], g.b, {0,0, col+n, tile+2}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            A_slice a0;
            B_slice b0;

            load(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[s][consumer_idx], {0,0}), lane_ofs);
            load(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[s][local_warp_id], {0,0}), lane_ofs);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_pc(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            load(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[s][consumer_idx], {0,1}), lane_ofs);
            load(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[s][local_warp_id], {0,1}), lane_ofs);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt_pc(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();    
        int tmp = s;
        s = n1; n1 = n2; n2 = tmp;
    }

    if (is_consumer) {
        rt_bf<BLOCK_SIZE,BLOCK_SIZE,row_l> a_reg, b_reg;
        load(a_reg, As[s][consumer_idx], lane_ofs);
        load(b_reg, Bs[s][local_warp_id], lane_ofs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_pc(C_accum, a_reg, b_reg, C_accum);
        __builtin_amdgcn_s_setprio(0);

        load(a_reg, As[n1][consumer_idx], lane_ofs);
        load(b_reg, Bs[n1][local_warp_id], lane_ofs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt_pc(C_accum, a_reg, b_reg, C_accum);
        __builtin_amdgcn_s_setprio(0);
    }

    if (is_consumer) {
        store(g.c, C_accum, {0, 0, row + consumer_idx, col + local_warp_id});
    }
}

void dispatch_micro(micro_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms=0.f; hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start); hipEventDestroy(stop);
    // printf("kernel_ms=%.3f\n", ms);
    hipDeviceSynchronize();
  }

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}

