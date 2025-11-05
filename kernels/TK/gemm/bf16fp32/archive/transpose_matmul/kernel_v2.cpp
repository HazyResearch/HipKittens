#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 128;  
constexpr int K_STEP     = 64;
constexpr int REG_BLOCK  = BLOCK_SIZE / 4; 

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 8192
#define K 8192
#define N 8192

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;

using G = kittens::group<4>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    dim3 grid()  { return dim3(N / BLOCK_SIZE, M / BLOCK_SIZE); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 32768; }
};

__device__ inline float2 issue_ds_read_b64(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        : "=v"(result)
        : "v"(lds_off)
        : "memory"
    );
    return result;
}

template<ducks::rt::all RT, ducks::st::all ST>
__forceinline__ __device__ inline void prefetch_reg_tile(RT &dst, const ST &src, int warp_row, int warp_col) {
    constexpr bool row_major = std::is_same_v<typename RT::layout, ducks::rt_layout::row>;

    int laneid = kittens::laneid();
    int row_offset = row_major ? laneid % 16 : 4 * (laneid / 16);
    int col_offset = row_major ? 4 * (laneid / 16) : laneid % 16;

    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        int row = i * dst.tile_size_row + row_offset;
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            int col = j * dst.tile_size_col + col_offset;
            uint32_t addr = reinterpret_cast<uintptr_t>(&src.data[0]);
            addr = src.idx(addr, {warp_row * REG_BLOCK + row, col});
            float2 val = issue_ds_read_b64(addr);
            using PackedT = typename base_types::packing<typename RT::dtype>::packed_type;
            auto packed = reinterpret_cast<PackedT*>(&val);
            auto &tile = dst.tiles[i][j];
            tile.data[0] = base_types::convertor<typename RT::dtype, PackedT>::convert(packed[0]);
            tile.data[1] = base_types::convertor<typename RT::dtype, PackedT>::convert(packed[1]);
        }
    }
}

__device__ inline void wait_for_ds() {
    asm volatile("s_waitcnt lgkmcnt(0)\n");
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    st_bf<BLOCK_SIZE, K_STEP> (&Bs) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();

    rt_bf<REG_BLOCK, K_STEP> a_reg_0, a_reg_1, b_reg_0, b_reg_1;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[4];
    for (int i = 0; i < 4; i++) { zero(C_accum[i]); }

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;
    const int warp_row_next = warp_row + 2;
    const int warp_col_next = warp_col + 2;

    // prefetch for tile 0
    G::load(As, g.a, {0, 0, row, 0});
    G::load(Bs, g.b, {0, 0, col, 0});
    __syncthreads();
    
    prefetch_reg_tile(a_reg_0, As, warp_row, 0);
    prefetch_reg_tile(a_reg_1, As, warp_row_next, 0);
    prefetch_reg_tile(b_reg_0, Bs, warp_col, 0);
    prefetch_reg_tile(b_reg_1, Bs, warp_col_next, 0);
    wait_for_ds();

    const int num_tiles = K / K_STEP;
    for (int tile = 1; tile < num_tiles; ++tile) {
        G::load(As, g.a, {0, 0, row, tile});
        G::load(Bs, g.b, {0, 0, col, tile});
        __syncthreads();

        // Compute on current
        mma_ABt(C_accum[0], a_reg_0, b_reg_0, C_accum[0]);
        mma_ABt(C_accum[1], a_reg_0, b_reg_1, C_accum[1]);
        mma_ABt(C_accum[2], a_reg_1, b_reg_0, C_accum[2]);
        mma_ABt(C_accum[3], a_reg_1, b_reg_1, C_accum[3]);
        // Issue loads into _next
        prefetch_reg_tile(a_reg_0, As, warp_row, 0);
        prefetch_reg_tile(a_reg_1, As, warp_row_next, 0);
        prefetch_reg_tile(b_reg_0, Bs, warp_col, 0);
        prefetch_reg_tile(b_reg_1, Bs, warp_col_next, 0);
        // Wait for the next loads to finish
        wait_for_ds();
    }

    mma_ABt(C_accum[0], a_reg_0, b_reg_0, C_accum[0]);
    mma_ABt(C_accum[1], a_reg_0, b_reg_1, C_accum[1]);
    mma_ABt(C_accum[2], a_reg_1, b_reg_0, C_accum[2]);
    mma_ABt(C_accum[3], a_reg_1, b_reg_1, C_accum[3]);

    for (int i = 0; i < 4; i++) {
        int global_row = row * 4 + (i / 2) * 2 + warp_row;  
        int global_col = col * 4 + (i % 2) * 2 + warp_col;
        store(g.c, C_accum[i], {0, 0, global_row, global_col});
    }
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)micro_tk,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c); 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}