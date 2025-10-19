#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int ATTN_D = 128; // dimension
constexpr int SLICE_QO = 32;
constexpr int DOT_SLICE_QO = 16;

template<int D, typename T=bf16, typename L=row_l, typename S=rt_16x16_s> using qo_tile = rt<T, DOT_SLICE_QO, D, L, S>;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

template<int D> struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;
    gl<bf16, -1, -1, -1, -1> out;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

// #define COL

template<int D> __global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals<D> g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<SLICE_QO, D, st_16x32_s> (&Q_i_smem) = al.allocate<st_bf<SLICE_QO, D, st_16x32_s>>();

    using Q_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<368, 383>>, 4>; // 16 registers
    ducks::rt::clobber<Q_ranges>();

    // Register tiles
    rt<bf16, DOT_SLICE_QO, D, row_l, rt_16x32_s, Q_ranges> Q_i; // 16 registers
    rt<bf16, DOT_SLICE_QO, D, col_l, rt_16x32_s, Q_ranges> Q_i_col; // 16 registers

    const int warpid = kittens::warpid();

    G::load<1, false>(Q_i_smem, g.in, {0, 0, 0, 0});

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    #ifdef COL
    load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem, {0, 0}));
    #else
    load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem, {0, 0}));
    #endif

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    #ifdef COL
    store<1>(g.out, Q_i_col, {0, 0, 0, 0}, {0, 0, 0, 0});
    #else
    store<1>(g.out, Q_i, {0, 0, 0, 0}, {0, 0, 0, 0});
    #endif

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    #ifdef COL
    load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem, {1, 0}));
    #else
    load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem, {1, 0}));
    #endif

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    #ifdef COL
    store<1>(g.out, Q_i_col, {0, 1, 0, 0}, {0, 0, 0, 0});
    #else
    store<1>(g.out, Q_i, {0, 1, 0, 0}, {0, 0, 0, 0});
    #endif
}

template<int D>
void dispatch_micro(micro_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", &micro_globals<ATTN_D>::in, &micro_globals<ATTN_D>::out);
}

