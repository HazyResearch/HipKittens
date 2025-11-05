#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "../utils.cpp"

using namespace kittens;

#define ATTN_B 1 // batch size
#define ATTN_H 1 // number of heads
#define ATTN_N 64 // sequence length
#define ATTN_D 32 // dimension
#define BLOCK_SIZE 64 // block size

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_QKVO = gl<bf16, -1, -1, -1, ATTN_D>;
struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Kg_out, Vg_out; 
    dim3 grid() { return dim3(1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 20000; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void attend_ker(const attn_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, ATTN_D> (&k_smem) = al.allocate<st_bf<BLOCK_SIZE, ATTN_D>>();
    st_bf<BLOCK_SIZE, ATTN_D> (&v_smem) = al.allocate<st_bf<BLOCK_SIZE, ATTN_D>>();


    // Initialize all of the register tiles.
    rt_bf<BLOCK_SIZE, ATTN_D> k_reg; // Q and K are both row layout.
    rt_bf<BLOCK_SIZE, ATTN_D, ducks::rt_layout::col> v_reg;

    int num_tiles = ATTN_N / BLOCK_SIZE;
    
    for (int j = 0; j < num_tiles; j++) {

        // Load K and V from global to shared memory.
        load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<BLOCK_SIZE, ATTN_D>>, NUM_THREADS>(
            g.Kg, {0, 0, j, 0}, k_smem);
        load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<BLOCK_SIZE, ATTN_D>>, NUM_THREADS>(
            g.Vg, {0, 0, j, 0}, v_smem);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __syncthreads();

        // Load K and V from shared memory to registers.
        load_lds_reg(k_reg, k_smem);
        load_lds_reg(v_reg, v_smem);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        
        // Store K and V to global memory.
        store(g.Kg_out, k_reg, {0, 0, j, 0});
        store(g.Vg_out, v_reg, {0, 0, j, 0});
    }
}


void dispatch_micro(attn_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &attn_globals::Qg, &attn_globals::Kg, &attn_globals::Vg, &attn_globals::Kg_out, &attn_globals::Vg_out);
}