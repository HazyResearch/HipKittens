#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// problem dims
constexpr int b = 1;
constexpr int h = 1;
constexpr int d = 32;          
constexpr int n = 32;      
constexpr int BLOCK_SIZE = n;

// total rows to reduce over 
constexpr int n_total = 128;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;   
    gl<bf16, -1, -1, -1, -1> out;  
    dim3 grid()  { return dim3(n_total / BLOCK_SIZE); }  
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 2048; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    const int tile_idx = blockIdx.x;       
    const int lane     = kittens::laneid();    

    // load one 32x32 tile
    rt_bf<n, d, row_l> tile;
    load(tile, g.in, {0, 0, tile_idx, 0});     

    coord<> out_coord = {0, 0, 0, 0};
    bf16* out_ptr = (bf16*)&g.out[out_coord];
    const uint32_t out_elems = n * d;                   
    const uint32_t buffer_size = out_elems * sizeof(bf16);   
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(out_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);

    int laneid = kittens::laneid();
    int row_offset = laneid%32;
    int col_offset = 8*(laneid/32);
    using U = rt_bf<n, d, row_l>::T;

    #pragma unroll
    for (int z = 0; z < 2; ++z) {           // two 16-byte loads per thread

        #pragma unroll
        for (int i = 0; i < tile.height; ++i) {
            int row = tile.tile_size_row*i + row_offset;

            #pragma unroll
            for (int j = 0; j < tile.width; ++j) {
                int col = tile.tile_size_col*j + col_offset + z*16;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {       // 4 bf16_2 per z

                    const uint32_t byte_offset = static_cast<uint32_t>((row * d + col + 2*k) * sizeof(bf16));
                    const bf16_2 val = tile.tiles[i][j].data[k + 4*z];
                    const uint32_t v_data = *reinterpret_cast<const uint32_t*>(&val);

                    asm volatile(
                        "buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen\n"
                        "s_waitcnt vmcnt(0)\n"
                        :
                        : "v"(v_data), "v"(byte_offset), "s"(*(i32x4*)&br)
                        : "memory"
                    );
                }
            }
        }
    }

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::in, &micro_globals::out);
}