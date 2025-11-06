#include "kittens.cuh"
#include "utils.cpp"
#include <random>
using namespace kittens;


#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define SIZE 128

using din = fp6_e2m3;
using dout = fp6_e2m3;


using _gl_tile_in = gl<din, -1, -1, -1, -1>;
using _gl_tile_out = gl<dout, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_tile_in input;
    _gl_tile_out output;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_f6<SIZE, SIZE> (&tile_fp6) = al.allocate<st_f6<SIZE, SIZE>>();
    rt_f6<SIZE, SIZE> tile_fp6_rt;

    using T = typename st_f6<SIZE, SIZE>::dtype;  // fp6_e2m3
    using U = typename _gl_tile_in::dtype;        // fp6_e2m3  
    using U2 = base_types::packing<U>::packed_type; // fp6_e2m3_4
    
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = (SIZE * SIZE * sizeof(U) + bytes_per_memcpy - 1) / bytes_per_memcpy;
    
    uint32_t swizzled_offsets[memcpy_per_tile];
    
    // Use axis=2 (like your working global-to-register code)
    prefill_swizzled_offsets<2, false, rt_f6<SIZE, SIZE>, st_f6<SIZE, SIZE>, _gl_tile_in, coord<st_f6<SIZE, SIZE>>, NUM_THREADS>(
        g.input, {0, 0, 0, 0}, tile_fp6, swizzled_offsets);
        
    load_global_to_shared_direct_with_swizzled_offsets<2, false, st_f6<SIZE, SIZE>, _gl_tile_in, coord<st_f6<SIZE, SIZE>>, NUM_THREADS>(
        g.input, {0, 0, 0, 0}, tile_fp6, swizzled_offsets);
        
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    load_lds_reg_row(tile_fp6_rt, tile_fp6);

    store(g.output, tile_fp6_rt, {0, 0, 0, 0});
}

int main() {
    std::cout << "=== Simple FP6 Kernel Test ===\n";
    
    // Create FP6 data, not float data
    din *h_input = new din[SIZE * SIZE];  // ← FP6, not float
    dout *h_output = new dout[SIZE * SIZE];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0f, 3.3f);

    // Initialize with FP6 values (convert from float)
    h_input[0] = din(1.0f);     // Convert float → FP6
    h_input[1] = din(2.45f);
    h_input[2] = din(-3.7f);
    h_input[3] = din(10.0f);
    h_input[4] = din(-10.0f);
    for (int i = 5; i < SIZE * SIZE; i++) {
        h_input[i] = din(dis(gen));
    }

    // Allocate device memory for FP6 input
    din *d_input;
    dout *d_output;
    hipMalloc(&d_input, SIZE * SIZE * sizeof(din));
    hipMalloc(&d_output, SIZE * SIZE * sizeof(dout));

    hipMemcpy(d_input, h_input, SIZE * SIZE * sizeof(din), hipMemcpyHostToDevice);

    // Setup kernel globals with proper FP6 type
    _gl_tile_in input_gl(d_input, 1, 1, SIZE, SIZE);
    _gl_tile_out output_gl(d_output, 1, 1, SIZE, SIZE);
    micro_globals globals{input_gl, output_gl};


    // Launch kernel
    micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory()>>>(globals);
    hipDeviceSynchronize();

    // Copy back result
    hipMemcpy(h_output, d_output, SIZE * SIZE * sizeof(dout), hipMemcpyDeviceToHost);

    // Print results - just convert directly, no extra array
    std::cout << "Comparison:\n";
    for (int i = 0; i < 10; i++) {
        float input_as_float = float(h_input[i]);  // Convert FP6 to float
        float output_as_float = float(h_output[i]);
        std::cout << "In: " << input_as_float << " Out: " << output_as_float << std::endl;
    }

    // Check errors  
    int large_diffs = 0;
    int large_diffs_printed = 0;
    std::cout << "\nDetailed comparison:\n";
    for (int i = 0; i < SIZE * SIZE; i++) {
        float input_as_float = float(h_input[i]);  // Convert FP6 to float
        float output_as_float = float(h_output[i]);
        float diff = std::abs(output_as_float - input_as_float);

        if (diff == 0.0f) {
            large_diffs++;
        }

        if (output_as_float != 0.0f && diff > 0.0 && large_diffs_printed < 40) {
            std::cout << "[" << i << "] " << input_as_float << " -> " << output_as_float 
                    << " (diff: " << diff << ")\n";
                    large_diffs_printed++;
        }
    }

    std::cout << "Number of correct: " << large_diffs << " / " << SIZE * SIZE << std::endl;

    // Clean up (remove the h_input_float delete)
    hipFree(d_input);
    hipFree(d_output);
    delete[] h_input;
    delete[] h_output;
}


