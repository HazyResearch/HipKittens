#include "kittens.cuh"
#include <random>
using namespace kittens;


#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define SIZE 64

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
    rt_fl<SIZE, SIZE> tile_fl;
    rt_f6<SIZE, SIZE> tile_fp6;
    rt_bf<SIZE, SIZE> tile_bf16;
    load(tile_fp6, g.input, {0, 0, 0, 0});
    __syncthreads();
    store(g.output, tile_fp6, {0, 0, 0, 0});
    __syncthreads();
}

int main() {
    std::cout << "=== Simple FP6 Kernel Test ===\n";
    
    // Create FP6 data, not float data
    din *h_input = new din[SIZE * SIZE];  // ← FP6, not float
    dout *h_output = new dout[SIZE * SIZE];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0f, 3.3f);

    // Initialize with FP6 values (convert from float)
    h_input[0] = din(1.0f);     // Convert float → FP6
    h_input[1] = din(2.45f);
    h_input[2] = din(-3.7f);
    h_input[3] = din(10.0f);
    h_input[4] = din(-10.0f);
    for (int i = 5; i < SIZE * SIZE; i++) {
        // h_input[i] = din(1.0f); // / 100.0f);
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

        if (diff > 0.1) {
            large_diffs++;
        }

        if (large_diffs_printed < 40 && diff > 0.1) {
            std::cout << "[" << i << "] " << input_as_float << " -> " << output_as_float 
                    << " (diff: " << diff << ")\n";
            large_diffs_printed++;
        }
    }

    std::cout << "Number of large differences: " << large_diffs << " / " << SIZE * SIZE << std::endl;

    // Clean up (remove the h_input_float delete)
    hipFree(d_input);
    hipFree(d_output);
    delete[] h_input;
    delete[] h_output;
}


