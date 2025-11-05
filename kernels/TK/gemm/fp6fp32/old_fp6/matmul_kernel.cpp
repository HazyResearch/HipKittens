#include "kittens.cuh"
#include <random> 
// #include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define SIZE 128

using din = fp6_e2m3;
using dout = float;


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
    rt_f6<SIZE, SIZE> tile_fp6;
    load(tile_fp6, g.input, {0, 0, 0, 0});
    
    rt_fl<SIZE, SIZE, accum_l> tile_fl_accum;
    zero(tile_fl_accum);

    mma_ABt(tile_fl_accum, tile_fp6, tile_fp6, tile_fl_accum);
    store(g.output, tile_fl_accum, {0, 0, 0, 0});
}

int main() {
    std::cout << "=== Simple MFMA Test ===\n";
    
    din *h_input = new din[SIZE * SIZE];
    dout *h_output = new dout[SIZE * SIZE];

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0f, 3.3f);

    // Initialize with different values
    for (int i = 0; i < SIZE * SIZE; i++) {
        h_input[i] = din(dis(gen));
    }

    din *d_input;
    dout *d_output;
    hipMalloc(&d_input, SIZE * SIZE * sizeof(din));
    hipMalloc(&d_output, SIZE * SIZE * sizeof(dout));

    hipMemcpy(d_input, h_input, SIZE * SIZE * sizeof(din), hipMemcpyHostToDevice);

    _gl_tile_in input_gl(d_input, 1, 1, SIZE, SIZE);
    _gl_tile_out output_gl(d_output, 1, 1, SIZE, SIZE);
    micro_globals globals{input_gl, output_gl};

    micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory()>>>(globals);
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, SIZE * SIZE * sizeof(dout), hipMemcpyDeviceToHost);

    // CPU reference: compute A * A^T 
    float *cpu_result = new float[SIZE * SIZE];
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            cpu_result[i * SIZE + j] = 0.0f;
            for (int k = 0; k < SIZE; k++) {
                cpu_result[i * SIZE + j] += float(h_input[i * SIZE + k]) * float(h_input[j * SIZE + k]);
            }
        }
    }
    
    // Compare results
    int errors = 0;
    int num_printed = 0;
    for (int i = 0; i < SIZE * SIZE; i++) {
        if (std::abs(cpu_result[i] - h_output[i]) > 0.1*std::abs(cpu_result[i])) {
            errors++;
            if (num_printed < 10) {
                std::cout << "[" << i << "] CPU: " << cpu_result[i] << " GPU: " << h_output[i] 
                          << " (diff: " << std::abs(cpu_result[i] - h_output[i]) << ")\n";
                num_printed++;
            }
        }
    }
    
    std::cout << "Errors: " << errors << "/" << (SIZE * SIZE) << std::endl;
    if (errors < 10) {
        std::cout << "MFMA test PASSED" << std::endl;
    } else {
        std::cout << "MFMA test FAILED" << std::endl;
    }
    
    delete[] cpu_result;
    hipFree(d_input);
    hipFree(d_output);
    delete[] h_input;
    delete[] h_output;
}