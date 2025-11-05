#include "kittens.cuh"
#include "utils.cpp"
#include <random>
#include <cstring>
#include <iomanip>
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
    std::cout << "=== FP6 Packed MFMA Test ===\n";
    
    // Host arrays for FP6 values
    din *h_input = new din[SIZE * SIZE];
    dout *h_output = new dout[SIZE * SIZE];
    
    // Calculate sizes for packed data
    int total_fp6_values = SIZE * SIZE;
    int total_bits = total_fp6_values * 6;
    int total_bytes = (total_bits + 7) / 8;
    int total_words = (total_bits + 31) / 32;
    
    std::cout << "Data dimensions:\n";
    std::cout << "  FP6 values: " << total_fp6_values << "\n";
    std::cout << "  Total bits: " << total_bits << "\n";
    std::cout << "  Total bytes: " << total_bytes << "\n";
    std::cout << "  Total 32-bit words: " << total_words << "\n\n";
    
    // Packed data arrays
    uint32_t *h_input_packed = new uint32_t[total_words];
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0f, 1.0f);
    
    // Initialize with random FP6 values
    for (int i = 0; i < SIZE * SIZE; i++) {
        h_input[i] = din(dis(gen));
    }
    
    // Pack the input data
    pack(h_input_packed, h_input, SIZE * SIZE);
    
    // Print first few packed words for debugging
    std::cout << "First 8 packed words (hex):\n";
    for (int i = 0; i < 8 && i < total_words; i++) {
        std::cout << "  Word " << i << ": 0x" << std::hex << std::setw(8) 
                  << std::setfill('0') << h_input_packed[i] << std::dec << "\n";
    }
    std::cout << "\n";
    
    // Allocate device memory - use actual packed size
    din *d_input_packed;
    dout *d_output;
    hipMalloc(&d_input_packed, total_bytes);
    hipMalloc(&d_output, SIZE * SIZE * sizeof(dout));
    
    // Copy packed input to device
    hipMemcpy(d_input_packed, h_input_packed, total_bytes, hipMemcpyHostToDevice);
    
    // Setup kernel globals
    _gl_tile_in input_gl(d_input_packed, 1, 1, SIZE, SIZE);
    _gl_tile_out output_gl(d_output, 1, 1, SIZE, SIZE);
    micro_globals globals{input_gl, output_gl};
    
    // Launch kernel
    std::cout << "Launching kernel...\n";
    micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory()>>>(globals);
    hipDeviceSynchronize();
    
    // Check for kernel errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy result back
    hipMemcpy(h_output, d_output, SIZE * SIZE * sizeof(dout), hipMemcpyDeviceToHost);
    
    // CPU reference: compute A * A^T
    std::cout << "Computing CPU reference...\n";
    float *cpu_result = new float[SIZE * SIZE];
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            cpu_result[i * SIZE + j] = 0.0f;
            for (int k = 0; k < SIZE; k++) {
                // A * A^T means row i of A dot product with row j of A
                cpu_result[i * SIZE + j] += float(h_input[i * SIZE + k]) * float(h_input[j * SIZE + k]);
            }
        }
    }
    
    // Compare results
    std::cout << "\nComparing results:\n";
    int errors = 0;
    int num_printed = 0;
    float max_rel_error = 0.0f;
    float avg_rel_error = 0.0f;
    int count = 0;
    
    for (int i = 0; i < SIZE * SIZE; i++) {
        float expected = cpu_result[i];
        float actual = h_output[i];
        float abs_error = std::abs(expected - actual);
        float rel_error = (expected != 0) ? abs_error / std::abs(expected) : abs_error;
        
        avg_rel_error += rel_error;
        count++;
        max_rel_error = std::max(max_rel_error, rel_error);
        
        // Use relative error threshold for large values, absolute for small
        float threshold = std::max(0.1f * std::abs(expected), 0.01f);
        
        if (abs_error > threshold) {
            errors++;
            if (num_printed < 10) {
                int row = i / SIZE;
                int col = i % SIZE;
                std::cout << "  [" << row << "," << col << "] CPU: " << expected 
                          << " GPU: " << actual 
                          << " (abs_err: " << abs_error 
                          << ", rel_err: " << rel_error << ")\n";
                num_printed++;
            }
        }
    }
    
    avg_rel_error /= count;
    
    std::cout << "\nError Statistics:\n";
    std::cout << "  Errors: " << errors << "/" << (SIZE * SIZE) << "\n";
    std::cout << "  Max relative error: " << max_rel_error << "\n";
    std::cout << "  Avg relative error: " << avg_rel_error << "\n";
    
    
    if (errors < SIZE * SIZE * 0.01) {  
        std::cout << "\nMFMA test PASSED\n";
    } else {
        std::cout << "\nMFMA test FAILED\n";
    }
    
    // Cleanup
    delete[] cpu_result;
    delete[] h_input;
    delete[] h_input_packed;
    delete[] h_output;
    hipFree(d_input_packed);
    hipFree(d_output);
    
    return 0;
}