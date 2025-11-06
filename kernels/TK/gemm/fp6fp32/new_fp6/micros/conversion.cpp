#include "kittens.cuh"
#include "utils.cpp"
#include <random>
#include <cstring>
#include <iomanip>
#include <set>
using namespace kittens;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)
#define SIZE 64

using din = float;
using dout = fp6_e2m3;
using _gl_tile_in = gl<din, -1, -1, -1, -1>;
using _gl_tile_out = gl<dout, -1, -1, -1, -1>;

struct micro_globals {
    _gl_tile_in input;
    _gl_tile_out output;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    rt_fl<SIZE, SIZE, accum_l> tile_fl;
    load(tile_fl, g.input, {0, 0, 0, 0});
    __syncthreads();
    store_fp6_convert(g.output, tile_fl, {0, 0, 0, 0});
    __syncthreads();
}

void unpack(dout *output, const uint32_t *input, int size) {

    // 1. Unpack from 32-bit words
    for (int i = 0; i < size; i++) {
        uint8_t tmp = 0;
        const int bit_pos = i * 6;
        const int word_idx = bit_pos >> 5;
        const int bit_off = bit_pos & 31;
        const int spill = bit_off + 6 - 32;
        tmp |= ((input[word_idx] >> bit_off) & 0x3Fu);
        if (spill > 0) {
            tmp |= ((input[word_idx + 1] << (6 - spill)) & 0x3Fu);
        }
        output[i] = std::bit_cast<dout>(tmp);
    }
}

int main() {
    std::cout << "=== Simple FP6 Kernel Test ===\n";
    
    // Create FP6 data, not float data
    din *h_input = new din[SIZE * SIZE];  // ← FP6, not float
    uint32_t *h_output_packed = new uint32_t[SIZE * SIZE * 6 / 32];
    dout *h_output = new dout[SIZE * SIZE];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0f, 3.3f);

    // Initialize with FP6 values (convert from float)
    for (int i = 0; i < SIZE * SIZE; i++) {
        h_input[i] = din(1.0f);
    }

    // Allocate device memory for FP6 input
    din *d_input;
    dout *d_output_packed;
    hipMalloc(&d_input, SIZE * SIZE * sizeof(float));
    hipMalloc(&d_output_packed, SIZE * SIZE * 6 / 8);
    hipMemcpy(d_input, h_input, SIZE * SIZE * sizeof(float), hipMemcpyHostToDevice);    

    // Setup kernel globals with proper FP6 type
    _gl_tile_in input_gl(d_input, 1, 1, SIZE, SIZE);
    _gl_tile_out output_gl(d_output_packed, 1, 1, SIZE, SIZE);
    micro_globals globals{input_gl, output_gl};

    // Launch kernel
    micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory()>>>(globals);
    hipDeviceSynchronize();

    // Copy back result
    hipMemcpy(h_output_packed, d_output_packed, SIZE * SIZE * 6 / 8, hipMemcpyDeviceToHost);
    unpack(h_output, h_output_packed, SIZE * SIZE);

    // Check errors  
    int large_diffs = 0;
    int large_diffs_printed = 0;
    int num_correct_printed = 0;
    std::cout << "\nDetailed comparison:\n";
    for (int i = 0; i < SIZE * SIZE; i++) {
        float input_as_float = float(h_input[i]);  // Convert FP6 to float
        float output_as_float = float(h_output[i]);
        float diff = std::abs(output_as_float - input_as_float);

        if (diff == 0.0f) {
            large_diffs++;
        }

        if (output_as_float != 0.0f && diff > 0.0 && large_diffs_printed < 5) {
            std::cout << "[" << i << "] " << input_as_float << " -> " << output_as_float 
                    << " (diff: " << diff << ")\n";
                    large_diffs_printed++;
        }

        if (num_correct_printed < 5) {
            if (diff == 0.0f) {
                std::cout << "[" << i << "] " << input_as_float << " -> " << output_as_float 
                        << " (diff: " << diff << ")\n";
                num_correct_printed++;
            }
        }
    }

    std::cout << "Number of correct: " << large_diffs << " / " << SIZE * SIZE << std::endl;

    // Clean up (remove the h_input_float delete)
    hipFree(d_input);
    hipFree(d_output_packed);
    delete[] h_input;
    delete[] h_output;
    delete[] h_output_packed;
}


