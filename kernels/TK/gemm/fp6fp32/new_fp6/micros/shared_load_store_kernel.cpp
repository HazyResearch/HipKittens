#include "kittens.cuh"
#include "../dword_utils.cpp"
#include <random>
using namespace kittens;


#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 8192
#define K 8192

#define BLOCK_SIZE 256
#define K_STEP 128
#define REG_BLOCK_M 128
#define DOT_SLICE 64

using din = fp6_e2m3;
using dout = fp6_e2m3;

using _gl_tile_in = gl<din, -1, -1, -1, -1>;
using _gl_tile_out = gl<dout, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_tile_in input;
    _gl_tile_out output;
    dim3 grid()  { return dim3(M / BLOCK_SIZE); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_f6<BLOCK_SIZE, K_STEP> (&tile_fp6) = al.allocate<st_f6<BLOCK_SIZE, K_STEP>>();
    rt_f6<REG_BLOCK_M, DOT_SLICE> tile_fp6_rt;

    uintptr_t tile_base = reinterpret_cast<uintptr_t>(&tile_fp6);
    st_f6<BLOCK_SIZE, K_STEP> *tile_fp6_ptrs = reinterpret_cast<st_f6<BLOCK_SIZE, K_STEP>*>(tile_base + (reinterpret_cast<uintptr_t>(&tile_fp6) - tile_base) * 6 / 8);

    const int row = blockIdx.x;

    // Info
    const int warp_id = warpid();
    const int warp_row = warp_id / 4;
    const int num_tiles = K / K_STEP;
    const int num_slices = K_STEP / DOT_SLICE;

    constexpr int bytes_per_thread = 4;
    constexpr int memcpy_per_tile = (BLOCK_SIZE * K_STEP * 6 / 8) / (bytes_per_thread * NUM_THREADS);
    uint32_t swizzled_offsets[memcpy_per_tile];
    
    // Use axis=2 (like your working global-to-register code)
    prefill_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_tile_in, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(
        g.input, {0, 0, row, 0}, *tile_fp6_ptrs, swizzled_offsets);

    for (int i = 0; i < num_tiles; i++) {
        load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<BLOCK_SIZE, K_STEP>, _gl_tile_in, coord<st_f6<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(
            g.input, {0, 0, row, i}, *tile_fp6_ptrs, swizzled_offsets);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        for (int j = 0; j < num_slices; j++) {
            load_lds_reg_row_fp6_shuffled(tile_fp6_rt, subtile_inplace<REG_BLOCK_M, DOT_SLICE>(*tile_fp6_ptrs, {warp_row, j}));
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            store_fp6(g.output, tile_fp6_rt, {0, 0, row * 2 + warp_row, i * num_slices + j});
        }
    }
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
    din *h_input = new din[M * K];  // ← FP6, not float
    uint32_t *h_input_packed = new uint32_t[M * K * 6 / 32];
    uint32_t *h_output_packed = new uint32_t[M * K * 6 / 32];
    dout *h_output = new dout[M * K];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0f, 3.3f);

    for (int i = 0; i < M * K; i++) {
        h_input[i] = din(dis(gen));
        // h_input[i] = din(1.125f);
    }
    pack(h_input_packed, h_input, M * K);

    // Allocate device memory for FP6 input
    din *d_input_packed;
    dout *d_output_packed;
    hipMalloc(&d_input_packed, M * K * 6 / 8);
    hipMalloc(&d_output_packed, M * K * 6 / 8);

    hipMemcpy(d_input_packed, h_input_packed, M * K * 6 / 8, hipMemcpyHostToDevice);

    // Setup kernel globals with proper FP6 type
    _gl_tile_in input_gl(d_input_packed, 1, 1, M, K);
    _gl_tile_out output_gl(d_output_packed, 1, 1, M, K);
    micro_globals globals{input_gl, output_gl};


    // Launch kernel
    micro_tk<<<globals.grid(), globals.block(), globals.dynamic_shared_memory()>>>(globals);
    hipDeviceSynchronize();

    // Copy back result
    hipMemcpy(h_output_packed, d_output_packed, M * K * 6 / 8, hipMemcpyDeviceToHost);
    unpack(h_output, h_output_packed, M * K);

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
    for (int i = 0; i < M * K; i++) {
        float input_as_float = float(h_input[i]);  // Convert FP6 to float
        float output_as_float = float(h_output[i]);
        float diff = std::abs(output_as_float - input_as_float);

        if (diff == 0.0f) {
            large_diffs++;
        }

        if (output_as_float != 0.0f && diff > 0.0 && large_diffs_printed < 16) {
            std::cout << "[" << i << "] " << input_as_float << " -> " << output_as_float 
                    << " (diff: " << diff << ")\n";
                    large_diffs_printed++;
        }
    }

    std::cout << "Number of correct: " << large_diffs << " / " << M * K << std::endl;

    // // Print the entire output
    // for (int i = 0; i < M * K; i++) {
    //     std::cout << float(h_output[i]) << " ";
    //     if ((i + 1) % K == 0) {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    // Clean up (remove the h_input_float delete)
    hipFree(d_input_packed);
    hipFree(d_output_packed);
    delete[] h_input;
    delete[] h_output;
    delete[] h_input_packed;
    delete[] h_output_packed;
}

