#pragma once

#include <cstdint>

// Just use the raw constants directly
constexpr int NONE = 0;
constexpr int ALL_ALU = 1 << 0;
constexpr int VALU = 1 << 1;
constexpr int SALU = 1 << 2;
constexpr int MFMA = 1 << 3;
constexpr int ALL_VMEM = 1 << 4;
constexpr int VMEM_READ = 1 << 5;
constexpr int VMEM_WRITE = 1 << 6;
constexpr int ALL_DS = 1 << 7;
constexpr int DS_READ = 1 << 8;
constexpr int DS_WRITE = 1 << 9;
constexpr int TRANS = 1 << 10;

#define schedule_group_barrier(mask, size, sync_id) \
    __builtin_amdgcn_sched_group_barrier(mask, size, sync_id)



// for 32x32x16 mfma 
template<int REG_BLOCK_M, int REG_BLOCK_N, int DOT_SLICE>
constexpr int cluster_mfma_count() {
    constexpr int D_HEIGHT = REG_BLOCK_M / 32;  // 32x32 tiles in M dimension
    constexpr int D_WIDTH = REG_BLOCK_N / 32;   // 32x32 tiles in N dimension  
    constexpr int A_WIDTH = DOT_SLICE / 16;     // 16 elements per MFMA in K dimension
    
    return D_HEIGHT * D_WIDTH * A_WIDTH;
}

// for 16 bytes per thread
template<int ST_ROWS, int ST_COLS, int DTYPE_SIZE, int N_THREADS>
constexpr int compute_buffer_loads() {
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * N_THREADS;
    constexpr int memcpy_per_tile = (ST_ROWS * ST_COLS * DTYPE_SIZE) / bytes_per_memcpy;
    return memcpy_per_tile;
}

// for ds_read b128
template<int RT_HEIGHT, int RT_WIDTH>
constexpr int compute_ds_reads() {
    // From load_lds_reg: nested loops with ds_read_b128 operations
    // for k in range(2):
    //   for j in range(dst.width):      // RT_WIDTH
    //     for i in range(dst.height):   // RT_HEIGHT
    //       ds_read_b128 (only in row-major layout)
    
    // Assuming row-major layout which uses ds_read_b128
    constexpr int k_iterations = 2;
    return k_iterations * RT_WIDTH * RT_HEIGHT;
}




