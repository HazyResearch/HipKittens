#include "fp4_load.cuh"

#ifdef TEST_WARP_MEMORY_TILE_FP4_LOAD

#include <algorithm>
#include <cmath>

// Exercises the FP4 load paths (global -> shared -> register) that PR 1 adds.
// The existing sharedreg_load_store round-trip can't be used for FP4 because the
// register -> shared store path isn't in #47's scope. This test runs a hand-rolled
// kernel that loads into a register tile and then dequantizes each thread's packed
// elements into a flat float buffer. Host validation checks that every dequantized
// FP4 value appears in the expected multiset.

using GL_fp4 = kittens::gl<kittens::fp4e2m1_2, 1, 1, 16, 128>;
// Output buffer: 32 threads x 32 fp4e2m1_4 per thread x 4 FP4 per packed element = 4096 floats.
using GL_fl  = kittens::gl<float, 1, 1, 1, 4096>;

__global__ static void fp4_load_kernel(GL_fp4 input, GL_fl output) {
    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::shared_allocator<16> al((int*)&__shm[0]);

    using ST = kittens::st<kittens::fp4e2m1_2, 16, 128, kittens::ducks::st_shape::st_16x128>;
    ST &shared_tile = al.allocate<ST>();

    kittens::load<2, false, ST, GL_fp4, kittens::coord<ST>>(shared_tile, input, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    using RT = kittens::rt<kittens::fp4e2m1_2, 16, 128,
                            kittens::ducks::rt_layout::row,
                            kittens::ducks::rt_shape::rt_16x128>;
    RT reg_tile;
    kittens::load(reg_tile, shared_tile);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    constexpr int floats_per_thread = RT::packed_per_thread * 4;
    const int tid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < RT::packed_per_thread; i++) {
        kittens::fp4e2m1_4 packed = reg_tile.tiles[0][0].data[i];
        float4 vals = float4(packed);
        const int base = tid * floats_per_thread + i * 4;
        output.raw_ptr[base + 0] = vals.x;
        output.raw_ptr[base + 1] = vals.y;
        output.raw_ptr[base + 2] = vals.z;
        output.raw_ptr[base + 3] = vals.w;
    }
}

void warp::memory::tile::fp4_load::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/fp4_load tests! -----\n" << std::endl;

    constexpr int tile_pairs = 16 * 128;       // device tile size in fp4e2m1_2 units
    constexpr int tile_fp4   = tile_pairs * 2; // logical FP4 value count = 4096

    std::vector<float> i_ref(tile_pairs);
    std::vector<float> o_ref(tile_fp4);

    kittens::fp4e2m1_2 *d_i;
    float *d_o;
    initialize<kittens::fp4e2m1_2, float>(&d_i, &d_o, i_ref, o_ref);

    GL_fp4 input_gl(d_i, nullptr, nullptr, nullptr, nullptr);
    GL_fl  output_gl(d_o, nullptr, nullptr, nullptr, nullptr);

    hipFuncSetAttribute(
        reinterpret_cast<void *>(fp4_load_kernel),
        hipFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY / 2
    );
    fp4_load_kernel<<<1, kittens::WARP_THREADS, kittens::MAX_SHARED_MEMORY / 2>>>(input_gl, output_gl);
    HipCheckError();

    // Expected: each fp4e2m1_2 pair packs (f, f), so both halves dequantize to the same value.
    // The kernel dumps 4096 floats total — each input pair contributes 2 identical values somewhere
    // in the output. Host builds the expected multiset as 2x every i_ref entry, sorts both sides,
    // and compares.
    std::vector<float> expected(tile_fp4);
    for (int idx = 0; idx < tile_pairs; idx++) {
        expected[2 * idx]     = i_ref[idx];
        expected[2 * idx + 1] = i_ref[idx];
    }

    float *o_host = new float[tile_fp4];
    hipDeviceSynchronize();
    HipCheckError();
    hipMemcpy(o_host, d_o, tile_fp4 * sizeof(float), hipMemcpyDeviceToHost);
    HipCheckError();

    std::vector<float> actual(o_host, o_host + tile_fp4);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());

    bool good = true;
    float max_diff = 0;
    int first_mismatch = -1;
    for (int i = 0; i < tile_fp4; i++) {
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > 0.5f) {
            good = false;
            if (first_mismatch < 0) first_mismatch = i;
            if (diff > max_diff) max_diff = diff;
        }
    }

    std::cout << "test `fp4_load=fp4e2m1_2` ";
    if (good) std::cout << " -- PASSED" << std::endl;
    else {
        std::cout << " ----- ALERT! FAILED (first mismatch at sorted idx " << first_mismatch
                  << ", max diff " << max_diff << ") -----" << std::endl;
    }

    hipFree(d_i);
    hipFree(d_o);
    delete[] o_host;
    HipCheckError();

    test_info info;
    info.label = "fp4_load=fp4e2m1_2";
    info.result = good ? test_result::PASSED : test_result::FAILED;
    results.push_back(info);
}

#endif
