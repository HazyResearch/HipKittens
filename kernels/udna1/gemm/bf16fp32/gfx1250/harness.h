/**
 * @file harness.h
 * @brief Host entry point for the gfx1250 GEMM ladder.
 *
 * Every rung defines `dispatch(gemm_globals, launch_config)` and includes this at the end of the file. Under
 * `-DHARNESS_PYEXT` that becomes a pybind11 module exporting `dispatch` for correctness checks and
 * `bench` for timing (see `pyext.h`). It is the only entry point a rung has, so every rung is
 * reached from Python: `test.py` is the correctness gate and `gemm_ladder.py` is the benchmark.
 *
 * Nothing here checks correctness. The reference is `torch.matmul`, on the caller's side of the
 * boundary.
 */

#pragma once

#ifdef HARNESS_PYEXT

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static inline void hip_check(hipError_t e, const char* what, int line) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error at line %d (%s): %s\n",
                     line, what, hipGetErrorString(e));
        std::exit(1);
    }
}
#define HIP_OK(call) hip_check((call), #call, __LINE__)

using dev_elem = gfx1250_gemm::elem_t;

#ifdef GEMM_C_COL_MAJOR
#error "C is always column-major on this ladder; GEMM_C_COL_MAJOR is not a knob."
#endif

#ifndef HK_WARMUP_ITERS
#define HK_WARMUP_ITERS 500
#endif
#ifndef HK_MEASURED_ITERS
#define HK_MEASURED_ITERS 100
#endif

#ifndef HK_FLUSH_BETWEEN_ITERS
#define HK_FLUSH_BETWEEN_ITERS 1
#endif

/* One `global_store_dwordx4` per lane per stride, as ordinary cached stores: a non-temporal
 * variant would stream past the cache and evict nothing. */
__global__ void hk_cache_flush_kernel(uint4* __restrict__ buf, size_t n, uint32_t pattern)
{
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    const uint4 v = make_uint4(pattern, pattern, pattern, pattern);
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        buf[i] = v;
}

struct cache_flusher {
    static constexpr int THREADS = 256;
    uint4* buf = nullptr;
    size_t n = 0;                                   // uint4 elements, 0 when the flush is off
    int blocks = 0;
    double l2_mb = 0.0;

    void init(const hipDeviceProp_t& props)
    {
        l2_mb = props.l2CacheSize / (1024.0 * 1024.0);
        if (!HK_FLUSH_BETWEEN_ITERS) return;
        const size_t l2 = props.l2CacheSize > 0 ? size_t(props.l2CacheSize) : 0;
        size_t want = std::max<size_t>(4 * l2, size_t(512) << 20);

        size_t free_b = 0, total_b = 0;
        HIP_OK(hipMemGetInfo(&free_b, &total_b));
        want = std::min(want, free_b / 4);          // the operands are already allocated
        want &= ~(sizeof(uint4) - 1);
        if (want < 2 * l2 || hipMalloc(&buf, want) != hipSuccess) { buf = nullptr; return; }

        n = want / sizeof(uint4);
        // Eight blocks per CU keeps every memory channel busy without a one-workgroup tail.
        blocks = int(std::min<size_t>(size_t(props.multiProcessorCount) * 8,
                                     (n + THREADS - 1) / THREADS));
    }

    void run(hipStream_t stream) const
    {
        if (!buf) return;
        hk_cache_flush_kernel<<<blocks, THREADS, 0, stream>>>(buf, n, 0xA5A5A5A5u);
    }

    void destroy() { if (buf) HIP_OK(hipFree(buf)); buf = nullptr; n = 0; }
    double mb() const { return double(n) * sizeof(uint4) / (1024.0 * 1024.0); }
};

struct hk_timing {
    double ms_per = 0.0, ms_min = 0.0, ms_max = 0.0;
    int measured = 0;
    double flush_mb = 0.0, l2_mb = 0.0;

    double gflops(int M, int N, int K) const {
        return (2.0 * M * N * K / 1.0e9) / (ms_per * 1.0e-3);
    }
    double spread_pct() const { return 100.0 * (ms_max - ms_min) / ms_per; }
};

static hk_timing hk_run_protocol(gfx1250_gemm::gemm_globals& g,
                                 const gfx1250_gemm::launch_config& launch,
                                 int measured)
{
    if (measured <= 0) measured = HK_MEASURED_ITERS;
    const hipStream_t stream = launch.stream;

    int dev = 0;
    HIP_OK(hipGetDevice(&dev));
    hipDeviceProp_t props{};
    HIP_OK(hipGetDeviceProperties(&props, dev));
    // Allocated after the operands, so the flush scratch never crowds them out.
    cache_flusher flusher;
    flusher.init(props);

    for (int i = 0; i < HK_WARMUP_ITERS; ++i) {
        flusher.run(stream);
        dispatch(g, launch);
    }
    HIP_OK(hipDeviceSynchronize());

    std::vector<hipEvent_t> ev_beg(measured), ev_end(measured);
    for (int i = 0; i < measured; ++i) {
        HIP_OK(hipEventCreate(&ev_beg[i]));
        HIP_OK(hipEventCreate(&ev_end[i]));
    }
    for (int i = 0; i < measured; ++i) {
        flusher.run(stream);
        HIP_OK(hipEventRecord(ev_beg[i], stream));
        dispatch(g, launch);
        HIP_OK(hipEventRecord(ev_end[i], stream));
    }
    HIP_OK(hipEventSynchronize(ev_end[measured - 1]));

    hk_timing t;
    t.measured = measured;
    t.flush_mb = flusher.mb();
    t.l2_mb    = flusher.l2_mb;
    double ms_sum = 0.0;
    for (int i = 0; i < measured; ++i) {
        float ms = 0.f;
        HIP_OK(hipEventElapsedTime(&ms, ev_beg[i], ev_end[i]));
        ms_sum += ms;
        t.ms_min = (i == 0) ? ms : std::min(t.ms_min, double(ms));
        t.ms_max = std::max(t.ms_max, double(ms));
        HIP_OK(hipEventDestroy(ev_beg[i]));
        HIP_OK(hipEventDestroy(ev_end[i]));
    }
    flusher.destroy();
    t.ms_per = ms_sum / measured;
    return t;
}

#include "pyext.h"

#endif // HARNESS_PYEXT
