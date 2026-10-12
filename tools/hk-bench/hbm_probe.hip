// How fast can a kernel stream bytes it reads exactly once, on this card.
//
// A decode GEMV is that and nothing else: each weight byte is read once per
// step and the math is free. So the number a GEMV should be judged against is
// not the 864 GB/s on the datasheet but what a pure read reaches -- at the
// sizes the weights actually are, because a 33 MB matrix is read in ~40 us
// and the ramp-up and tail of a kernel that short are part of what it pays.
//
// Every read comes from a different buffer, rotated through 4 x 288 MB, so
// nothing can be served from the 96 MB Infinity Cache. Sweeps grid size, block
// size and loads in flight per thread; reports the best per size.
//
//   hipcc -O3 --offload-arch=gfx1100 -o /tmp/hbm_probe tools/hk-bench/hbm_probe.hip
//   HIP_VISIBLE_DEVICES=0 /tmp/hbm_probe

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                              \
    do {                                                                      \
        hipError_t e_ = (x);                                                  \
        if (e_ != hipSuccess) {                                               \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,            \
                         hipGetErrorString(e_));                              \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

template <int UNROLL>
__global__ void read_kernel(const uint4 *__restrict__ p, size_t n16,
                            uint4 *sink) {
    uint4 acc = {0, 0, 0, 0};
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (UNROLL - 1) * stride < n16; i += UNROLL * stride) {
        uint4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = p[i + u * stride];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w;
        }
    }
    for (; i < n16; i += stride) {
        uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    // Never true for the fill below; it keeps the loads from being dead.
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) sink[0] = acc;
}

using Launch = void (*)(const uint4 *, size_t, uint4 *, int, int);

template <int U>
static void launch(const uint4 *p, size_t n16, uint4 *sink, int grid,
                   int block) {
    read_kernel<U><<<grid, block>>>(p, n16, sink);
}

int main() {
    const size_t buf_bytes = 288ull << 20;  // 3x the Infinity Cache
    const int n_bufs = 4;
    std::vector<uint4 *> bufs(n_bufs);
    for (auto &b : bufs) {
        CHECK(hipMalloc(&b, buf_bytes));
        CHECK(hipMemset(b, 0x5a, buf_bytes));
    }
    uint4 *sink;
    CHECK(hipMalloc(&sink, sizeof(uint4)));

    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("%s, %d CUs, %s\n", prop.gcnArchName, prop.multiProcessorCount,
                "GB/s of bytes read once (best over grid/block/unroll)");

    // Qwen3-8B's linear weights in bf16, plus a size big enough to be steady.
    struct Size { const char *name; size_t bytes; };
    const Size sizes[] = {
        {"o 4096x4096", 4096ull * 4096 * 2},
        {"qkv 6144x4096", 6144ull * 4096 * 2},
        {"down 4096x12288", 4096ull * 12288 * 2},
        {"gate_up 24576x4096", 24576ull * 4096 * 2},
        {"288 MB", buf_bytes},
    };
    const Launch launches[] = {launch<1>, launch<2>, launch<4>, launch<8>};
    const int unrolls[] = {1, 2, 4, 8};
    const int grids_per_cu[] = {1, 2, 4, 8, 16, 32};
    const int blocks[] = {256, 512, 1024};

    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));

    for (const Size &s : sizes) {
        const size_t n16 = s.bytes / 16;
        double best = 0;
        int bg = 0, bb = 0, bu = 0;
        for (int ui = 0; ui < 4; ++ui)
            for (int gpc : grids_per_cu)
                for (int block : blocks) {
                    const int grid = gpc * prop.multiProcessorCount;
                    // Warm, then time 12 reads, each from the next buffer, and
                    // keep the fastest: the question is what is reachable.
                    int bi = 0;
                    for (int w = 0; w < 4; ++w)
                        launches[ui](bufs[bi++ % n_bufs], n16, sink, grid,
                                     block);
                    float ms_min = 1e30f;
                    for (int r = 0; r < 12; ++r) {
                        CHECK(hipEventRecord(t0));
                        launches[ui](bufs[bi++ % n_bufs], n16, sink, grid,
                                     block);
                        CHECK(hipEventRecord(t1));
                        CHECK(hipEventSynchronize(t1));
                        float ms;
                        CHECK(hipEventElapsedTime(&ms, t0, t1));
                        ms_min = std::min(ms_min, ms);
                    }
                    const double gbs = s.bytes / (ms_min * 1e-3) / 1e9;
                    if (gbs > best) {
                        best = gbs; bg = grid; bb = block; bu = unrolls[ui];
                    }
                }
        std::printf("%-20s %8.1f MB  %6.1f GB/s  %5.1f%% of 864   "
                    "(grid %d, block %d, unroll %d)\n",
                    s.name, s.bytes / 1e6, best, 100 * best / 864.0, bg, bb,
                    bu);
    }
    return 0;
}
