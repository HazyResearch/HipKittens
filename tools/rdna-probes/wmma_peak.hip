// What is the real WMMA issue ceiling on this part, and at what clock?
//
// No memory at all in the inner loop: ACC independent accumulators chained only
// through themselves, so the only limit is how fast the SIMD retires
// v_wmma_f32_16x16x16_bf16. Whatever this reports is the number a GEMM on this
// architecture is actually competing against; the 122.6 TFLOPs on the spec
// sheet is boost clock times a theoretical FLOP/clk and assumes both.
#include <hip/hip_runtime.h>
#include <cstdio>

typedef __attribute__((__vector_size__(16 * sizeof(short)))) short  bf16x16;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float  f32x8;

#ifndef ACC
#define ACC 8          // independent accumulator chains, to cover WMMA latency
#endif
#ifndef ITERS
#define ITERS 4096
#endif

__global__ __launch_bounds__(256)
void wmma_peak(float *sink, int iters) {
    bf16x16 a, b;
    for (int i = 0; i < 16; i++) { a[i] = (short)(threadIdx.x + i); b[i] = (short)(i); }

    f32x8 c[ACC];
    #pragma unroll
    for (int j = 0; j < ACC; j++)
        for (int i = 0; i < 8; i++) c[j][i] = 0.f;

    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int j = 0; j < ACC; j++)
            c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c[j]);
    }

    // Keep everything live without a real store.
    float s = 0;
    #pragma unroll
    for (int j = 0; j < ACC; j++) for (int i = 0; i < 8; i++) s += c[j][i];
    if (s == 1234.5678f) sink[0] = s;
}

int main() {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    const int cus = p.multiProcessorCount;
    printf("device: %s, %d CU, clockRate %.3f GHz (reported peak)\n",
           p.gcnArchName, cus, p.clockRate / 1e6);

    // Fill the machine: 8 workgroups of 256 threads per CU.
    const int wgs = cus * 8, threads = 256;
    float *sink; hipMalloc(&sink, 4);

    for (int rep = 0; rep < 3; rep++) {
        hipLaunchKernelGGL(wmma_peak, dim3(wgs), dim3(threads), 0, 0, sink, ITERS);
        hipDeviceSynchronize();
    }

    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    const int reps = 20;
    for (int r = 0; r < reps; r++)
        hipLaunchKernelGGL(wmma_peak, dim3(wgs), dim3(threads), 0, 0, sink, ITERS);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms; hipEventElapsedTime(&ms, s, e); ms /= reps;

    // One wave32 WMMA 16x16x16 = 2*16*16*16 FLOP. waves = wgs*threads/32.
    const double waves = (double)wgs * threads / 32.0;
    const double flop  = waves * (double)ITERS * ACC * 2.0 * 16 * 16 * 16;
    printf("ACC=%d ITERS=%d  %.3f ms  ->  %.1f TFLOPs\n",
           ACC, ITERS, ms, flop / (ms * 1e9));
    return 0;
}
