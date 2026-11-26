// Which gfx11 hardware facilities does HipKittens rely on, and which exist here?
// Each block is guarded so we can bisect with -D flags; the driver script below
// compiles them one at a time and reports availability.
#include <hip/hip_runtime.h>
#include <cstdio>

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));
typedef float    float4v __attribute__((ext_vector_type(4)));
typedef __bf16   bf16x16 __attribute__((ext_vector_type(16)));
typedef short    short16 __attribute__((ext_vector_type(16)));

#if defined(F_BUFFER_LOAD_LDS)
// direct global->LDS DMA through a buffer resource (CDNA's async copy path)
__global__ void k(__amdgpu_buffer_rsrc_t rsrc, __attribute__((address_space(3))) void *lds) {
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, lds, 4, 0, 0, 0, 0);
}
#elif defined(F_GLOBAL_LOAD_LDS)
__global__ void k(const __attribute__((address_space(1))) unsigned *g,
                  __attribute__((address_space(3))) unsigned *l) {
    __builtin_amdgcn_global_load_lds(g, l, 4, 0, 0);
}
#elif defined(F_SCHED_BARRIER)
__global__ void k() { __builtin_amdgcn_sched_barrier(0); }
#elif defined(F_SCHED_GROUP_BARRIER)
__global__ void k() { __builtin_amdgcn_sched_group_barrier(8, 1, 0); }
#elif defined(F_IGLP_OPT)
__global__ void k() { __builtin_amdgcn_iglp_opt(0); }
#elif defined(F_SETPRIO)
__global__ void k() { __builtin_amdgcn_s_setprio(1); }
#elif defined(F_PERMLANE16)
__global__ void k(unsigned *o, unsigned v) {
    *o = __builtin_amdgcn_permlane16(v, v, 0u, 0u, false, false);
}
#elif defined(F_PERMLANE32_SWAP)
__global__ void k(unsigned *o, unsigned a, unsigned b) {
    // gfx950-only swap; expected to fail on gfx11
    *o = __builtin_amdgcn_permlane32_swap(a, b, false, false).x;
}
#elif defined(F_DS_BPERMUTE)
__global__ void k(int *o, int a, int b) { *o = __builtin_amdgcn_ds_bpermute(a, b); }
#elif defined(F_DOT2)
__global__ void k(float *o, float v) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    h2 a = {(_Float16)v, (_Float16)v};
    *o = __builtin_amdgcn_fdot2(a, a, v, false);
}
#elif defined(F_WMMA_BF16)
__global__ void k(float8 *o, short16 a, short16 b, float8 c) {
    *o = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
#elif defined(F_WMMA_F16_ACC)
__global__ void k(half16 *o, half16 a, half16 b, half16 c) {
    *o = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c, false);
}
#elif defined(F_SWMMAC)
__global__ void k(float8 *o, half16 a, half16 b, float8 c, short i) {
    *o = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c, i, 0);
}
#elif defined(F_DS_READ_TR)
__global__ void k(__attribute__((address_space(3))) void *p, short *o) {
    typedef short s4 __attribute__((ext_vector_type(4)));
    s4 r = __builtin_amdgcn_ds_read_tr16_b64_v4i16(p);
    *o = r[0];
}
#elif defined(F_S_WAITCNT_VSCNT)
__global__ void k() { __builtin_amdgcn_s_waitcnt(0); }
#elif defined(F_LDS_SIZE)
extern "C" __global__ void k(float *o) {
    __shared__ float buf[LDS_FLOATS];
    buf[threadIdx.x] = o[threadIdx.x];
    __syncthreads();
    o[threadIdx.x] = buf[LDS_FLOATS - 1 - threadIdx.x];
}
#else
__global__ void k() {}
#endif

int main() { printf("ok\n"); return 0; }
