// How many bytes per clock does LDS actually deliver on gfx11, under the
// specific access patterns a GEMM inner loop uses?
//
// wmma_peak.hip establishes the math ceiling. This is the other half: the GEMM
// reads operand tiles out of LDS every iteration, and if that stage is
// bandwidth- or conflict-bound then no amount of tile reshaping helps until the
// access pattern changes.
//
// Four patterns, all reading the same total bytes so the numbers are directly
// comparable:
//
//   LINEAR  lane l reads 16B at l*16        -- perfectly coalesced, the ceiling
//   TILE16  lane l reads 16B at (l%16)*ROW  -- what HipKittens does: a wave32
//           operand tile is mirrored across the halves, so only 16 distinct
//           rows are touched and lanes l, l+16 hit the same address
//   TILE16X TILE16 plus the *CDNA* swizzle (16B-block key, 8B flip). Kept as a
//           control: a ds_read_b128 ignores the low 4 address bits, so flipping
//           bit 3 is invisible to it and this must measure identical to TILE16.
//   ST      the swizzle the rdna3 tree actually uses, reproducing the GEMM's
//           addressing exactly: A/B tiles are st<bf16,128,32> so subtile_cols is
//           32 and the row stride is 64B, a lane reads rows r, r+16, r+32, r+48
//           as two ds_read_b128 each, and off ^= ((off % 1024) >> 7) << 4
//   XOR4    TILE16 with the 16B block index inside each row XORed by row&3.
//           A K_STEP=32 row is 64B = four 16B blocks, so this is the widest
//           permutation that fits inside one row: it spreads 16 lanes over 4
//           bank groups instead of 2.
//   K64     the same trick with K_STEP=64, where a row is 128B = eight blocks
//           and row&7 spreads all 16 lanes over 8 groups
//   U16     lane l reads 2B at l*2 + k*256  -- what rocBLAS/Tensile does:
//           one bf16 element per instruction, 32 lanes covering 64 contiguous
//           bytes, K strided far apart. This is the transposed (K-major) LDS
//           layout, and it is conflict-free by construction.
//
// Report is bytes/clk/CU, which is the number to compare against the LDS
// traffic a kernel's inner loop demands.
#include <hip/hip_runtime.h>
#include <cstdio>

#ifndef ITERS
#define ITERS 2048
#endif
#define ROW 64            // bytes between rows: K_STEP=32 bf16
#define LDS_BYTES 16384

typedef __attribute__((__vector_size__(4 * sizeof(int)))) int int4_t;

// No waitcnt inside these: waiting per load would measure LDS *latency*, not
// bandwidth. The eight loads of an unrolled group go in flight together and a
// single lds_wait() at the bottom retires them, which is what the GEMM does.
__device__ __forceinline__ int4_t ds_read_b128(unsigned addr) {
    int4_t r;
    asm volatile("ds_read_b128 %0, %1" : "=v"(r) : "v"(addr) : "memory");
    return r;
}
__device__ __forceinline__ unsigned ds_read_u16(unsigned addr) {
    unsigned r;   // must be a full VGPR; a short operand is rejected
    asm volatile("ds_read_u16 %0, %1" : "=v"(r) : "v"(addr) : "memory");
    return r;
}
__device__ __forceinline__ void lds_wait() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// 0 LINEAR, 1 TILE16, 2 TILE16X, 3 U16, 4 XOR4, 5 K64, 6 ST, 7 ST_HALF, 8 ST32
//
// 7 and 8 exist to answer one question: does the wave-half mirror cost LDS
// bandwidth, or does the hardware broadcast it for free? A gfx11 WMMA operand
// requires lanes l and l+16 to hold identical data, so ST issues 32 lane-slots
// against only 16 distinct addresses. If LDS charges per lane, half of every
// operand read in the GEMM is waste and masking it off would be worth 2x.
//
//   ST_HALF  ST with exec = 0xffff: same distinct bytes, half the lane-slots.
//   ST32     ST with 32 distinct rows: same lane-slots, twice the distinct
//            bytes.
//
// One of the two must move. If ST_HALF is 2x faster, LDS is lane-slot bound and
// the mirror is pure waste. If ST32 is 2x slower, LDS is distinct-byte bound
// and the mirror is already free.
//
// Neither moves. ST 1.978 ms, ST_HALF 1.969, ST32 1.975 -- half the lanes and
// twice the distinct bytes both cost exactly the same. A conflict-free
// ds_read_b128 on gfx1100 has a **fixed cost per instruction**, independent of
// how many lanes are active and of how many distinct addresses they name. Two
// consequences for a GEMM, and they point opposite ways:
//
//   - The wave-half mirror is free. Masking it off, or reconstructing the high
//     half with permlane, buys nothing; there is no bandwidth being wasted.
//   - The only way to spend less on LDS is to *issue fewer ds_read*. That count
//     is 2*(16/M + 16/N) per WMMA for an MxN warp tile and nothing else moves
//     it, which is why the warp tile is the whole ballgame.
template <int MODE>
__global__ __launch_bounds__(256)
void lds_bw(unsigned *sink, int iters) {
    __shared__ char lds[LDS_BYTES];
    for (int i = threadIdx.x * 4; i < LDS_BYTES; i += blockDim.x * 4)
        *(int *)&lds[i] = i;
    __syncthreads();

    const unsigned base = (unsigned)(unsigned long long)(&lds[0]);
    const int lane = threadIdx.x & 31;
    unsigned acc = 0;

    if constexpr (MODE == 6) {
        // st<bf16,128,32>::idx(): off = 2*(r*32 + c), swizzled. Eight reads =
        // four base tiles (rows r, r+16, r+32, r+48) x two 16B halves, which is
        // the A-operand half of one dot_tile at REG_BLOCK_M=32 plus part of B.
        const int r = lane & 15;
        for (int it = 0; it < iters; it++) {
            int4_t v[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                const int row  = r + 16 * (k >> 1);
                unsigned  off  = 64u * row + 32u * (k & 1) + (unsigned)(it & 1) * 8192u;
                off ^= ((off % 1024u) >> 7) << 4;
                v[k] = ds_read_b128(base + (off & (LDS_BYTES - 1)));
            }
            lds_wait();
            #pragma unroll
            for (int k = 0; k < 8; k++) acc += v[k][0] + v[k][3];
        }
    } else if constexpr (MODE == 7 || MODE == 8) {
        // MODE 8 drops the `& 15`, so the 32 lanes touch 32 distinct rows.
        const int r = (MODE == 8) ? lane : (lane & 15);
        // The exec mask goes around the whole loop, not around the loads. Put
        // it inside and the compiler pays s_and_saveexec plus a zero-init of
        // v[] every iteration, which is what is being timed instead of LDS.
        // Out here it is one s_mov before the loop and the body is otherwise
        // instruction-identical to MODE 6.
        if (MODE == 8 || lane < 16) {
            for (int it = 0; it < iters; it++) {
                int4_t v[8];
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    const int row  = r + 16 * (k >> 1);
                    unsigned  off  = 64u * row + 32u * (k & 1) + (unsigned)(it & 1) * 8192u;
                    off ^= ((off % 1024u) >> 7) << 4;
                    v[k] = ds_read_b128(base + (off & (LDS_BYTES - 1)));
                }
                lds_wait();
                #pragma unroll
                for (int k = 0; k < 8; k++) acc += v[k][0] + v[k][3];
            }
        }
    } else if constexpr (MODE == 3) {
        // 8 u16 per "tile row", stepping K by 256B, to match Tensile.
        unsigned a = base + lane * 2;
        for (int it = 0; it < iters; it++) {
            unsigned v[8];
            #pragma unroll
            for (int k = 0; k < 8; k++)
                v[k] = ds_read_u16(a + ((k * 256 + it * 16) & (LDS_BYTES - 1)));
            lds_wait();
            #pragma unroll
            for (int k = 0; k < 8; k++) acc += v[k];
        }
    } else {
        const int  r   = lane & 15;
        const unsigned row = (MODE == 5) ? 128u : (unsigned)ROW;
        unsigned a = (MODE == 0) ? base + lane * 16 : base + r * row;
        for (int it = 0; it < iters; it++) {
            int4_t v[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                unsigned off  = (k * 1024 + it * 16) & (LDS_BYTES - 1);
                unsigned addr = a + off;
                if constexpr (MODE == 2) {
                    // st.cuh's swizzle: XOR 8 bytes according to address bit 7.
                    addr ^= (((addr >> 7) & 1) << 3);
                } else if constexpr (MODE == 4) {
                    addr = (addr & ~0x30u) | (((addr >> 4) ^ (unsigned)r) & 3) << 4;
                } else if constexpr (MODE == 5) {
                    addr = (addr & ~0x70u) | (((addr >> 4) ^ (unsigned)r) & 7) << 4;
                }
                v[k] = ds_read_b128(addr & ~15u);
            }
            lds_wait();
            #pragma unroll
            for (int k = 0; k < 8; k++) acc += v[k][0] + v[k][3];
        }
    }
    if (acc == 0xdeadbeef) sink[0] = acc;
}

template <int MODE>
void run(const char *name, int cus, double mhz) {
    const int wgs = cus * 4, threads = 256;
    unsigned *sink;
    hipMalloc(&sink, 4);
    for (int r = 0; r < 3; r++) {
        hipLaunchKernelGGL((lds_bw<MODE>), dim3(wgs), dim3(threads), 0, 0, sink, ITERS);
        hipDeviceSynchronize();
    }
    hipEvent_t s, e;
    hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    const int reps = 10;
    for (int r = 0; r < reps; r++)
        hipLaunchKernelGGL((lds_bw<MODE>), dim3(wgs), dim3(threads), 0, 0, sink, ITERS);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms; hipEventElapsedTime(&ms, s, e); ms /= reps;

    const double waves = (double)wgs * threads / 32.0;
    const double per_lane = (MODE == 3) ? 2.0 : 16.0;   // bytes each lane asks for
    const double active   = (MODE == 7) ? 16.0 : 32.0;  // lanes the exec mask lets through
    // bytes actually requested by the instruction stream
    const double bytes = waves * active * ITERS * 8.0 * per_lane;
    const double bpc_cu = bytes / (ms * 1e-3) / (mhz * 1e6) / cus;
    printf("  %-8s %8.3f ms   %8.1f GB/s   %7.1f bytes/clk/CU\n",
           name, ms, bytes / (ms * 1e6), bpc_cu);
}

int main() {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    // multiProcessorCount is WGPs on RDNA; a WGP is 2 CUs.
    const int cus = p.multiProcessorCount * 2;
    // Clock under an LDS-only load; pass the measured one in if it differs.
    const double mhz = 2100.0;
    printf("device: %s, %d CU (%d WGP), assuming %.0f MHz under load\n",
           p.gcnArchName, cus, p.multiProcessorCount, mhz);
    run<0>("LINEAR",  cus, mhz);
    run<1>("TILE16",  cus, mhz);
    run<2>("TILE16X", cus, mhz);
    run<4>("XOR4",    cus, mhz);
    run<5>("K64",     cus, mhz);
    run<6>("ST",      cus, mhz);
    run<7>("ST_HALF", cus, mhz);
    run<8>("ST32",    cus, mhz);
    run<3>("U16",     cus, mhz);
    return 0;
}
