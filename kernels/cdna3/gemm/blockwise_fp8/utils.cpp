
#pragma once

#include <type_traits>

typedef int int32x4_lds_t __attribute__((ext_vector_type(4)));
struct __attribute__((packed)) buf_res { const void *ptr; uint32_t range; uint32_t config; };
__device__ inline int32x4_lds_t make_buf_res(const void *ptr, uint32_t size) {
    buf_res r{ptr, size, 0x00020000u};
    return __builtin_bit_cast(int32x4_lds_t, r);
}
extern "C" __device__ __uint128_t
llvm_amdgcn_raw_buffer_load_b128(int32x4_lds_t rsrc, int voffset, int soffset,
                                 int aux) __asm("llvm.amdgcn.raw.buffer.load.v4f32");
extern "C" __device__ float
llvm_amdgcn_s_buffer_load_f32(int32x4_lds_t rsrc, int offset,
                              int cachepolicy) __asm("llvm.amdgcn.s.buffer.load.f32");
extern "C" __device__ float
llvm_amdgcn_raw_buffer_load_f32(int32x4_lds_t rsrc, int voffset, int soffset,
                                int aux) __asm("llvm.amdgcn.raw.buffer.load.f32");

template <int HEIGHT>
__device__ inline void load_scale_global_reg(float (&sa_reg)[HEIGHT * 4], const float *sa_base,
                                             int local_m_base, uint32_t range_bytes) {
    const int lane = kittens::laneid();
    const int row_g = 4 * (lane / 16);
    int32x4_lds_t srsrc = make_buf_res((const void*)sa_base, range_bytes);
    #pragma unroll
    for (int i = 0; i < HEIGHT; i++) {
        const int m0 = local_m_base + i * 16 + row_g;
        __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, m0 * 4, 0, 0);
        *reinterpret_cast<float4*>(&sa_reg[i * 4]) = *reinterpret_cast<float4*>(&raw);
    }
}

template <int WIDTH>
__device__ inline void load_scaleB_global_reg(float (&sb_reg)[WIDTH], const float *sb_base,
                                              int local_n_base, uint32_t range_bytes) {
    const int lane = kittens::laneid();
    const int col_l = lane % 16;
    int32x4_lds_t srsrc = make_buf_res((const void*)sb_base, range_bytes);
    #pragma unroll
    for (int j = 0; j < WIDTH; j++) {
        const int n0 = local_n_base + j * 16 + col_l;
        sb_reg[j] = llvm_amdgcn_raw_buffer_load_f32(srsrc, n0 * 4, 0, 0);
    }
}

__device__ inline float rtne_bias(float v) {
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    if ((bits & 0x7f800000u) == 0x7f800000u) return v;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __builtin_bit_cast(float, bits);
}

template <typename OType>
__device__ inline OType convert_out(float v) {
    if constexpr (std::is_same_v<OType, kittens::bf16>) {
        return kittens::base_types::convertor<OType, float>::convert(rtne_bias(v));
    } else {
        return kittens::base_types::convertor<OType, float>::convert(v);
    }
}

template <typename OType, typename AccType>
__device__ inline void store_output(OType *c_ptr, const AccType &Cacc,
                                    int Rtile, int Ctile, int M, int N) {
    const int lane = kittens::laneid();
    const int m_base = Rtile * AccType::rows + 4 * (lane / 16);
    const int n_base = Ctile * AccType::cols + (lane % 16);
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const int m0 = m_base + i * 16;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const int col = n_base + j * 16;
            c_ptr[(m0 + 0) * N + col] = convert_out<OType>(Cacc.tiles[i][j].data[0].x);
            c_ptr[(m0 + 1) * N + col] = convert_out<OType>(Cacc.tiles[i][j].data[0].y);
            c_ptr[(m0 + 2) * N + col] = convert_out<OType>(Cacc.tiles[i][j].data[1].x);
            c_ptr[(m0 + 3) * N + col] = convert_out<OType>(Cacc.tiles[i][j].data[1].y);
        }
    }
}

template <typename AccType>
__device__ inline void apply_block_scale_1d2d(
    AccType &Cacc, const AccType &partial, const float (&sa_reg)[AccType::height * 4], float sb) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const float s0 = sa_reg[i * 4 + 0] * sb;
        const float s1 = sa_reg[i * 4 + 1] * sb;
        const float s2 = sa_reg[i * 4 + 2] * sb;
        const float s3 = sa_reg[i * 4 + 3] * sb;
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * s0;
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * s1;
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * s2;
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * s3;
        }
    }
}

template <typename AccType>
__device__ inline void apply_block_scale_1d1d(
    AccType &Cacc, const AccType &partial, const float (&sa_reg)[AccType::height * 4],
    const float (&sb_reg)[AccType::width]) {
    #pragma unroll
    for (int i = 0; i < AccType::height; i++) {
        const float a0 = sa_reg[i * 4 + 0];
        const float a1 = sa_reg[i * 4 + 1];
        const float a2 = sa_reg[i * 4 + 2];
        const float a3 = sa_reg[i * 4 + 3];
        #pragma unroll
        for (int j = 0; j < AccType::width; j++) {
            const float sb = sb_reg[j];
            Cacc.tiles[i][j].data[0].x += partial.tiles[i][j].data[0].x * (a0 * sb);
            Cacc.tiles[i][j].data[0].y += partial.tiles[i][j].data[0].y * (a1 * sb);
            Cacc.tiles[i][j].data[1].x += partial.tiles[i][j].data[1].x * (a2 * sb);
            Cacc.tiles[i][j].data[1].y += partial.tiles[i][j].data[1].y * (a3 * sb);
        }
    }
}
