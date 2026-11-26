/**
 * @file
 * @brief Counter-based random numbers into a register tile.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

namespace detail {

/// Philox4x32 with Triton's constants and round order (triton.language.random
/// philox_impl), so a kernel and its Triton reference draw the same numbers
/// from the same seed and offsets.
__device__ inline uint32_t philox_first(uint32_t c0, uint32_t k0, uint32_t k1,
                                        int rounds = 10) {
    uint32_t c1 = 0, c2 = 0, c3 = 0;
    #pragma unroll
    for (int r = 0; r < rounds; r++) {
        const uint32_t a0 = c0, a2 = c2;
        c0 = __umulhi(0xCD9E8D57u, a2) ^ c1 ^ k0;
        c2 = __umulhi(0xD2511F53u, a0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * a2;
        c3 = 0xD2511F53u * a0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c0;
}

/// Triton's uint_to_uniform_float for a 32-bit draw: [0, 1), the sign bit
/// folded rather than dropped.
__device__ inline float uniform_from(uint32_t x) {
    int32_t s = int32_t(x);
    s = s < 0 ? -s - 1 : s;
    float u = float(s) * 4.6566127342e-10f;
    // Rounded here, as Triton rounds it. Contracted into a following
    // subtraction (r - p, a dropout's mask) the product would skip this
    // rounding, and a draw that rounds onto p would land on the other side.
    asm("" : "+v"(u));
    return u;
}

} // namespace detail

/**
 * @brief Fill an fp32 tile with uniform [0, 1) draws: element (r, c) is
 * Philox(seed, offset + r * row_stride + c), as triton.language.rand(seed,
 * offsets) draws it for that offset. `seed_hi` is the seed's upper 32 bits.
 */
template<ducks::rt::all RT>
__device__ inline void philox_uniform(RT &dst, uint32_t seed_lo, uint32_t seed_hi,
                                      int offset, int row_stride) {
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using L = typename RT::layout;
    static_assert(std::is_same_v<T, float>, "draws go into an fp32 tile");
    constexpr int E = rt_base<T, L>::elements_per_thread;
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int e = 0; e < E; e++) {
                const int2 c = rt_base_coord<T, L>(e, lane);
                const uint32_t ctr = uint32_t(offset + (i * dst.tile_size_row + c.x) * row_stride +
                                              j * dst.tile_size_col + c.y);
                const float v = detail::uniform_from(detail::philox_first(ctr, seed_lo, seed_hi));
                if (e & 1) dst.tiles[i][j].data[e >> 1].y = v;
                else       dst.tiles[i][j].data[e >> 1].x = v;
            }
        }
    }
}

} // namespace kittens
