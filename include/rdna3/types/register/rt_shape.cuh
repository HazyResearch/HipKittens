/**
 * @file
 * @brief The register-tile base shape, for architectures that have exactly one.
 */

#pragma once

#include <concepts>

#include "../../common/common.cuh"

namespace kittens {
namespace ducks {
/**
 * @namespace rt_shape
 *
 * @brief Names the shape of the base tile a register tile is built from.
 *
 * On CDNA this namespace enumerates the MFMA instruction shapes -- 16x16x16,
 * 32x32x8 and so on -- and the shape is a real degree of freedom that callers
 * pick per tile. gfx11 has exactly one matrix instruction geometry,
 * `v_wmma_*_16x16x16_*`, so there is exactly one member here. It is kept as a
 * duck rather than folded away so that arch-agnostic code (the unit test
 * harness in particular) can be written once against the CDNA API and
 * instantiated here without a separate spelling.
 *
 * `stride` is the number of elements a lane holds in one contiguous run along
 * the reduction axis. On gfx11 an operand lane holds the whole K=16 vector, so
 * a run is the full 16.
 */
namespace rt_shape {

template<int _rows, int _cols, int _stride>
struct rt_shape {
    static constexpr int rows = _rows;
    static constexpr int cols = _cols;
    static constexpr int stride = _stride;
    static constexpr int num_elements = rows*cols;
};

using rt_16x16 = rt_shape<16, 16, 16>;

template<typename T>
concept all = std::is_same_v<T, rt_16x16>;

/// gfx11's base tile is square, so transposing one does not change its shape.
template<all L> struct transpose { using type = rt_16x16; };

} // namespace rt_shape
} // namespace ducks
} // namespace kittens
