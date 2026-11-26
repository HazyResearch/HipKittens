/**
 * @file
 * @brief The shared-tile base block shape.
 */

#pragma once

#include <concepts>

#include "../../common/common.cuh"

namespace kittens {
namespace ducks {
/**
 * @namespace st_shape
 *
 * @brief Names the base block a shared tile is tiled out of.
 *
 * A note on what this does and does not control here. On CDNA4 the shape is a
 * layout *selector*: `st<T, rows, cols, shape>` asks the shape for the address
 * of every element, so swapping `st_16x16` for `st_16x16_swizzled` genuinely
 * changes where the bytes land. rdna3's `st` descends from the cdna3 design,
 * where the XOR swizzle is chosen from the tile's own width inside
 * `st::idx()` -- see `st.cuh` -- and there is no shape parameter to pass.
 *
 * So the member below is a *description* of what rdna3's shared tiles already
 * do, not a knob. It exists because the shared unit test harness is written
 * against the CDNA API and names a shape in nearly every signature; giving
 * rdna3 the one shape its tiles actually use lets those tests be ported
 * without forking the harness. If the shared tile ever does become
 * shape-parameterized on this arch, this is where the alternatives go. The
 * swizzle retune is not such a case: it landed as a change to the granule and
 * repeat inside `st::idx()`, which every tile shape wants anyway.
 */
namespace st_shape {

struct st_16x16 {
    static constexpr int rows = 16;
    static constexpr int cols = 16;

    /// The width of one thread's global<->shared transfer. rdna3 moves shared
    /// tiles a `float4` at a time regardless of dtype; see global_to_shared.cuh.
    template<typename _T>
    static constexpr int bytes_per_thread() {
        static_assert(sizeof(_T) == 2 || sizeof(_T) == 4, "Unsupported type");
        return 16;
    }
};

template<typename T>
concept all = std::is_same_v<T, st_16x16>;

} // namespace st_shape
} // namespace ducks
} // namespace kittens
