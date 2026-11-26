/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace rv_layout
 * 
 * @brief A namespace for template metaprogramming with register vector layouts.
 */
namespace rv_layout {

/**
 * @brief A dummy type used to identify an aligned layout.
 *
 * "Aligned" means the vector's index runs along a tile's *element* axis, so a
 * lane holds several entries and the whole vector is replicated across the 16
 * lanes of a wave half.  On gfx11 the number of entries per lane depends on the
 * element type -- 8 for an f32 accumulator, 16 for a bf16/half WMMA operand --
 * so unlike upstream there is no single inner_dim here; rv computes it from
 * WMMA_REPLICATION.
 */
struct align {};
/**
 * @brief A dummy type used to identify an orthogonal layout.
 *
 * "Orthogonal" means the vector's index runs along the *lane* axis (l%16), so
 * each lane holds exactly one entry per subtile, replicated across the two wave
 * halves.
 */
struct ortho { constexpr static int inner_dim = 1; };
/**
 * @brief A dummy type used to identify an unreplicated layout, for better coalesced loads and vector operations like layernorm.
 */
struct naive { constexpr static int inner_dim = 1; };

/**
 * @brief A concept to check if a type is a register tile layout.
 */
template<typename T>
concept all = std::is_same_v<T, align> || std::is_same_v<T, ortho> || std::is_same_v<T, naive>;

} // namespace rv_layout
} // namespace ducks
} // namespace kittens