/**
 * @file
 * @brief Layout concepts for shared memory tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace st_layout
 *
 * @brief A namespace for template metaprogramming with shared memory tile layouts.
 *
 * Layout is orthogonal to shape: layout says which axis is contiguous, shape says how addresses
 * are permuted within the tile for bank-conflict avoidance.
 */
namespace st_layout {

/**
 * @brief A dummy type used to identify a row-major layout for a shared memory tile.
 */
struct row {}; // for most matrices
/**
 * @brief A dummy type used to identify a col-major layout for a shared memory tile.
 */
struct col {}; // for a column-major C tile, and the B-matrix of MMA ops.

/**
 * @brief A concept to check if a type is a shared memory tile layout.
 */
template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;

} // namespace st_layout
} // namespace ducks
} // namespace kittens
