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
 */
namespace st_layout {

/**
 * @brief A dummy type used to identify a row-major layout for a shared memory tile.
 */
struct row {}; // for most matrices
/**
 * @brief A dummy type used to identify a col-major layout for a shared memory tile.
 */
struct col {}; // for the B-matrix of MMA ops.
/**
 * @brief A dummy type used to identify an accumulator col-major layout for a shared memory tile.
 */
struct accumulator_col {};
/**
 * @brief A dummy type used to identify an accumulator row-major layout for a shared memory tile.
 */
struct accumulator_row {};

/**
 * @brief A concept to check if a type is a shared memory tile layout.
 */
template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col> || std::is_same_v<T, accumulator_col> || std::is_same_v<T, accumulator_row>;

} // namespace st_layout
} // namespace ducks
} // namespace kittens