/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace rt_layout
 * 
 * @brief A namespace for template metaprogramming with register tile layouts.
 */
namespace rt_layout {

/**
 * @brief A dummy type used to identify a row-major layout for a register tile.
 */
struct row {}; // for most matrices
/**
 * @brief A dummy type used to identify a col-major layout for a register tile.
 */
struct col {}; // for the B-matrix of MMA ops.

#ifdef KITTENS_CDNA4
/**
 * @brief A dummy type used to identify a 32x32 layout for a register tile.
 */
struct accumulator {};
#endif

/**
 * @brief A concept to check if a type is a register tile layout.
 */
template<typename T>
#ifdef KITTENS_CDNA4
concept all = std::is_same_v<T, row> || std::is_same_v<T, col> || std::is_same_v<T, accumulator>;
#else
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;
#endif

/**
 * @brief A struct to generate a transposed layout.
 */
template<all L> struct transpose      { using type = col; };
template<>      struct transpose<col> { using type = row; };

} // namespace rt_layout
} // namespace ducks
} // namespace kittens