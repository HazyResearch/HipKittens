/**
 * @file
 * @brief Layouts and their manipulations for global memory descriptors.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace gl_layout
 * 
 * @brief A namespace for template metaprogramming with global memory layouts.
 */
namespace gl_layout {

/**
 * @brief A dummy type used to identify a row-major layout for a global descriptor.
 */
struct row_major {}; // element (r,c) at (..*rows + r)*cols + c; the column axis is unit-stride
/**
 * @brief A dummy type used to identify a col-major layout for a global descriptor.
 */
struct col_major {}; // element (r,c) at (..*cols + c)*rows + r; the row axis is unit-stride

/**
 * @brief A concept to check if a type is a global memory layout.
 */

template<typename T>
concept all = std::is_same_v<T, row_major> || std::is_same_v<T, col_major>;

/**
 * @brief Always false, so an overload that exists in order to refuse can say why.
 *
 * `static_assert(false, ...)` in a template body is ill-formed regardless of instantiation, so
 * the condition has to depend on the template parameter. Unsupported layouts would fail to
 * resolve anyway; this buys a message instead of "no matching function".
 */
template<typename> inline constexpr bool unhandled = false;

} // namespace gl_layout
} // namespace ducks
} // namespace kittens
