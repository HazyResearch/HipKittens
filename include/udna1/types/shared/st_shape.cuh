/**
 * @file
 * @brief Layouts and their manipulations for shared tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
* @namespace st_shape
* 
* @brief A namespace for template metaprogramming with shared tile layouts.
*/
namespace st_shape {

struct st_16x16 {
    static constexpr int rows = 16;
    static constexpr int cols = 16;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
        
    }
};

struct st_16x16_swizzled {
    static constexpr int rows = 16;
    static constexpr int cols = 16;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2) {
            return 4;
        } else if constexpr (sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 512) >> 7) << 3;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
        
    }
};

struct st_32x32 {
    static constexpr int rows = 32;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int first_swizzle = ((offset % 1024) >> 9) << 5;
            const int second_swizzle = ((offset % 2048) >> 10) << 4;
            const int swizzled_offset = offset ^ first_swizzle ^ second_swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x32 {
    static constexpr int rows = 16;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 1024) >> 9) << 5;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_32x16 {
    static constexpr int rows = 32;
    static constexpr int cols = 16;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 1024) >> 9) << 4;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_8x32 {
    static constexpr int rows = 8;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

struct st_16x128 {
    static constexpr int rows = 16;
    static constexpr int cols = 128;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 1) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;

        const uint32_t offset = sizeof(T)*(r*cols + c);

        if constexpr (sizeof(T) == 1) {
            const int swizzle = ((offset % (16*128)) >> 8) << 4;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

/**
 * @brief 16x32 shared subtile with LDS bank-conflict padding -- gfx1250.
 *
 * Inserts `PAD_AMOUNT` pad elements every `PAD_INTERVAL` logical elements.
 * Both parameters are counted in **elements, not bytes**: at `sizeof(bf16) = 2`
 * the defaults `<128, 8>` are exactly the recommended "16 B padding every 256 B"
 * (8 elems x 2 B = 16 B inserted per 128 elems x 2 B = 256 B). So the `8` here
 * and the 16 B recommendation are the same quantity in different units.
 * The padded offset is computed as `flat + (flat / PAD_INTERVAL) * PAD_AMOUNT`
 * where `flat` is the row-major byte offset divided by `sizeof(T)`.
 *
 * The shape's `swizzle()` returns the padded *byte* offset directly so
 * downstream callers do not need to know about the layout. Storage size in
 * elements is `rows*cols + (rows*cols / PAD_INTERVAL) * PAD_AMOUNT`; `st` sizes
 * `st::data[]` directly off this via the shape's `storage_elems()`, so an
 * `st_bf<R,C>` allocation already carries the padded slab (size-preserving
 * shapes report `rows*cols` and are unaffected).
 */
// PAD_INTERVAL / PAD_AMOUNT count ELEMENTS, not bytes: <128, 8> == 16 B every 256 B at bf16.
template<int PAD_INTERVAL = 128, int PAD_AMOUNT = 8>
struct st_16x32_padded {
    static constexpr int rows          = 16;
    static constexpr int cols          = 32;
    static constexpr int pad_interval  = PAD_INTERVAL;
    static constexpr int pad_amount    = PAD_AMOUNT;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        const int r = coord.x, c = coord.y;
        const int flat = r * cols + c;
        const int padded_flat = flat + (flat / PAD_INTERVAL) * PAD_AMOUNT;
        return sizeof(_T) * padded_flat;
    }

    static constexpr int padded_elems(int total_elems) {
        return total_elems + (total_elems / PAD_INTERVAL) * PAD_AMOUNT;
    }

    // Padding interface consumed by `st` (storage sizing + LDS address map).
    __device__ __host__ __forceinline__ static constexpr int padded(int flat) {
        return flat + (flat / PAD_INTERVAL) * PAD_AMOUNT;
    }
    static constexpr int storage_elems(int total_elems) { return padded_elems(total_elems); }
};

template<typename T>
concept all = std::is_same_v<T, st_16x16> || 
              std::is_same_v<T, st_16x16_swizzled> || 
              std::is_same_v<T, st_32x32> || 
              std::is_same_v<T, st_16x32> || 
              std::is_same_v<T, st_32x16> || 
              std::is_same_v<T, st_8x32>  ||
              std::is_same_v<T, st_16x128> ||
              std::is_same_v<T, st_16x32_padded<>> ||
              std::is_same_v<T, st_16x32_padded<128, 8>> ||
              std::is_same_v<T, st_16x32_padded<256, 8>> ||
              std::is_same_v<T, st_16x32_padded<512, 8>> ||
              std::is_same_v<T, st_16x32_padded<1024, 8>> ||
              std::is_same_v<T, st_16x32_padded<64, 8>> ||
              std::is_same_v<T, st_16x32_padded<64, 4>>;

/* ----------  SHAPE KIND  ----------
 *
 * The two kinds of address map a shape can carry. They are mutually exclusive and between them
 * they decide how every op addresses a tile:
 *
 *   padded    -- size-INCREASING. Row-major with periodic padding inserted to break bank
 *                conflicts. This is the map the TDM engine deposits, which is why it is the only
 *                kind a TDM destination may have.
 *   swizzled  -- size-PRESERVING with a non-trivial within-subtile permutation. The ops must
 *                honour the permutation on both ends; hardware that writes plain row-major cannot
 *                feed one.
 *
 * A size-preserving shape whose map is the identity (`st_16x16`, `st_8x32`) is neither.
 *
 * Do not infer either kind from a shape's NAME: `st_16x32`, `st_32x16`, `st_32x32` and `st_16x128`
 * all XOR-swizzle despite lacking the `_swizzled` suffix, and `st_16x16_swizzled` swizzles at 2-byte
 * elements but is the identity at 4-byte ones. `swizzled` is therefore evaluated FROM the shape's own
 * `swizzle()` at compile time, and takes the element type, so it cannot drift from what the shape
 * actually does and cannot be wrong about a type-dependent map.
 */

/// @brief Does the shape carry the periodic-padding interface `st` and the TDM descriptor consume?
template<typename S>
concept padded = requires {
    { S::pad_interval } -> std::convertible_to<int>;
    { S::pad_amount   } -> std::convertible_to<int>;
    S::padded(0);
    S::storage_elems(0);
};

/// @brief Does the shape's map differ from plain row-major anywhere, for element type `T`?
template<typename S, typename T>
__host__ __device__ constexpr bool permutes_for() {
    for (int r = 0; r < S::rows; r++)
        for (int c = 0; c < S::cols; c++)
            if (S::template swizzle<T>(int2{r, c})
                != static_cast<uint32_t>(sizeof(T) * (r * S::cols + c)))
                return true;
    return false;
}

/// @brief A size-preserving shape with a non-trivial permutation, for element type `T`.
template<typename S, typename T>
concept swizzled = !padded<S> && permutes_for<S, T>();

/**
 * @brief Is a run of `E` elements starting at a multiple of `E` one contiguous, aligned run?
 *
 * For a padded shape `padded(f) = f + (f/PI)*PA`, so the run stays contiguous iff `PI % E == 0` and
 * aligned iff `PA % E == 0`; `st_16x32_padded<64, 4>` fails the second at bf16, which is why this is
 * computed rather than assumed. Every size-preserving swizzle here moves an offset by a multiple of
 * 8 bytes using only bits 7 and above, so it is constant across a run of at most 16 bytes -- a shape
 * with finer swizzle granularity would break that and must be added here.
 *
 * Whether a run can be moved as one sized transfer is a property of the shape, so it lives here and
 * both the LDS->global stream and the register->LDS stage ask the same question of the same code.
 */
template<typename S, int E>
__host__ __device__ constexpr bool run_is_contiguous() {
    if constexpr (padded<S>) return (S::pad_interval % E == 0) && (S::pad_amount % E == 0);
    else return true;
}

} // namespace st_shape
} // namespace ducks
} // namespace kittens