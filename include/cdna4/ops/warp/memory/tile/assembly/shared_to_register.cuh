/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

 #pragma once

 #include <type_traits>
 
 #include "../../../../../common/common.cuh"
 #include "../../../../../types/types.cuh"
 #include "../../util/util.cuh"
 
 namespace kittens {
 // These probably need to be redone to reduce bank conflicts.
 // They currently work fine with xor layout but it should be
 // possible to reduce their bank conflicts with other layouts too.
 
 /**
  * @brief Compute the per-lane shared addresses for an art tile.
  *
  * @tparam RT The register tile type
  * @tparam ST The shared tile type
  * @param dst[out] The destination register tile.
  * @param src[in]  The source shared tile.
  * @return uint2_t for the two fp8 chunks (shared tile must be 2 KiB aligned), uint32_t otherwise.
  */
 
 template<ducks::art::all RT, ducks::st::all ST>
 __device__ inline static auto get_address(RT &dst, const ST &src) {

     if constexpr (std::is_same_v<typename RT::T, fp8e4m3>) {
        static_assert(std::is_same_v<typename ST::dtype, typename RT::T>, "register and shared tile types must match");
        static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "fp8 operand tiles are row layout");
        static_assert(std::is_same_v<typename RT::shape, rt_16x128_s> && std::is_same_v<typename ST::shape, st_16x128_s>,
                      "fp8 loads take rt_16x128 subtiles from st_16x128 shared tiles");
        static_assert(RT::rows == ST::rows && RT::cols == ST::cols && RT::width == 1, "tile shapes must match with one column subtile");
        constexpr int chunk_stride =
            RT::base_tile_elements_per_stride_group * sizeof(typename ST::dtype);
        const int laneid = kittens::laneid();
        const uint32_t base = reinterpret_cast<uintptr_t>(
            &src.data[(laneid % RT::base_tile_rows) * ST::underlying_cols + (laneid / RT::base_tile_rows) * 16]);
        return uint2_t{base ^ (((base % (16 * 128)) >> 8) << 4),
                       (base + chunk_stride) ^ ((((base + chunk_stride) % (16 * 128)) >> 8) << 4)};
     } else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
 
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        const int laneid = kittens::laneid();
    
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int shared_tile_stride = 16 * 16 * sizeof(U);
    
            const int shared_tile_offset = (laneid / 32) * shared_tile_stride;
            const int lane_col_offset = ((laneid % 32) / 16) * 8;
            const int lane_row_offset = (laneid % 16);
    
            const int lane_byte_offset = (lane_row_offset * 16 + lane_col_offset) * sizeof(U) + shared_tile_offset;
            const uint32_t addr = src_ptr + lane_byte_offset;
    
            return addr;
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int lane_col_offset = (laneid / 16) * 8;
            const int lane_row_offset = (laneid % 16);
    
            const int lane_byte_offset = (lane_row_offset * 32 + lane_col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
    
            return addr;
        } else {
            static_assert(false, "Unsupported shape");
        }
    } else {

        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        const int laneid = kittens::laneid();
    
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
        
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_swizzled_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
            const int col_offset = ((laneid % 4) * 4);
            const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 7) << 3);
    
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
            return addr;
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
            const int col_offset = ((laneid % 4) * 4);
            const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(U);
    
            const uint32_t addr = src_ptr + lane_byte_offset;
            return addr;
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
            const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
            const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
    
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
            return addr;
        } else {
            static_assert(false, "Unsupported shape");
        }
    }
 }
 
 // Load one 16-byte chunk of an fp8 subtile using addr[CHUNK] from get_address.
 template<int N, int M, int CHUNK, ducks::art::all RT, ducks::st::all ST>
 requires std::is_same_v<typename RT::T, fp8e4m3>
 __device__ inline static void load(RT &dst, const ST &src, uint32_t addr) {
    static_assert(std::is_same_v<typename RT::T, fp8e4m3> && std::is_same_v<typename ST::dtype, typename RT::T>,
                  "register and shared tile must both be fp8e4m3");
    static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "fp8 operand tiles are row layout");
    static_assert(std::is_same_v<typename RT::shape, rt_16x128_s> && std::is_same_v<typename ST::shape, st_16x128_s>,
                  "fp8 loads take rt_16x128 subtiles from st_16x128 shared tiles");
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols, "register tile and shared tile must match");
    static_assert(N >= 0 && N < RT::height && M == 0 && RT::width == 1 && (CHUNK == 0 || CHUNK == 1),
                  "subtile / chunk out of range");
    using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N>;
    static_assert(range_type::size == 8, "an rt_16x128 fp8 subtile occupies 8 registers");
    constexpr int row_stride = RT::base_tile_rows * ST::underlying_cols * sizeof(typename ST::dtype);
    macros::ds_read_b128<range_type::lo + 4 * CHUNK>(addr, N * row_stride);
 }

 template<int N, int M, ducks::art::all RT, ducks::st::all ST>
 requires std::is_same_v<typename RT::T, fp8e4m3>
 __device__ inline static void load(RT &dst, const ST &src, uint2_t addr) {
    load<N, M, 0>(dst, src, addr[0]);
    load<N, M, 1>(dst, src, addr[1]);
 }

 template<int N, int M, ducks::art::all RT, ducks::st::all ST>
 __device__ inline static void load(RT &dst, const ST &src, uint32_t addr) {
 
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            constexpr int shared_tile_stride = 16 * 16 * sizeof(U);
            constexpr int row_stride = shared_tile_stride * ST::underlying_subtiles_per_row;
            constexpr int register_tile_stride = shared_tile_stride * 2;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b128 requires 4 consecutive registers");
            
            constexpr int offset = N * row_stride + M * register_tile_stride;
            macros::ds_read_b128<range_type::lo>(addr, offset);
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            constexpr int tile_stride = 16 * 32 * sizeof(U);
            constexpr int row_stride = tile_stride * ST::underlying_subtiles_per_row;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b128 requires 4 consecutive registers");
    
            constexpr int offset = N * row_stride + M * tile_stride;
            macros::ds_read_b128<range_type::lo>(addr, offset); 
        } else {
            static_assert(false, "Unsupported shape");
        }
    } else {
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_swizzled_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            constexpr int tile_stride = 16 * 16 * sizeof(U);
            constexpr int shared_tile_offset = 16 * 16 * ST::underlying_subtiles_per_row * sizeof(U);
            constexpr int row_stride = shared_tile_offset * 2;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
            
            constexpr int offset_0 = N * row_stride + M * tile_stride;
            constexpr int offset_1 = N * row_stride + M * tile_stride + shared_tile_offset;
            macros::ds_read_b64_tr_b16<range_type::lo>(addr, offset_0);
            macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, offset_1);
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            constexpr int tile_stride = 16 * 16 * sizeof(U);
            constexpr int row_stride = tile_stride * ST::underlying_subtiles_per_row;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
            
            macros::ds_read_b64_tr_b16<range_type::lo>(addr, N * (2 * row_stride) + M * tile_stride);
            macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, N * (2 * row_stride) + M * tile_stride + row_stride);
    
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            constexpr int tile_stride = 16 * 32 * sizeof(U);
            constexpr int row_stride = tile_stride * ST::underlying_subtiles_per_row;
    
            using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
            static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
            
            constexpr int offset_0 = N * row_stride + M * tile_stride;
            constexpr int offset_1 = N * row_stride + M * tile_stride + 4 * 32 * sizeof(U);
            macros::ds_read_b64_tr_b16<range_type::lo>(addr, offset_0);
            macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, offset_1);
        } else {
            static_assert(false, "Unsupported shape");
        }
    }
 }
 
 
 template<ducks::art::all RT, ducks::st::all ST>
 __device__ inline static void load(RT &dst, const ST &src) {
 
    if constexpr (std::is_same_v<typename RT::T, fp8e4m3>) {
        const auto addr = get_address(dst, src);
        [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
            (load<Ns, 0>(dst, src, addr), ...);
        }(std::make_index_sequence<RT::height>{});
    } else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        const int laneid = kittens::laneid();
    
        const int row_offset = laneid % dst.base_tile_rows;
        const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);
    
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int shared_tile_stride = 16 * 16 * sizeof(U);
            const int row_stride = shared_tile_stride * ST::underlying_subtiles_per_row;
            const int register_tile_stride = shared_tile_stride * 2;
    
            const int shared_tile_offset = (laneid / 32) * shared_tile_stride;
            const int lane_col_offset = ((laneid % 32) / 16) * 8;
            const int lane_row_offset = (laneid % 16);
    
            const int lane_byte_offset = (lane_row_offset * 16 + lane_col_offset) * sizeof(U) + shared_tile_offset;
            const uint32_t addr = src_ptr + lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
                
                const int offset = N * row_stride + M * register_tile_stride;
    
                macros::ds_read_b128<range_type::lo>(addr, offset);
            };
    
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int tile_stride = 16 * 32 * sizeof(U);
            const int row_stride = tile_stride * ST::underlying_subtiles_per_row;
            const int lane_col_offset = (laneid / 16) * 8;
            const int lane_row_offset = (laneid % 16);
    
            const int lane_byte_offset = (lane_row_offset * 32 + lane_col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b128 requires 4 consecutive registers");
                const int offset = N * row_stride + M * tile_stride;
                macros::ds_read_b128<range_type::lo>(addr, offset); 
            };
    
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else {
            static_assert(false, "Unsupported shape");
        }
    } else {
        static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
        static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    
        using T2 = RT::dtype;
        using T  = base_types::packing<T2>::unpacked_type;
        using U  = ST::dtype;
        using U2 = base_types::packing<U >::packed_type;
        constexpr int packing = base_types::packing<typename RT::dtype>::num();
    
        const int laneid = kittens::laneid();
    
        const int row_offset = ((laneid % 16) / 4) + ((laneid / dst.base_tile_cols) * dst.base_tile_stride);
        const int col_offset = ((laneid % 4) * 4) + (16 * ((laneid % dst.base_tile_cols) / 16));
    
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
        
        if constexpr (std::is_same_v<typename ST::shape, st_16x16_swizzled_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            const int tile_stride = 16 * 16 * sizeof(U);
            const int shared_tile_offset = 16 * 16 * ST::underlying_subtiles_per_row * sizeof(U);
            const int row_stride = shared_tile_offset * 2;
            const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
            const int col_offset = ((laneid % 4) * 4);
            const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 7) << 3);
    
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
                
                const int offset_0 = N * row_stride + M * tile_stride;
                const int offset_1 = N * row_stride + M * tile_stride + shared_tile_offset;
                macros::ds_read_b64_tr_b16<range_type::lo>(addr, offset_0);
                macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, offset_1);
            };
    
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x16_s> && std::is_same_v<typename RT::shape, rt_32x16_4_s>) {
            const int tile_stride = 16 * 16 * sizeof(U);
            const int row_stride = tile_stride * ST::underlying_subtiles_per_row;
            const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
            const int col_offset = ((laneid % 4) * 4);
            const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(U);
    
            const uint32_t addr = src_ptr + lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
                
                macros::ds_read_b64_tr_b16<range_type::lo>(addr, N * (2 * row_stride) + M * tile_stride);
                macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, N * (2 * row_stride) + M * tile_stride + row_stride);
            };
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else if constexpr (std::is_same_v<typename ST::shape, st_16x32_s> && std::is_same_v<typename RT::shape, rt_16x32_s>) {
            const int tile_stride = 16 * 32 * sizeof(U);
            const int row_stride = tile_stride * ST::underlying_subtiles_per_row;
            const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
            const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
            const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(U);
            const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
    
            const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
    
            auto perform_load_at = [&]<int N, int M>() {
                using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
                static_assert(range_type::lo + 3 == range_type::hi, "ds_read_b64_tr_b16 requires 4 consecutive registers");
                
                const int offset_0 = N * row_stride + M * tile_stride;
                const int offset_1 = N * row_stride + M * tile_stride + 4 * 32 * sizeof(U);
                macros::ds_read_b64_tr_b16<range_type::lo>(addr, offset_0);
                macros::ds_read_b64_tr_b16<range_type::lo + 2>(addr, offset_1);
            };
    
            // Compile-time nested loops over N and M
            [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                ([&]<std::size_t N>() {
                    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                        ([&]<std::size_t M>() {
                            perform_load_at.template operator()<N, M>();
                        }.template operator()<Ms>(), ...);
                    }(std::make_index_sequence<RT::width>{});
                }.template operator()<Ns>(), ...);
            }(std::make_index_sequence<RT::height>{});
        } else {
            static_assert(false, "Unsupported shape");
        }
    }
 }
 
 /**
  * @brief Store data into a shared tile from a register tile.
  *
  * @tparam RT The register tile type
  * @tparam ST The shared tile type
  * @param dst[out] The destination shared tile.
  * @param src[in]  The source register tile.
  */
 
 template<ducks::art::all RT, ducks::st::all ST>
 __device__ inline static uint32_t get_address(ST &dst, const RT &src) {
 
     static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "RT must be a row-major layout tile");
     static_assert(std::is_same_v<typename ST::shape, st_16x16_swizzled_s>, "Only st_16x16_swizzled_s is supported");
     static_assert(std::is_same_v<typename RT::shape, rt_16x16_s>, "Only rt_16x16_s is supported");
 
     using T2 = RT::dtype;
     using T  = base_types::packing<T2>::unpacked_type;
     using U  = ST::dtype;
     using U2 = base_types::packing<U >::packed_type;
 
     const int laneid = kittens::laneid();
     const int col_offset = (laneid / 16) * 4;
     const int row_offset = (laneid % 16);
 
     const int lane_byte_offset = (row_offset * ST::underlying_subtile_cols + col_offset) * sizeof(U);
     const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 7) << 3);
     const uint32_t addr = reinterpret_cast<uintptr_t>(&dst.data[0]) + swizzled_lane_byte_offset;
 
     return addr;
 }
 
 template<int N, int M, ducks::art::all RT, ducks::st::all ST>
 __device__ inline static void store(ST &dst, const RT &src, uint32_t addr) {
     static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "RT must be a row-major layout tile");
     static_assert(std::is_same_v<typename ST::shape, st_16x16_swizzled_s>, "Only st_16x16_swizzled_s is supported");
     static_assert(std::is_same_v<typename RT::shape, rt_16x16_s>, "Only rt_16x16_s is supported");
 
     using T2 = RT::dtype;
     using T  = base_types::packing<T2>::unpacked_type;
     using U  = ST::dtype;
     using U2 = base_types::packing<U >::packed_type;
 
     constexpr int tile_stride = ST::underlying_subtile_rows * ST::underlying_subtile_cols;
     constexpr int row_stride = tile_stride * ST::underlying_subtiles_per_row;
 
     using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
     static_assert(range_type::lo + 1 == range_type::hi, "ds_write_b64 requires 2 consecutive registers");
 
     constexpr int offset = (N * row_stride + M * tile_stride) * sizeof(U);
     macros::ds_write_b64<range_type::lo>(addr, offset);
 }
 
 template<ducks::art::all RT, ducks::st::all ST>
 __device__ inline static void store(ST &dst, const RT &src) {
     static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "RT must be a row-major layout tile");
     static_assert(std::is_same_v<typename ST::shape, st_16x16_swizzled_s>, "Only st_16x16_swizzled_s is supported");
     static_assert(std::is_same_v<typename RT::shape, rt_16x16_s>, "Only rt_16x16_s is supported");
 
     using T2 = RT::dtype;
     using T  = base_types::packing<T2>::unpacked_type;
     using U  = ST::dtype;
     using U2 = base_types::packing<U >::packed_type;
 
     const int laneid = kittens::laneid();
     const int col_offset = (laneid / 16) * 4;
     const int row_offset = (laneid % 16);
 
     const int lane_byte_offset = (row_offset * ST::underlying_subtile_cols + col_offset) * sizeof(U);
     const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 7) << 3);
     const uint32_t addr = reinterpret_cast<uintptr_t>(&dst.data[0]) + swizzled_lane_byte_offset;
 
     const int tile_stride = ST::underlying_subtile_rows * ST::underlying_subtile_cols;
     const int row_stride = tile_stride * ST::underlying_subtiles_per_row;
 
     auto perform_store_at = [&]<int N, int M>() {
         using range_type = ducks::art::get_nth_range_t<typename RT::register_ranges, N * RT::width + M>;
 
         static_assert(range_type::lo + 1 == range_type::hi, "ds_write_b64 requires 2 consecutive registers");
 
         const int offset = (N * row_stride + M * tile_stride) * sizeof(U);
 
         macros::ds_write_b64<range_type::lo>(addr, offset);
     };
 
     // Compile-time nested loops over N and M
     [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
         ([&]<std::size_t N>() {
             [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                 ([&]<std::size_t M>() {
                     perform_store_at.template operator()<N, M>();
                 }.template operator()<Ms>(), ...);
             }(std::make_index_sequence<ST::subtiles_per_row>{});
         }.template operator()<Ns>(), ...);
     }(std::make_index_sequence<ST::subtiles_per_col>{});
 }
 
}
