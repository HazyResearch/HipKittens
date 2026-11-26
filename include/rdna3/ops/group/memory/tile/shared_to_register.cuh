/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer data directly between shared memory and registers and back.
 */

/*
 * Row-sliced version of the warp-level path in
 * ops/warp/memory/tile/shared_to_register.cuh: warp w owns rows
 * [w*RT::height, (w+1)*RT::height) of subtiles and runs exactly the same lane
 * mapping within them. See that file for the gfx11 layouts and for why the
 * vectorized case issues two 16-byte reads.
 *
 * Note that unlike the CDNA version this goes through src.idx() / operator[]
 * rather than indexing src.data[] directly, so the shared tile's XOR swizzle is
 * applied. Raw indexing is only correct for an unswizzled layout.
 */

/**
 * @brief Collaboratively load data from a shared tile into register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    const int lane = ::kittens::laneid();
    const int l16  = lane & 15;
    const int warp_row_offset = warpid() * warp_height;
    const uint32_t src_ptr = (uint32_t)(uintptr_t)&src.data[0];

    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        const int row_base = (warp_row_offset + i) * base::tile_size_row;
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            const int col_base = j * base::tile_size_col;
            if constexpr (vectorizable) {
                #pragma unroll
                for(int b = 0; b < 2; b++) {
                    const uint32_t p = src.idx(src_ptr, {row_base + l16, col_base + 8*b});
                    const float4 v = load_shared_vec4_async(p);
                    __builtin_memcpy((void*)&dst.tiles[i][j].data[4*b], &v, sizeof(v));
                }
            }
            else {
                #pragma unroll
                for(int e = 0; e < E; e++) {
                    const int2 c = rt_base_coord<T, L>(e, lane);
                    const T val = base_types::convertor<T, U>::convert(
                        src[{row_base + c.x, col_base + c.y}]);
                    if (e & 1) dst.tiles[i][j].data[e>>1].y = val;
                    else       dst.tiles[i][j].data[e>>1].x = val;
                }
            }
        }
    }
    if constexpr (vectorizable) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}


/**
 * @brief Collaboratively store data into a shared tile from register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::st::all ST, ducks::rt::all RT>
__device__ inline static void store(ST &dst, const RT &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    const int lane = ::kittens::laneid();
    const int l16  = lane & 15;
    const int warp_row_offset = warpid() * warp_height;
    const uint32_t dst_ptr = (uint32_t)(uintptr_t)&dst.data[0];

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        const int row_base = (warp_row_offset + i) * base::tile_size_row;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int col_base = j * base::tile_size_col;
            if constexpr (vectorizable) {
                #pragma unroll
                for(int b = 0; b < 2; b++) {
                    const uint32_t p = dst.idx(dst_ptr, {row_base + l16, col_base + 8*b});
                    float4 v;
                    __builtin_memcpy(&v, (const void*)&src.tiles[i][j].data[4*b], sizeof(v));
                    store_shared_vec4(p, v);
                }
            }
            else {
                #pragma unroll
                for(int e = 0; e < E; e++) {
                    const int2 c = rt_base_coord<T, L>(e, lane);
                    const T val = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                          : src.tiles[i][j].data[e>>1].x;
                    dst[{row_base + c.x, col_base + c.y}] =
                        base_types::convertor<U, T>::convert(val);
                }
            }
        }
    }
    if constexpr (vectorizable) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}
