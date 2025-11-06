/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer data directly between shared memory and registers and back.
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
    int local_warpid = warpid();
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();

    int warp_row_offset = local_warpid * warp_height;
    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = warp_laneid%16;
        col_offset = 4*(warp_laneid/16);
    }
    else {
        row_offset = 4*(warp_laneid/16);
        col_offset = warp_laneid%16;
    }

    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        int row = (warp_row_offset + i) * dst.tile_size_row + row_offset;
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            int col = j * dst.tile_size_col + col_offset;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row, col+0}]));
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src[{row, col+2}]));
            }
            else { // handle the column-major layout
                U2 tmp[2];
                
                tmp[0] = U2{*(U*)(&src[{row+0, col}]), *(U*)(&src[{row+1, col}]) };
                tmp[1] = U2{*(U*)(&src[{row+2, col}]), *(U*)(&src[{row+3, col}]) };

                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
            }
        }
    }
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
    int local_warpid = warpid();
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();

    int warp_row_offset = local_warpid * warp_height;
    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = warp_laneid%16;
        col_offset = 4*(warp_laneid/16);
    }
    else {
        row_offset = 4*(warp_laneid/16);
        col_offset = warp_laneid%16;
    }
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = (warp_row_offset + i) * src.tile_size_row + row_offset;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j * src.tile_size_col + col_offset;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout
                *(U2*)(&dst[{row, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                *(U2*)(&dst[{row, col+2}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            }
            else { // handle the column-major layout
                U2 tmp[2];

                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);

                *(U*)(&dst[{row+0, col}]) = std::bit_cast<U>(tmp[0].x);
                *(U*)(&dst[{row+1, col}]) = std::bit_cast<U>(tmp[0].y);
                *(U*)(&dst[{row+2, col}]) = std::bit_cast<U>(tmp[1].x);
                *(U*)(&dst[{row+3, col}]) = std::bit_cast<U>(tmp[1].y);
            }            
        }
    }
}