/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/* ----------  LAYOUT SWAPS  ---------- */

/**
 * @brief Swaps the layout of a register base tile.
 *
 * This function swaps the layout of a register base tile by performing a series of layout swaps
 * on its constituent bf16_2 elements. It is used to change the data layout within a register tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register base tile where the result will be stored.
 * @param src[in] Reference to the source register base tile to be swapped.
 */
 #ifdef KITTENS_CDNA4
template<typename T, ducks::rt_layout::accum layout>
__device__ inline void swap_layout(rt_base<T, typename ducks::rt_layout::col> &dst, const rt_base<T, layout> &src) {
    
    if constexpr (std::is_same_v<T, bf16>) {
        uint32_t* src_ptr = (uint32_t*)&src.data[0];
        uint32_t* dst_ptr = (uint32_t*)&dst.data[0];
        
        // Load source values
        uint32_t temp0 = src_ptr[0], temp1 = src_ptr[1], temp2 = src_ptr[2], temp3 = src_ptr[3];
        uint32_t temp4 = src_ptr[4], temp5 = src_ptr[5], temp6 = src_ptr[6], temp7 = src_ptr[7];
        
        // Try using single calls with immediate access to avoid bound_ctrl
        // This should generate e32 instead of e64
        dst_ptr[0] = __builtin_amdgcn_permlane32_swap(temp0, temp2, false, false)[0];
        dst_ptr[1] = __builtin_amdgcn_permlane32_swap(temp1, temp3, false, false)[0];
        dst_ptr[4] = __builtin_amdgcn_permlane32_swap(temp4, temp6, false, false)[0];
        dst_ptr[5] = __builtin_amdgcn_permlane32_swap(temp5, temp7, false, false)[0];
        
        // For the second elements, use the swap result from the same lanes but different element
        dst_ptr[2] = __builtin_amdgcn_permlane32_swap(temp0, temp2, false, false)[1];
        dst_ptr[3] = __builtin_amdgcn_permlane32_swap(temp1, temp3, false, false)[1];
        dst_ptr[6] = __builtin_amdgcn_permlane32_swap(temp4, temp6, false, false)[1];
        dst_ptr[7] = __builtin_amdgcn_permlane32_swap(temp5, temp7, false, false)[1];
    }
    else {
        // Keep original for other types
        typedef unsigned int  uint32_t;
        typedef uint32_t      uint2_t __attribute__((ext_vector_type(2)));
 
         T src_tmp[16] = {
             src.data[0].x, src.data[0].y,
             src.data[1].x, src.data[1].y,
             src.data[2].x, src.data[2].y,
             src.data[3].x, src.data[3].y,
             src.data[4].x, src.data[4].y,
             src.data[5].x, src.data[5].y,
             src.data[6].x, src.data[6].y,
             src.data[7].x, src.data[7].y,
         };
 
         T dst_tmp[16];
        #pragma unroll
        for(int k = 0; k < 8; k++) {
            const int kk = (k / 4) * 8 + (k % 4);
            if constexpr (std::is_same_v<T, float>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(src_tmp[kk]), __float_as_uint(src_tmp[kk + 4]), false, true);
                dst_tmp[kk] = __uint_as_float(res.x);
                dst_tmp[kk + 4] = __uint_as_float(res.y);
            }
            else if constexpr (std::is_same_v<T, half>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(src_tmp[kk]), __half_as_ushort(src_tmp[kk + 4]), false, true);
                dst_tmp[kk] = __ushort_as_half(res.x);
                dst_tmp[kk + 4] = __ushort_as_half(res.y);
            }
        }
        memcpy(&dst.data[0], &dst_tmp[0], sizeof(dst.data));
    }
}

  template<typename T, ducks::rt_layout::accum layout>
 __device__ inline void swap_layout(rt_base<T, typename ducks::rt_layout::row> &dst, const rt_base<T, layout> &src) {
    const int lane = laneid();

    int block_src_trans = 32*((lane%8)/4) + 8*(lane/32); 
    int thread_offset = lane % 4;
    int block_offset = ((lane % 16) / 8) * 4;
    int send_offset = ((lane % 8) / 4) * 4;
    int to_flip = ((lane % 32) / 16) * 8;
    int or_not_to_flip = ((lane % 32) / 16) * 16;
    

    T src_tmp[16] = {
        src.data[0].x, src.data[0].y,
        src.data[1].x, src.data[1].y,
        src.data[2].x, src.data[2].y,
        src.data[3].x, src.data[3].y,
        src.data[4].x, src.data[4].y,
        src.data[5].x, src.data[5].y,
        src.data[6].x, src.data[6].y,
        src.data[7].x, src.data[7].y,
    };

    T dst_tmp[16];
    #pragma unroll
    for(int k = 0; k < 16; k++) {
        if constexpr (std::is_same_v<T, bf16>) {
            int that_is_the_question = (k / 8) * 24;
            int setting = thread_offset^block_offset^k^to_flip;
            int sending = thread_offset^send_offset^k^to_flip;
            int from = block_src_trans + thread_offset^block_offset^k^or_not_to_flip^that_is_the_question;
            dst_tmp[setting] = __float2bfloat16(__shfl(__bfloat162float(src_tmp[sending]), from));
        }
        else {
            int that_is_the_question = (k / 8) * 24;
            int setting = thread_offset^block_offset^k^to_flip;
            int sending = thread_offset^send_offset^k^to_flip;
            int from = block_src_trans + thread_offset^block_offset^k^or_not_to_flip^that_is_the_question;
            dst_tmp[setting] = __shfl(src_tmp[sending], from);
        }
    }

    memcpy(&dst.data[0], &dst_tmp[0], sizeof(dst.data));
 }
 #endif

#ifdef KITTENS_CDNA4
template<typename T, ducks::rt_layout::classic layout>
__device__ inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    int lane = laneid();

    int to_flip = ((lane % 32) / 16) * 8;
    int or_not_to_flip = ((lane % 32) / 16) * 16;
    int block_src_trans = 32*((lane%16)/8) + 8*(lane/32); 
    int block_offset = lane%8; 

    T src_tmp[16] = {
        src.data[0].x, src.data[0].y,
        src.data[1].x, src.data[1].y,
        src.data[2].x, src.data[2].y,
        src.data[3].x, src.data[3].y,
        src.data[4].x, src.data[4].y,
        src.data[5].x, src.data[5].y,
        src.data[6].x, src.data[6].y,
        src.data[7].x, src.data[7].y,
    };

    T dst_tmp[16];
    #pragma unroll
    for(int k = 0; k < 16; k++) {
        int that_is_the_question = (k / 8) * 24;
        if constexpr (std::is_same_v<T, bf16>) {
            dst_tmp[block_offset^k^to_flip] = __float2bfloat16(__shfl(__bfloat162float(src_tmp[block_offset^k^to_flip]), (block_src_trans + block_offset^k^or_not_to_flip^that_is_the_question)));
            // printf("Thread: %d, Setting: %d, Sending: %d, From: %d\n", lane, block_offset^k^to_flip, block_offset^k^to_flip, block_src_trans + block_offset^k^or_not_to_flip^that_is_the_question);
        }
        else {
            dst_tmp[block_offset^k^to_flip] = __shfl(src_tmp[block_offset^k^to_flip], block_src_trans + block_offset^k^or_not_to_flip^that_is_the_question);
        }
    }

    dst.data[0].x = dst_tmp[0];
    dst.data[0].y = dst_tmp[1];
    dst.data[1].x = dst_tmp[2];
    dst.data[1].y = dst_tmp[3];
    dst.data[2].x = dst_tmp[4];
    dst.data[2].y = dst_tmp[5];
    dst.data[3].x = dst_tmp[6];
    dst.data[3].y = dst_tmp[7];
    dst.data[4].x = dst_tmp[8];
    dst.data[4].y = dst_tmp[9];
    dst.data[5].x = dst_tmp[10];
    dst.data[5].y = dst_tmp[11];
    dst.data[6].x = dst_tmp[12];
    dst.data[6].y = dst_tmp[13];
    dst.data[7].x = dst_tmp[14];
    dst.data[7].y = dst_tmp[15];
}
#else
template<typename T, ducks::rt_layout::all layout>
__device__ inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    int lane = laneid();

    int block_src_trans = 16 * ((lane % 16) / 4) + 4 * (lane / 16);
    int block_offset    = lane % 4;

    T src_tmp[4] = {
        src.data[0].x, src.data[0].y,
        src.data[1].x, src.data[1].y
    };

    T dst_tmp[4];
    #pragma unroll
    for(int k = 0; k < 4; k++) {
        if constexpr (std::is_same_v<T, bf16>) {
            dst_tmp[block_offset^k] = __float2bfloat16(__shfl(__bfloat162float(src_tmp[block_offset^k]), block_src_trans + block_offset^k));
        }
        else {
            dst_tmp[block_offset^k] = __shfl(src_tmp[block_offset^k], block_src_trans + block_offset^k);
        }
    }

    dst.data[0].x = dst_tmp[0];
    dst.data[0].y = dst_tmp[1];
    dst.data[1].x = dst_tmp[2];
    dst.data[1].y = dst_tmp[3];
}
#endif

/**
 * @brief Swaps the layout of a register tile.
 *
 * This function swaps the layout of a register tile by iterating over its height and width
 * and performing layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register tile where the result will be stored.
 * @param src[in] Reference to the source register tile to be swapped.
 */
#ifdef KITTENS_CDNA4
template<typename T2, int _height, int _width, ducks::rt_layout::accum layout>
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::col> &dst, const rt<T2, _height, _width, layout> &src) {

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

template<typename T2, int _height, int _width, ducks::rt_layout::accum layout>
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::row> &dst, const rt<T2, _height, _width, layout> &src) {

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}
#endif

#ifdef KITTENS_CDNA4
template<typename T2, int _height, int _width, ducks::rt_layout::classic layout>
#else
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
#endif
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/**
 * @brief Swaps the layout of a register base tile in place.
 *
 * This function swaps the layout of a register base tile in place by casting it to the
 * transposed layout type and then performing the layout swap.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register base tile to be swapped in place.
 * @return A reference to the swapped register base tile.
 */
#ifdef KITTENS_CDNA4
template<typename T2, ducks::rt_layout::classic layout>
#else
template<typename T2, ducks::rt_layout::all layout>
#endif
__device__ inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}

#ifdef KITTENS_CDNA4
template<typename DesiredLayout, typename T2, ducks::rt_layout::accum layout>
__device__ inline rt_base<T2, DesiredLayout>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, DesiredLayout> &dst = *(rt_base<T2, DesiredLayout>*)(&src);
    swap_layout(dst, src);
    return dst;
}
template<typename DstT, typename SrcT, ducks::rt_layout::accum layout>
__device__ inline rt_base<DstT, typename ducks::rt_layout::row>& swap_layout_inplace_copy(const rt_base<SrcT, layout> &src) {
    rt_base<DstT, typename ducks::rt_layout::row> &dst = *(rt_base<DstT, typename ducks::rt_layout::row>*)(&src);
    swap_layout_copy(dst, src);
    return dst;
}
#endif
/**
 * @brief Swaps the layout of a register tile in place.
 *
 * This function swaps the layout of a register tile in place by iterating over its height and width
 * and performing in-place layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */
#ifdef KITTENS_CDNA4
template<typename T2, int _rows, int _cols, ducks::rt_layout::classic layout>
#else
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
#endif
__device__ static inline rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(rt<T2, _rows, _cols, layout> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
}

#ifdef KITTENS_CDNA4
template<typename DesiredLayout, typename T2, int _rows, int _cols, ducks::rt_layout::accum layout>
__device__ static inline rt<T2, _rows, _cols, DesiredLayout>& swap_layout_inplace(rt<T2, _rows, _cols, layout> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace<DesiredLayout>(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _rows, _cols, DesiredLayout>*)(&tile);
}

template<typename DstT, typename SrcT, int _rows, int _cols, ducks::rt_layout::accum layout>
__device__ static inline rt<DstT, _rows, _cols, typename ducks::rt_layout::row>& swap_layout_inplace_copy(rt<SrcT, _rows, _cols, layout> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace_copy<DstT, SrcT>(tile.tiles[i][j]);
        }
    }
    return *(rt<DstT, _rows, _cols, typename ducks::rt_layout::row>*)(&tile);
}
#endif

/* ----------  TRANSPOSE  ---------- */

/**
 * @brief Transposes a register base tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register base tile to be transposed.
 */
template<typename T, ducks::rt_layout::all layout>
__device__ inline void transpose(rt_base<T, layout> &dst, const rt_base<T, layout> &src) {
    int lane = laneid();
    int block_src_trans = 16*((lane%16)/4) + 4*(lane/16);
    int block_offset = lane%4;

    #ifdef KITTENS_CDNA4
    static_assert(0, "transpose is not supported on CDNA4");
    #endif

    T src_tmp[4] = {
        src.data[0].x, src.data[0].y,
        src.data[1].x, src.data[1].y
    };

    T dst_tmp[4];

    #pragma unroll
    for(int k = 0; k < 4; k++) {
        if constexpr (std::is_same_v<T, bf16>) {
            dst_tmp[block_offset^k] = __float2bfloat16(__shfl(__bfloat162float(src_tmp[block_offset^k]), block_src_trans + block_offset^k));
        }
        else {
            dst_tmp[block_offset^k] = __shfl(src_tmp[block_offset^k], block_src_trans + block_offset^k);
        }
    }

    dst.data[0].x = dst_tmp[0];
    dst.data[0].y = dst_tmp[1];
    dst.data[1].x = dst_tmp[2];
    dst.data[1].y = dst_tmp[3];
}
/**
 * @brief Transposes a register tile.
 * 
 * This function is marked "sep", which means that the registers underlying dst MUST be separate
 * from the registers underlying src.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the src register tile, and the width of the dst tile.
 * @tparam _width The width of the src register tile, and the height of the dst tile.
 * @tparam layout The layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register tile to be transposed.
 */
template<ducks::rt::all RT>
__device__ static inline void transpose_sep(RT &dst, const rt<typename RT::T, RT::cols, RT::rows, typename RT::layout> &src) {
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            transpose(dst.tiles[i][j], src.tiles[j][i]);
        }
    }
}

/**
 * @brief Transposes a register base tile in-place.
 *
 * @tparam T2 The data type of the register base tile elements.
 * @tparam layout The current layout of the register base tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    transpose(src, src);
    return src;
}
/**
 * @brief Transposes a square register tile in-place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height (in units of 16) of the src register tile, and the width of the dst tile. (Must be the same as _width.)
 * @tparam _width The width (in units of 16) of the src register tile, and the height of the dst tile. (Must be the same as _height.)
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, layout>& transpose_inplace(rt<T2, _rows, _cols, layout> &tile) {
    static_assert(_cols == _rows, "in-place register tile transpose is only allowed for square tiles.");
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < i; j++) {
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        transpose_inplace(tile.tiles[i][i]);
    }
    return tile;
}

/* ----------  TYPE SWAPS  ---------- */

/**
 * @brief Copies a register base tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam layout The current layout of the register base tile.
 * @param[out] dst A reference to the destination register base tile.
 * @param[in] src A reference to the source register base tile.
 */
 template<typename T, typename U, ducks::rt_layout::all layout>
 __device__ static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
     using T2 = typename base_types::packing<T>::packed_type;
     using U2 = typename base_types::packing<U>::packed_type;
     #pragma unroll
     for(int k = 0; k < dst.packed_per_thread; k++) {
         dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
     }
 }


/**
* @brief Copies a register tile, converting the underlying type if necessary.
*
* @tparam T2 The data type of the destination register elements.
* @tparam U2 The data type of the source register elements.
* @tparam _height The height (in units of 16) of the register tiles.
* @tparam _width The width (in units of 16) of the register tiles.
* @tparam layout The current layout of the register tile.
* @param[out] dst A reference to the destination register tile.
* @param[in] src A reference to the source register tile.
*/
#ifdef KITTENS_CDNA4
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {

    int condition = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0);
    if constexpr ((std::is_same_v<U2, float> && std::is_same_v<T2, fp6_e2m3>)) {
        if (condition) {
            printf("Copying float to fp6_e2m3\n");
        }
    } 
    else if constexpr (std::is_same_v<U2, fp6_e2m3> && std::is_same_v<T2, float>) {
        if (condition) {
            printf("Copying fp6_e2m3 to float\n");
        }
    }
    // default case where the layouts map 1:1 in thread ownership logic
    else {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                copy(dst.tiles[i][j], src.tiles[i][j]);
            }
        }
    }
}
#else
 template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
 __device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
     #pragma unroll
     for(int i = 0; i < dst.height; i++) {
         #pragma unroll
         for(int j = 0; j < dst.width; j++) {
             copy(dst.tiles[i][j], src.tiles[i][j]);
         }
     }
 }
#endif



/* ----------  CAUSAL  ---------- */

/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * This function modifies a square register tile in-place to make it causal. All elements
 * above the main diagonal are set to zero, while elements on or below the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j < i) { // below the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > i) { // above the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint64_t MASKS[4] = {0xF000FF00FFF0FFFF, 0xE000FE00FFE0FFFE,
                                               0xC000FC00FFC0FFFC, 0x8000F800FF80FFF8}; // magic numbers for on-diagonal core matrices

                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    if ((MASKS[k * 2] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                    }
                    else {
                        dst.tiles[i][j].data[k].x = val;
                    }
                    if ((MASKS[k * 2 + 1] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                    else {
                        dst.tiles[i][j].data[k].y = val;
                    }
                }
            }
            // __syncwarp();
        }
    }
}

/**
 * @brief Makes a square register tile anti-causal by zeroing elements below the main diagonal.
 *
 * This function modifies a square register tile in-place to make it anti-causal. All elements
 * below the main diagonal are set to zero, while elements on or above the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void make_causal_t(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if(j > i) { // above the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j < i) { // below the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint64_t MASKS[4] = {0x1FFF01FF001F0001, 0x3FFF03FF003F0003,
                                               0x7FFF07FF007F0007, 0xFFFF0FFF00FF000F}; // magic numbers for on-diagonal core matrices

                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    if ((MASKS[k * 2] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                    }
                    else {
                        dst.tiles[i][j].data[k].x = val;
                    }
                    if ((MASKS[k * 2 + 1] >> laneid()) & 1) {
                        dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                    else {
                        dst.tiles[i][j].data[k].y = val;
                    }
                }
                
            }
            // __syncwarp();
        }
    }
}

/* ----------  TRIANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile triangular by zeroing elements above the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int lane = laneid();
    const int row = lane % 16;
    const int col = 4 * (lane / 16);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx   = (i * dst.tile_size_row) + row;
                const int global_col_idx_x = (j * dst.tile_size_col) + col + 2 * k;
                const int global_col_idx_y = (j * dst.tile_size_col) + col + 2 * k + 1;

                if (global_row_idx < row_idx) { dst.tiles[i][j].data[k] = packed_val; }
                else {
                    if (global_col_idx_x <= global_row_idx - row_idx) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                    else                                              { dst.tiles[i][j].data[k].x = val; }

                    if (global_col_idx_y <= global_row_idx - row_idx) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                    else                                              { dst.tiles[i][j].data[k].y = val; }
                }
            }
        }
        // __syncwarp();
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    
    const int lane = laneid();
    const int row = 4 * (lane / 16);
    const int col = lane % 16;
    
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx_x = (i * dst.tile_size_row) + row + 2 * k;
                const int global_row_idx_y = (i * dst.tile_size_row) + row + 2 * k + 1;
                const int global_col_idx   = (j * dst.tile_size_col) + col;

                if (global_row_idx_x < row_idx) { dst.tiles[i][j].data[k].x = val; }
                else { 
                    if (global_col_idx <= global_row_idx_x - row_idx) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                    else                                              { dst.tiles[i][j].data[k].x = val; }
                }

                if (global_row_idx_y < row_idx) { dst.tiles[i][j].data[k].y = val; }
                else { 
                    if (global_col_idx <= global_row_idx_y - row_idx) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                    else                                              { dst.tiles[i][j].data[k].y = val; }
                }
            }
        }
        // __syncwarp();
    }
}

/**
 * @brief Makes a register tile triangular by zeroing elements below the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int lane = laneid();
    const int row = lane % 16;
    const int col = 4 * (lane / 16);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx   = (i * dst.tile_size_row) + row;
                const int global_col_idx_x = (j * dst.tile_size_col) + col + 2 * k;
                const int global_col_idx_y = (j * dst.tile_size_col) + col + 2 * k + 1;

                if (global_row_idx < row_idx) { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
                else {
                    if (global_col_idx_x < global_row_idx - row_idx) { dst.tiles[i][j].data[k].x = val; }
                    else                                             { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }

                    if (global_col_idx_y < global_row_idx - row_idx) { dst.tiles[i][j].data[k].y = val; }
                    else                                             { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                }
            }
        }
        // __syncwarp();
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void triu(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    
    const int lane = laneid();
    const int row = 4 * (lane / 16);
    const int col = lane % 16;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int global_row_idx_x = (i * dst.tile_size_row) + row + 2 * k;
                const int global_row_idx_y = (i * dst.tile_size_row) + row + 2 * k + 1;
                const int global_col_idx   = (j * dst.tile_size_col) + col;

                if (global_row_idx_x < row_idx) { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                else                            { 
                    if (global_col_idx < global_row_idx_x - row_idx) { dst.tiles[i][j].data[k].x = val; }
                    else                                             { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                }

                if (global_row_idx_y < row_idx) { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                else                            { 
                    if (global_col_idx < global_row_idx_y - row_idx) { dst.tiles[i][j].data[k].y = val; }
                    else                                             { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
                }
            }
        }
        // __syncwarp();
    }
}

/* ----------  RECTANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile right filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill from and onwards to the right.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx >= dst.cols) return;

    const int col = 4 * (laneid() / 16);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int col_idx_x = (j * dst.tile_size_col) + col + 2 * k;
                const int col_idx_y = (j * dst.tile_size_col) + col + 2 * k + 1;
                if (col_idx_x >= col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (col_idx_y >= col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    
    const int col = laneid() % 16;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int t_col_idx = (j * dst.tile_size_col) + col; 
                if (t_col_idx >= col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        // __syncwarp();
    }
}

/**
 * @brief Makes a register tile left filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill to the left (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx <= 0) return;

    const int col = 4 * (laneid() / 16);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int col_idx_x = (j * dst.tile_size_col) + col + 2 * k;
                const int col_idx_y = (j * dst.tile_size_col) + col + 2 * k + 1;
                if (col_idx_x < col_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                      { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (col_idx_y < col_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                      { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int col = laneid() % 16;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int thread_col = (j * dst.tile_size_col) + col;
                if (thread_col < col_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
        // __syncwarp();
    }
}

/**
 * @brief Makes a register tile upper filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill to, from the top (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx <= 0) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int row = laneid() % 16;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int thread_row = (i * dst.tile_size_row) + row;
                if (thread_row < row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                       { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const int row = 4 * (laneid() / 16);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int row_idx_x = (i * dst.tile_size_row) + row + 2 * k;
                const int row_idx_y = (i * dst.tile_size_row) + row + 2 * k + 1;
                if (row_idx_x < row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                      { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (row_idx_y < row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                      { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}

/**
 * @brief Makes a register tile lower filled with a given value.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill from and onwards to the bottom of the tile (inclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx >= dst.rows) return;
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);

    const int row = laneid() % 16;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int thread_row = (i * dst.tile_size_row) + row;
                if (thread_row >= row_idx)  { dst.tiles[i][j].data[k] = packed_val; }
                else                        { dst.tiles[i][j].data[k] = src.tiles[i][j].data[k]; }
            }
        }
    }
}
template<ducks::rt::col_layout RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const int row = 4 * (laneid() / 16);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.packed_per_tile; k++) {
                const int row_idx_x = (i * dst.tile_size_row) + row + 2 * k;
                const int row_idx_y = (i * dst.tile_size_row) + row + 2 * k + 1;
                if (row_idx_x >= row_idx)  { dst.tiles[i][j].data[k].x = val; }
                else                       { dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x; }
                if (row_idx_y >= row_idx)  { dst.tiles[i][j].data[k].y = val; }
                else                       { dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y; }
            }
        }
    }
}


/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam RT The type of the input tile, which must satisfy the ducks::rt::all concept.
* @param src The input tile.
* @param idx The coord of the subtile.
* @return A reference to the subtile.
*
* @note The subtile height must evenly divide the tile height.
*/
template<int subtile_rows, ducks::rt::all RT>
__device__ inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    using T = typename RT::T;
    static_assert(RT::height % (subtile_rows / TILE_ROW_DIM<T>) == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout>&>(
        src.tiles[idx*(subtile_rows / TILE_ROW_DIM<T>)]
    );
}

}
